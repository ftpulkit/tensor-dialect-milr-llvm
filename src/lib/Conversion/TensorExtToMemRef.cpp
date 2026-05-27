//===- TensorExtToMemRef.cpp - Lower tensor_ext -> memref/scf/arith ------===//
#include "TensorDialect/Passes.h"
#include "TensorDialect/TensorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tensor_ext;

//===----------------------------------------------------------------------===//
// Helper
//===----------------------------------------------------------------------===//

static MemRefType toMemRef(ArrayType arr) {
  return MemRefType::get(arr.getShape(), arr.getElementType());
}

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

void mlir::tensor_ext::populateTensorExtTypeConverter(TypeConverter &tc) {
  tc.addConversion([](Type t) -> std::optional<Type> { return t; });
  tc.addConversion([](ArrayType arr) -> std::optional<Type> {
    return toMemRef(arr);
  });
  auto unrealizedCast = [](OpBuilder &b, Type toType, ValueRange inputs,
                            Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;
    return b.create<UnrealizedConversionCastOp>(loc, TypeRange{toType}, inputs)
        .getResult(0);
  };
  tc.addSourceMaterialization(unrealizedCast);
  tc.addTargetMaterialization(unrealizedCast);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct AllocOpLowering : public OpConversionPattern<AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(AllocOp op, OpAdaptor,
                                ConversionPatternRewriter &rw) const override {
    auto arr = llvm::cast<ArrayType>(op.getResult().getType());
    rw.replaceOpWithNewOp<memref::AllocOp>(op, toMemRef(arr));
    return success();
  }
};

struct LoadOpLowering : public OpConversionPattern<LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override {
    rw.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getTensor(),
                                          adaptor.getIndices());
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override {
    rw.replaceOpWithNewOp<memref::StoreOp>(op, adaptor.getValue(),
                                            adaptor.getTensor(),
                                            adaptor.getIndices());
    return success();
  }
};

struct SliceOpLowering : public OpConversionPattern<SliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override {
    Location loc = op.getLoc();
    auto resArr = llvm::cast<ArrayType>(op.getResult().getType());
    int64_t rank = resArr.getRank();
    ArrayRef<int64_t> offsets = op.getOffsets();
    ArrayRef<int64_t> sizes   = op.getSizes();

    Value dst  = rw.create<memref::AllocOp>(loc, toMemRef(resArr));
    Value zero = rw.create<arith::ConstantIndexOp>(loc, 0);
    Value one  = rw.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value> ubs(rank), offs(rank);
    for (int64_t i = 0; i < rank; ++i) {
      ubs[i]  = rw.create<arith::ConstantIndexOp>(loc, sizes[i]);
      offs[i] = rw.create<arith::ConstantIndexOp>(loc, offsets[i]);
    }

    SmallVector<Value> ivs;
    std::function<void(int64_t)> nest = [&](int64_t d) {
      if (d == rank) {
        SmallVector<Value> si(rank);
        for (int64_t i = 0; i < rank; ++i)
          si[i] = rw.create<arith::AddIOp>(loc, offs[i], ivs[i]);
        Value v = rw.create<memref::LoadOp>(loc, adaptor.getSource(), si);
        rw.create<memref::StoreOp>(loc, v, dst, ivs);
        return;
      }
      auto forOp = rw.create<scf::ForOp>(loc, zero, ubs[d], one);
      ivs.push_back(forOp.getInductionVar());
      OpBuilder::InsertionGuard g(rw);
      rw.setInsertionPointToStart(forOp.getBody());
      nest(d + 1);
      ivs.pop_back();
    };
    nest(0);

    rw.replaceOp(op, dst);
    return success();
  }
};

struct TransposeOpLowering : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override {
    Location loc = op.getLoc();
    auto resArr = llvm::cast<ArrayType>(op.getResult().getType());
    int64_t rank = resArr.getRank();
    ArrayRef<int64_t> perm = op.getPermutation();

    Value dst  = rw.create<memref::AllocOp>(loc, toMemRef(resArr));
    Value zero = rw.create<arith::ConstantIndexOp>(loc, 0);
    Value one  = rw.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<int64_t> inv(rank);
    for (int64_t k = 0; k < rank; ++k)
      inv[perm[k]] = k;

    SmallVector<Value> ubs(rank);
    for (int64_t i = 0; i < rank; ++i)
      ubs[i] = rw.create<arith::ConstantIndexOp>(loc, resArr.getShape()[i]);

    SmallVector<Value> ivs;
    std::function<void(int64_t)> nest = [&](int64_t d) {
      if (d == rank) {
        SmallVector<Value> si(rank);
        for (int64_t a = 0; a < rank; ++a)
          si[a] = ivs[inv[a]];
        Value v = rw.create<memref::LoadOp>(loc, adaptor.getSource(), si);
        rw.create<memref::StoreOp>(loc, v, dst, ivs);
        return;
      }
      auto forOp = rw.create<scf::ForOp>(loc, zero, ubs[d], one);
      ivs.push_back(forOp.getInductionVar());
      OpBuilder::InsertionGuard g(rw);
      rw.setInsertionPointToStart(forOp.getBody());
      nest(d + 1);
      ivs.pop_back();
    };
    nest(0);

    rw.replaceOp(op, dst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertTensorExtToMemRefPass
    : public PassWrapper<ConvertTensorExtToMemRefPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTensorExtToMemRefPass)

  StringRef getArgument()    const final { return "convert-tensor-ext-to-memref"; }
  StringRef getDescription() const final {
    return "Lower tensor_ext dialect to memref + scf + arith.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect,
                    arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp    mod  = getOperation();

    TypeConverter tc;
    populateTensorExtTypeConverter(tc);

    ConversionTarget target(*ctx);
    target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect,
                           arith::ArithDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    target.addIllegalDialect<TensorExtDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp fn) {
      return tc.isSignatureLegal(fn.getFunctionType()) &&
             tc.isLegal(&fn.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp, func::CallOp>(
        [&](Operation *op) { return tc.isLegal(op); });

    RewritePatternSet patterns(ctx);
    populateTensorExtToMemRefPatterns(tc, patterns);

    if (failed(applyFullConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void mlir::tensor_ext::populateTensorExtToMemRefPatterns(
    TypeConverter &tc, RewritePatternSet &patterns) {
  patterns.add<AllocOpLowering, LoadOpLowering, StoreOpLowering,
               SliceOpLowering, TransposeOpLowering>(tc, patterns.getContext());
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, tc);
  populateReturnOpTypeConversionPattern(patterns, tc);
  populateCallOpTypeConversionPattern(patterns, tc);
}

std::unique_ptr<Pass>
mlir::tensor_ext::createConvertTensorExtToMemRefPass() {
  return std::make_unique<ConvertTensorExtToMemRefPass>();
}

void mlir::tensor_ext::registerConvertTensorExtToMemRefPass() {
  PassRegistration<ConvertTensorExtToMemRefPass>();
}
