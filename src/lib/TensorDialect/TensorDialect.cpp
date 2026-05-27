//===- TensorDialect.cpp - tensor_ext dialect implementation --------------===//
#include "TensorDialect/TensorDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tensor_ext;

//===----------------------------------------------------------------------===//
// Dialect class (generated)
//===----------------------------------------------------------------------===//
#include "TensorDialect/TensorOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom parser/printer for ArrayType shape.
// MUST be defined BEFORE the typedef .inc include.
// Signature matches what TableGen generates for ArrayRefParameter<"int64_t">.
//===----------------------------------------------------------------------===//

static ParseResult parseArrayShape(AsmParser &parser,
                                   SmallVectorImpl<int64_t> &shape,
                                   Type &elementType) {
  SmallVector<int64_t, 4> dims;
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false,
                                /*withTrailingX=*/true))
    return failure();
  shape.assign(dims.begin(), dims.end());
  return parser.parseType(elementType);
}

static void printArrayShape(AsmPrinter &printer,
                            ArrayRef<int64_t> shape,
                            Type elementType) {
  for (int64_t d : shape)
    printer << d << 'x';
  printer << elementType;
}

//===----------------------------------------------------------------------===//
// Type definitions (generated) — MUST come after helpers above
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "TensorDialect/TensorOpsTypes.cpp.inc"

void TensorExtDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TensorDialect/TensorOpsTypes.cpp.inc"
      >();
}

int64_t ArrayType::getNumElements() const {
  int64_t n = 1;
  for (int64_t d : getShape()) n *= d;
  return n;
}

//===----------------------------------------------------------------------===//
// Op definitions (generated)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "TensorDialect/TensorOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect init
//===----------------------------------------------------------------------===//

void TensorExtDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "TensorDialect/TensorOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Custom assembly format for LoadOp
//   Syntax:  tensor_ext.load %A[%i, %j] : !tensor_ext.array<4x8xf32> -> f32
//===----------------------------------------------------------------------===//

void LoadOp::print(OpAsmPrinter &p) {
  p << ' ' << getTensor() << '[' << getIndices() << ']';
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getTensor().getType() << " -> " << getResult().getType();
}

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand tensorOperand;
  SmallVector<OpAsmParser::UnresolvedOperand> indexOperands;
  Type tensorType, resultType;

  if (parser.parseOperand(tensorOperand) ||
      parser.parseOperandList(indexOperands, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tensorType) ||
      parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  auto arrType = llvm::dyn_cast<ArrayType>(tensorType);
  if (!arrType)
    return parser.emitError(parser.getNameLoc(), "expected tensor_ext array type");

  SmallVector<Type> indexTypes(indexOperands.size(),
                               IndexType::get(parser.getContext()));
  if (parser.resolveOperand(tensorOperand, tensorType, result.operands) ||
      parser.resolveOperands(indexOperands, indexTypes,
                             parser.getNameLoc(), result.operands))
    return failure();

  result.addTypes(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// Custom assembly format for StoreOp
//   Syntax:  tensor_ext.store %v, %A[%i, %j] : f32, !tensor_ext.array<4x8xf32>
//===----------------------------------------------------------------------===//

void StoreOp::print(OpAsmPrinter &p) {
  p << ' ' << getValue() << ", " << getTensor()
    << '[' << getIndices() << ']';
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getValue().getType() << ", " << getTensor().getType();
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand valueOperand, tensorOperand;
  SmallVector<OpAsmParser::UnresolvedOperand> indexOperands;
  Type valueType, tensorType;

  if (parser.parseOperand(valueOperand) ||
      parser.parseComma() ||
      parser.parseOperand(tensorOperand) ||
      parser.parseOperandList(indexOperands, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() ||
      parser.parseType(valueType) ||
      parser.parseComma() ||
      parser.parseType(tensorType))
    return failure();

  SmallVector<Type> indexTypes(indexOperands.size(),
                               IndexType::get(parser.getContext()));
  if (parser.resolveOperand(valueOperand, valueType, result.operands) ||
      parser.resolveOperand(tensorOperand, tensorType, result.operands) ||
      parser.resolveOperands(indexOperands, indexTypes,
                             parser.getNameLoc(), result.operands))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  auto arrTy = llvm::cast<ArrayType>(getTensor().getType());
  if (getIndices().size() != static_cast<size_t>(arrTy.getRank()))
    return emitOpError("expected ") << arrTy.getRank()
           << " index operands, got " << getIndices().size();
  if (getResult().getType() != arrTy.getElementType())
    return emitOpError("result type must match tensor element type");
  return success();
}

LogicalResult StoreOp::verify() {
  auto arrTy = llvm::cast<ArrayType>(getTensor().getType());
  if (getIndices().size() != static_cast<size_t>(arrTy.getRank()))
    return emitOpError("expected ") << arrTy.getRank()
           << " index operands, got " << getIndices().size();
  if (getValue().getType() != arrTy.getElementType())
    return emitOpError("value type must match tensor element type");
  return success();
}

LogicalResult SliceOp::verify() {
  auto srcTy  = llvm::cast<ArrayType>(getSource().getType());
  auto resTy  = llvm::cast<ArrayType>(getResult().getType());
  ArrayRef<int64_t> offsets = getOffsets();
  ArrayRef<int64_t> sizes   = getSizes();
  int64_t rank = srcTy.getRank();

  if ((int64_t)offsets.size() != rank)
    return emitOpError("`offsets` must have ") << rank << " entries";
  if ((int64_t)sizes.size() != rank)
    return emitOpError("`sizes` must have ") << rank << " entries";
  if (resTy.getRank() != rank)
    return emitOpError("result rank must match source rank");
  if (srcTy.getElementType() != resTy.getElementType())
    return emitOpError("element types must match");
  for (int64_t i = 0; i < rank; ++i) {
    if (offsets[i] < 0 || sizes[i] < 0)
      return emitOpError("offsets and sizes must be non-negative");
    if (offsets[i] + sizes[i] > srcTy.getShape()[i])
      return emitOpError("slice goes out of bounds on dim ") << i;
    if (resTy.getShape()[i] != sizes[i])
      return emitOpError("result shape dim ")
             << i << " must equal sizes[" << i << "]";
  }
  return success();
}

LogicalResult TransposeOp::verify() {
  auto srcTy = llvm::cast<ArrayType>(getSource().getType());
  auto resTy = llvm::cast<ArrayType>(getResult().getType());
  ArrayRef<int64_t> perm = getPermutation();
  int64_t rank = srcTy.getRank();

  if ((int64_t)perm.size() != rank)
    return emitOpError("permutation must have ") << rank << " entries";
  if (resTy.getRank() != rank)
    return emitOpError("result rank must match source rank");
  if (srcTy.getElementType() != resTy.getElementType())
    return emitOpError("element types must match");

  SmallVector<bool> seen(rank, false);
  for (int64_t p : perm) {
    if (p < 0 || p >= rank)
      return emitOpError("permutation entry out of range: ") << p;
    if (seen[p])
      return emitOpError("permutation must be a bijection; duplicate: ") << p;
    seen[p] = true;
  }
  for (int64_t i = 0; i < rank; ++i)
    if (resTy.getShape()[i] != srcTy.getShape()[perm[i]])
      return emitOpError("result shape dim ")
             << i << " must equal source shape dim " << perm[i];
  return success();
}
