//===- tensor-opt.cpp - Command-line driver for the tensor_ext dialect ---===//
//
// This is the `tensor-opt` tool.  It behaves exactly like upstream
// `mlir-opt` but additionally:
//
//   * registers the custom `tensor_ext` dialect, and
//   * registers the `--convert-tensor-ext-to-memref` pass.
//
// Usage examples:
//
//   # Parse, verify and round-trip a file written in tensor_ext:
//   tensor-opt input.mlir
//
//   # Run the lowering:
//   tensor-opt --convert-tensor-ext-to-memref input.mlir
//
//   # Continue lowering down to LLVM dialect:
//   tensor-opt --convert-tensor-ext-to-memref \
//              --convert-scf-to-cf \
//              --convert-func-to-llvm \
//              --finalize-memref-to-llvm \
//              --convert-arith-to-llvm \
//              --reconcile-unrealized-casts \
//              input.mlir
//
//===----------------------------------------------------------------------===//

#include "TensorDialect/Passes.h"
#include "TensorDialect/TensorDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  // Pick up every upstream dialect and pass -- this gives us things like
  // scf, memref, arith, func, and the standard lowering passes that we
  // chain after --convert-tensor-ext-to-memref.
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  // Register our custom dialect and pass.
  registry.insert<mlir::tensor_ext::TensorExtDialect>();
  mlir::tensor_ext::registerConvertTensorExtToMemRefPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "tensor-opt driver\n", registry));
}
