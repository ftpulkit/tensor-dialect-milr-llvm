// Test 3 — Progressive lowering from tensor_ext all the way to LLVM dialect.
//
// Run each stage individually to see the IR transform:
//
//   Stage 0 (source, this file):
//     tensor-opt testcases/03_pipeline.mlir
//
//   Stage 1 (tensor_ext -> memref + scf + arith):
//     tensor-opt --convert-tensor-ext-to-memref testcases/03_pipeline.mlir
//
//   Stage 2 (scf -> explicit cf branches):
//     tensor-opt --convert-tensor-ext-to-memref --convert-scf-to-cf testcases/03_pipeline.mlir
//
//   Stage 3 (everything -> llvm dialect):
//     tensor-opt --convert-tensor-ext-to-memref --convert-scf-to-cf \
//                --convert-arith-to-llvm --convert-func-to-llvm     \
//                --finalize-memref-to-llvm --convert-cf-to-llvm     \
//                --reconcile-unrealized-casts testcases/03_pipeline.mlir

func.func @matrix_demo(%i: index, %j: index, %v: f32)
                      -> !tensor_ext.array<8x4xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  tensor_ext.store %v, %A[%i, %j] : f32, !tensor_ext.array<4x8xf32>
  %S = tensor_ext.slice %A offsets = [1, 2] sizes = [2, 4]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<2x4xf32>
  %T = tensor_ext.transpose %A permutation = [1, 0]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<8x4xf32>
  return %T : !tensor_ext.array<8x4xf32>
}
