// Test 7 / Evaluation workload.
// RUN: tensor-opt --convert-tensor-ext-to-memref %s
//
// Used by scripts/evaluation.sh to measure SLOC and op counts vs baseline.
// Allocates a 4x8 f32 matrix, writes to [0,0], transposes to 8x4.

func.func @workload(%v: f32) -> !tensor_ext.array<8x4xf32> {
  %A  = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  %c0 = arith.constant 0 : index
  tensor_ext.store %v, %A[%c0, %c0] : f32, !tensor_ext.array<4x8xf32>
  %T  = tensor_ext.transpose %A permutation = [1, 0]
          : !tensor_ext.array<4x8xf32> to !tensor_ext.array<8x4xf32>
  return %T : !tensor_ext.array<8x4xf32>
}
