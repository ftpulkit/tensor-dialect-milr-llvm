// Test 2 — Lower every tensor_ext op to memref/scf/arith.
// RUN: tensor-opt --convert-tensor-ext-to-memref %s

// After lowering, NO tensor_ext.* ops should remain.

// --- alloc / load / store ---------------------------------------------------
func.func @basic(%i: index, %j: index, %v: f32) {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  %x = tensor_ext.load  %A[%i, %j] : !tensor_ext.array<4x8xf32> -> f32
  tensor_ext.store %v, %A[%i, %j] : f32, !tensor_ext.array<4x8xf32>
  return
}

// --- slice ------------------------------------------------------------------
func.func @do_slice() -> !tensor_ext.array<2x4xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  %S = tensor_ext.slice %A offsets = [0, 2] sizes = [2, 4]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<2x4xf32>
  return %S : !tensor_ext.array<2x4xf32>
}

// --- transpose --------------------------------------------------------------
func.func @do_transpose() -> !tensor_ext.array<8x4xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  %T = tensor_ext.transpose %A permutation = [1, 0]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<8x4xf32>
  return %T : !tensor_ext.array<8x4xf32>
}
