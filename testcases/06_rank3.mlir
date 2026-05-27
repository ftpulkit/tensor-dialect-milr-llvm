// Test 6 — Rank-3 tensors: proves the loop-nest builder is rank-generic.
// RUN: tensor-opt --convert-tensor-ext-to-memref %s
//
// Expected: rank-3 ops produce 3 nested scf.for loops.

func.func @rank3_slice() -> !tensor_ext.array<1x2x2xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<2x3x4xf32>
  %S = tensor_ext.slice %A offsets = [0, 1, 1] sizes = [1, 2, 2]
         : !tensor_ext.array<2x3x4xf32> to !tensor_ext.array<1x2x2xf32>
  return %S : !tensor_ext.array<1x2x2xf32>
}

// permutation [2,0,1] on <2x3x4> gives <4x2x3>
func.func @rank3_transpose() -> !tensor_ext.array<4x2x3xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<2x3x4xf32>
  %T = tensor_ext.transpose %A permutation = [2, 0, 1]
         : !tensor_ext.array<2x3x4xf32> to !tensor_ext.array<4x2x3xf32>
  return %T : !tensor_ext.array<4x2x3xf32>
}
