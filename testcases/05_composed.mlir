// Test 5 — End-to-end: all four non-trivial ops composed in one function.
// RUN: tensor-opt --convert-tensor-ext-to-memref %s

func.func @composed(%i: index, %j: index, %v: f32)
                   -> !tensor_ext.array<2x2xf32> {
  // 1. Allocate 4x4 matrix
  %A = tensor_ext.alloc : !tensor_ext.array<4x4xf32>
  // 2. Write one element
  tensor_ext.store %v, %A[%i, %j] : f32, !tensor_ext.array<4x4xf32>
  // 3. Take a 2x2 sub-block starting at [1,1]
  %S = tensor_ext.slice %A offsets = [1, 1] sizes = [2, 2]
         : !tensor_ext.array<4x4xf32> to !tensor_ext.array<2x2xf32>
  // 4. Transpose it
  %T = tensor_ext.transpose %S permutation = [1, 0]
         : !tensor_ext.array<2x2xf32> to !tensor_ext.array<2x2xf32>
  return %T : !tensor_ext.array<2x2xf32>
}
