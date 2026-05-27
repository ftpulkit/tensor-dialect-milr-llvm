// Test 4 — Verifier negative tests.
// RUN: tensor-opt --split-input-file --verify-diagnostics %s
//
// Each section (separated by `// -----`) is a deliberately malformed program.
// The `expected-error` annotation tells the framework what error to expect.
// The test PASSES if every expected error is actually emitted.

// -----
func.func @bad_load_arity(%i: index) {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  // expected-error @+1 {{expected 2 index operands, got 1}}
  %x = tensor_ext.load %A[%i] : !tensor_ext.array<4x8xf32> -> f32
  return
}

// -----
func.func @bad_store_type(%i: index, %j: index, %v: i32) {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  // expected-error @+1 {{value type must match tensor element type}}
  tensor_ext.store %v, %A[%i, %j] : f32, !tensor_ext.array<4x8xf32>
  return
}

// -----
func.func @bad_slice_oob() -> !tensor_ext.array<2x4xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  // offsets[0]=3, sizes[0]=2 => 3+2=5 > 4 — out of bounds
  // expected-error @+1 {{slice goes out of bounds on dim 0}}
  %S = tensor_ext.slice %A offsets = [3, 0] sizes = [2, 4]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<2x4xf32>
  return %S : !tensor_ext.array<2x4xf32>
}

// -----
func.func @bad_slice_shape() -> !tensor_ext.array<2x4xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  // sizes=[2,5] but result type says <2x4xf32> — mismatch on dim 1
  // expected-error @+1 {{result shape dim 1 must equal sizes[1]}}
  %S = tensor_ext.slice %A offsets = [0, 0] sizes = [2, 5]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<2x4xf32>
  return %S : !tensor_ext.array<2x4xf32>
}

// -----
func.func @bad_perm_dup() -> !tensor_ext.array<8x4xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  // [0,0] is not a valid permutation
  // expected-error @+1 {{permutation must be a bijection; duplicate: 0}}
  %T = tensor_ext.transpose %A permutation = [0, 0]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<8x4xf32>
  return %T : !tensor_ext.array<8x4xf32>
}

// -----
func.func @bad_perm_shape() -> !tensor_ext.array<4x8xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  // perm=[1,0] on <4x8> should produce <8x4>, not <4x8>
  // expected-error @+1 {{result shape dim 0 must equal source shape dim 1}}
  %T = tensor_ext.transpose %A permutation = [1, 0]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<4x8xf32>
  return %T : !tensor_ext.array<4x8xf32>
}
