// Test 1 — Parse and round-trip every tensor_ext op.
// RUN: tensor-opt %s | tensor-opt
//
// This file contains one of each op. If parsing or printing regresses,
// running tensor-opt twice will fail (the second run can't parse the first's output).

func.func @roundtrip(%i: index, %j: index, %v: f32) {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  %x = tensor_ext.load  %A[%i, %j] : !tensor_ext.array<4x8xf32> -> f32
  tensor_ext.store %v, %A[%i, %j] : f32, !tensor_ext.array<4x8xf32>
  %S = tensor_ext.slice %A offsets = [0, 2] sizes = [2, 4]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<2x4xf32>
  %T = tensor_ext.transpose %A permutation = [1, 0]
         : !tensor_ext.array<4x8xf32> to !tensor_ext.array<8x4xf32>
  return
}
