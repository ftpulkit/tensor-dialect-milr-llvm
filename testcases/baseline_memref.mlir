// Baseline: same computation as 07_workload.mlir written by hand in
// memref/scf/arith — what you'd write WITHOUT the tensor_ext dialect.
// Used in EVALUATION.md section A (source-level productivity comparison).
// RUN: tensor-opt %s

func.func @workload_baseline(%v: f32) -> memref<8x4xf32> {
  %A   = memref.alloc() : memref<4x8xf32>
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c4  = arith.constant 4 : index
  %c8  = arith.constant 8 : index

  memref.store %v, %A[%c0, %c0] : memref<4x8xf32>

  %T = memref.alloc() : memref<8x4xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %x = memref.load %A[%j, %i] : memref<4x8xf32>
      memref.store %x, %T[%i, %j] : memref<8x4xf32>
    }
  }
  return %T : memref<8x4xf32>
}
