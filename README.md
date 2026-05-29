# Assignment 10 — Custom MLIR Dialect for Tensor Operations

> **One-line summary:** A mini compiler built on MLIR that defines a custom
> high-level tensor dialect (`tensor_ext`), verifies it, and progressively
> lowers it all the way down to LLVM-compatible IR — exactly the way
> TensorFlow, PyTorch, and IREE work internally.

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [The big picture — lowering pipeline](#2-the-big-picture--lowering-pipeline)
3. [The custom dialect — `tensor_ext`](#3-the-custom-dialect--tensor_ext)
4. [How lowering works](#4-how-lowering-works)
5. [Directory layout](#5-directory-layout)
6. [Prerequisites & build](#6-prerequisites--build)
7. [How to run](#7-how-to-run)
8. [Test cases](#8-test-cases)
9. [Key design decisions](#9-key-design-decisions)
10. [Further reading](#10-further-reading)

---

## 1. What this project is

MLIR ("Multi-Level IR") is the compiler framework powering modern AI stacks.
Unlike LLVM's single fixed IR, MLIR lets you define **domain-specific
dialects** and chain them together so a program is gradually simplified —
each stage only needs to understand the two dialects on either side of it.

This project builds that idea from scratch on a small, self-contained example:

- We invented a dialect called **`tensor_ext`** that exposes five clean
  high-level operations for multi-dimensional arrays.
- We wrote **TableGen** definitions so MLIR knows the syntax, types, and
  verification rules.
- We wrote a **conversion pass** (`--convert-tensor-ext-to-memref`) using
  MLIR's `DialectConversion` framework that rewrites every `tensor_ext` op
  into equivalent `memref` + `scf` + `arith` ops.
- We then chain that with the upstream MLIR lowering passes to reach the
  **LLVM dialect**, ready for `mlir-translate` / `llc`.

---

## 2. The big picture — lowering pipeline

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  High-level C++ style input  (scripts/frontend.py)              │
  │                                                                  │
  │   float A[4][8];                                                 │
  │   float T[8][4] = transpose(A, {1,0});                          │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │  frontend.py  (Python translator)
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 0 — tensor_ext dialect  (our custom dialect)             │
  │                                                                  │
  │   %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>            │
  │   %T = tensor_ext.transpose %A permutation = [1, 0]             │
  │          : !tensor_ext.array<4x8xf32> to                        │
  │            !tensor_ext.array<8x4xf32>                           │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │  --convert-tensor-ext-to-memref
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 1 — memref + scf + arith                                  │
  │                                                                  │
  │   %A = memref.alloc() : memref<4x8xf32>                         │
  │   %T = memref.alloc() : memref<8x4xf32>                         │
  │   scf.for %i = 0 to 4 {                                         │
  │     scf.for %j = 0 to 8 {                                       │
  │       %v = memref.load %A[%i,%j]                                │
  │       memref.store %v, %T[%j,%i]                                │
  │     }                                                            │
  │   }                                                              │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │  --convert-scf-to-cf
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 2 — memref + cf + arith  (explicit branch/goto style)    │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │  --convert-arith-to-llvm
                                 │  --convert-func-to-llvm
                                 │  --finalize-memref-to-llvm
                                 │  --convert-cf-to-llvm
                                 │  --reconcile-unrealized-casts
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 3 — llvm dialect  (machine-level, ready for LLVM backend) │
  └─────────────────────────────────────────────────────────────────┘
```

This mirrors exactly what production compilers do:

| Real compiler | "Stage 0" dialect | Final target |
|---------------|-------------------|--------------|
| TensorFlow / XLA | HLO | LLVM / GPU PTX |
| PyTorch / Torch-MLIR | `torch` dialect | LLVM / CUDA |
| IREE | `flow` → `stream` | Vulkan SPIR-V / LLVM |
| **This project** | `tensor_ext` | LLVM dialect |

---

## 3. The custom dialect — `tensor_ext`

### Type

```
!tensor_ext.array<NxMxf32>   e.g.  !tensor_ext.array<4x8xf32>
```

A statically-shaped N-dimensional array of `f32` elements.  Defined in
`TensorOps.td` as `TensorExt_ArrayType` using MLIR's `TypeDef` TableGen class.

### Operations (defined in `TensorOps.td`)

| Op | Syntax | What it does |
|----|--------|--------------|
| `alloc` | `%A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>` | Allocate heap memory for a tensor |
| `load` | `%x = tensor_ext.load %A[%i, %j] : ... -> f32` | Read one element |
| `store` | `tensor_ext.store %v, %A[%i, %j] : f32, ...` | Write one element |
| `slice` | `%S = tensor_ext.slice %A offsets=[0,2] sizes=[2,4] : ... to ...` | Copy a rectangular sub-region into a new tensor |
| `transpose` | `%T = tensor_ext.transpose %A permutation=[1,0] : ... to ...` | Permute dimensions (e.g. rows↔cols) |

All five ops carry **verifiers** — for example, `load` checks that the number
of indices equals the tensor's rank, and `transpose` checks that the
permutation is a valid bijection.

### Example — composed usage

```mlir
func.func @composed(%i: index, %j: index, %v: f32)
                   -> !tensor_ext.array<2x2xf32> {
  %A = tensor_ext.alloc : !tensor_ext.array<4x4xf32>
  tensor_ext.store %v, %A[%i, %j] : f32, !tensor_ext.array<4x4xf32>
  %S = tensor_ext.slice %A offsets = [1, 1] sizes = [2, 2]
         : !tensor_ext.array<4x4xf32> to !tensor_ext.array<2x2xf32>
  %T = tensor_ext.transpose %S permutation = [1, 0]
         : !tensor_ext.array<2x2xf32> to !tensor_ext.array<2x2xf32>
  return %T : !tensor_ext.array<2x2xf32>
}
```

---

## 4. How lowering works

Lowering is implemented in `src/lib/Conversion/TensorExtToMemRef.cpp` using
MLIR's `DialectConversion` framework:

```
DialectConversion framework
│
├── TypeConverter
│     maps  !tensor_ext.array<4x8xf32>
│       →   memref<4x8xf32>
│
└── ConversionPatterns  (one per op)
      AllocOpLowering    →  memref.alloc()
      LoadOpLowering     →  memref.load
      StoreOpLowering    →  memref.store
      SliceOpLowering    →  memref.alloc + scf.for nest + memref.load/store
      TransposeOpLowering →  memref.alloc + scf.for nest + memref.load/store
```

**Transpose lowering in detail — what `tensor_ext.transpose` becomes:**

```
tensor_ext.transpose %A permutation=[1,0]

  →  %T = memref.alloc()              // new output buffer
     scf.for %i = 0 to dim[0] {      // loop over source rows
       scf.for %j = 0 to dim[1] {    // loop over source cols
         %v = memref.load  %A[%i, %j]
              memref.store %v, %T[%j, %i]   // permuted index
       }
     }
```

The pattern is **rank-generic**: it reads the shape at compile time and
generates as many nested `scf.for` loops as the tensor has dimensions, so
it works on 2D, 3D, or N-D tensors without any code changes.

---

## 5. Directory layout

```
tensor-dialect/
├── README.md                    ← this file
├── build.sh                     ← one-command build (required)
├── run.sh                       ← interactive demo + test runner (required)
├── CMakeLists.txt
│
├── scripts/
│   ├── evaluation.sh            ← metrics report (op counts, timing)
│   └── frontend.py              ← C++ style → tensor_ext MLIR translator
│
├── src/
│   ├── include/TensorDialect/
│   │   ├── TensorOps.td         ← TableGen: dialect, type, all 5 ops
│   │   ├── TensorDialect.h
│   │   ├── Passes.h
│   │   └── CMakeLists.txt       ← runs mlir-tblgen
│   │
│   ├── lib/
│   │   ├── TensorDialect/
│   │   │   └── TensorDialect.cpp   ← verifiers, type parser/printer
│   │   └── Conversion/
│   │       └── TensorExtToMemRef.cpp  ← 5 ConversionPatterns + TypeConverter
│   │
│   └── tools/tensor-opt/
│       └── tensor-opt.cpp       ← main(), registers all dialects + passes
│
└── testcases/
    ├── 01_parse.mlir            ← round-trip test
    ├── 02_lower.mlir            ← basic lowering
    ├── 03_pipeline.mlir         ← full pipeline to LLVM
    ├── 04_verifier.mlir         ← negative / error tests
    ├── 05_composed.mlir         ← all ops composed
    ├── 06_rank3.mlir            ← rank-3 generic lowering
    ├── 07_workload.mlir         ← evaluation workload
    └── baseline_memref.mlir     ← hand-written equivalent for comparison
```

---

## 6. Prerequisites & build

Tested on **Ubuntu 22.04 / 24.04** with MLIR/LLVM 18.

```bash
# Install dependencies (one time)
sudo apt update
sudo apt install -y cmake ninja-build clang-18 llvm-18-dev libmlir-18-dev mlir-18-tools

# Build
./build.sh
```

`build.sh` configures CMake, runs `mlir-tblgen` to generate C++ from the
`.td` file, then compiles `tensor-opt` into `./build/bin/tensor-opt`.
First build: ~2 min. Incremental: seconds.

---

## 7. How to run

### Interactive menu (recommended)

```bash
./run.sh
```

Pick from the menu:
- `[1–7]` — run an individual test case
- `[a]`   — run the full test suite
- `[e]`   — run the evaluation / metrics report
- `[m]`   — **manual input**: type C++ style code, get MLIR + lowered output
- `[q]`   — quit

### Manual input mode — what you can type

```cpp
// Allocate a tensor
float A[4][8];

// Load / store
float x = A[i][j];
A[i][j] = 3.14;

// Slice (offset per dim, size per dim)
float S[2][4] = slice(A, {0,2}, {2,4});

// Transpose (permutation of dimension indices)
float T[8][4] = transpose(A, {1,0});

// Optional function wrapper
float work(float v) {
  float A[4][8];
  A[0][0] = v;
  float T[8][4] = transpose(A, {1,0});
  return T;
}
```

Type `END` on its own line when done. Then choose which lowering stage
(1 = show MLIR only, 4 = full pipeline to LLVM dialect).

### Non-interactive (CI / scripts)

```bash
./run.sh --all          # full test suite, exit 0 on pass
./run.sh --eval         # evaluation metrics
./run.sh --test 3       # single test (1–7)
```

### `tensor-opt` directly

```bash
# Parse and print
./build/bin/tensor-opt testcases/01_parse.mlir

# Lower to memref
./build/bin/tensor-opt --convert-tensor-ext-to-memref testcases/02_lower.mlir

# Full pipeline to LLVM dialect
./build/bin/tensor-opt \
  --convert-tensor-ext-to-memref \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  testcases/03_pipeline.mlir
```

---

## 8. Test cases

| # | File | What it tests | Pass condition |
|---|------|---------------|----------------|
| 1 | `01_parse.mlir` | Parser round-trip | Exit 0, output equals input |
| 2 | `02_lower.mlir` | All 5 ops lowered | No `tensor_ext.*` ops remain |
| 3 | `03_pipeline.mlir` | Full pipeline → LLVM dialect | Exit 0 through all 7 passes |
| 4 | `04_verifier.mlir` | Verifier rejects bad inputs | `--verify-diagnostics` passes |
| 5 | `05_composed.mlir` | All ops composed end-to-end | Correct output shape |
| 6 | `06_rank3.mlir` | Rank-3 tensor lowering | 3-deep `scf.for` nests |
| 7 | `07_workload.mlir` | Evaluation workload | Used for op-count metrics |

The evaluation script (`./run.sh --eval`) also compares `tensor_ext` source
line count and op count against the hand-written `baseline_memref.mlir`,
quantifying the abstraction reduction the dialect provides.

---

## 9. Key design decisions

### Why MLIR's `DialectConversion` and not manual rewriting?

`DialectConversion` gives us:

- **Type conversion** — `!tensor_ext.array<4x8xf32>` → `memref<4x8xf32>`
  happens automatically across all SSA uses, including function signatures.
- **Legality checking** — we declare which ops are "illegal" after the pass;
  MLIR errors if any survive.
- **Rollback on failure** — if any pattern fails, the IR is left unchanged
  (no half-lowered mess).

### Why TableGen for op definitions?

TableGen auto-generates ~80% of the boilerplate: C++ class declarations,
`build()` methods, default `print()`/`parse()` implementations, and the
registration calls. We only hand-write the pieces that need custom logic
(custom assembly format for `load`/`store`, verifier bodies).

### Progressive lowering vs LLVM's single-IR model

| | LLVM (single IR) | MLIR (progressive lowering) |
|---|---|---|
| Representation | One fixed IR (`llvm::Module`) | Many dialects, each specialized |
| Optimization level | Generic mid-level only | Domain-specific opts at each level |
| Adding a new abstraction | Requires passes that understand all of LLVM IR | New dialect is self-contained |
| Example | `clang → llvm IR → machine code` | `tensor_ext → memref → cf → llvm` |

MLIR's approach lets each optimization target the right level of
abstraction (e.g., tensor tiling before bufferization, loop fusion after).

---

## 10. Further reading

- [MLIR documentation](https://mlir.llvm.org/docs/)
- [Dialect conversion framework](https://mlir.llvm.org/docs/DialectConversion/)
- [TableGen language reference](https://llvm.org/docs/TableGen/ProgRef.html)
- [Torch-MLIR](https://github.com/llvm/torch-mlir) — real-world dialect lowering for PyTorch
- [IREE](https://iree.dev/) — full ML compiler using the same progressive lowering ideas
