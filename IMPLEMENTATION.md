# IMPLEMENTATION.md — LLVM / MLIR specifics

This document walks through the *engineering* details of the project: the
LLVM/MLIR APIs used, why each one is there, and how the pieces fit together.
Read `DESIGN.md` first for the *why*; this file covers the *how*.

---

## 1. High‑level architecture

```
    TableGen              mlir-tblgen           C++ compilation           Link
   (TensorOps.td)  ───►  (.h.inc / .cpp.inc) ────────►  tensor_ext.o   ──►  tensor-opt
                                                        tensor_ext_lowering.o
```

Three C++ translation units:

1. `src/lib/TensorDialect/TensorDialect.cpp` — the dialect itself (types,
   ops, verifiers, dialect registration).
2. `src/lib/Conversion/TensorExtToMemRef.cpp` — the lowering pass.
3. `src/tools/tensor-opt/tensor-opt.cpp` — the CLI driver (30 lines of main).

Each of (1) and (2) `#include`s an `.inc` file that was generated from
`TensorOps.td` by `mlir-tblgen`. This is the canonical MLIR project layout.

---

## 2. TableGen: what goes in the `.td`

`src/include/TensorDialect/TensorOps.td` contains four things:

### 2.1 The dialect record

```tablegen
def TensorExt_Dialect : Dialect {
  let name = "tensor_ext";
  let cppNamespace = "::mlir::tensor_ext";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{ void registerTypes(); }];
}
```

`useDefaultTypePrinterParser = 1` tells the dialect class's generated
`parseType` / `printType` to dispatch to per‑type `assemblyFormat`
hooks. Without it we would have had to write those hooks in C++ by
hand.

### 2.2 The custom type

```tablegen
def TensorExt_ArrayType : TensorExt_Type<"Array", "array"> {
  let parameters = (ins
    ArrayRefParameter<"int64_t", "dimension sizes">:$shape,
    "Type":$elementType);
  let assemblyFormat = "`<` custom<ArrayShape>($shape, $elementType) `>`";
}
```

The `custom<ArrayShape>` hook is what requires `parseArrayShape` /
`printArrayShape` to exist at file scope in `TensorDialect.cpp` —
TableGen emits unqualified calls to those free functions in the generated
`.cpp.inc`. This is the single most common place newcomers trip up, and
it is why `TensorDialect.cpp` defines those helpers **above** the
`#include "TensorOpsTypes.cpp.inc"` line.

### 2.3 Ops

Each op follows the same skeleton:

```tablegen
def TensorExt_AllocOp : TensorExt_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let arguments = (ins);
  let results   = (outs TensorExt_AnyArray:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}
```

`MemoryEffects<[MemAlloc]>` is a trait from `SideEffectInterfaces.td` that
tells MLIR's analyses that `alloc` allocates new storage. It participates
in dead‑code elimination and pattern matching. Similarly `load` is
`MemRead`, `store` is `MemWrite`, and `slice`/`transpose` are
`[MemRead, MemWrite]` because they read the source and write a freshly
allocated destination.

`TensorExt_AnyArray` is a type constraint we define (via `CPred`) that
matches exactly `mlir::tensor_ext::ArrayType`. It lets the generated
op accessors return the specialised type directly instead of a generic
`Value` the caller has to cast.

### 2.4 What TableGen emits for each op

For `tensor_ext.load` the tablegen run produces (in `TensorOps.h.inc`):

- A class `LoadOp` deriving from `mlir::Op<LoadOp, ...>`.
- Accessors: `getTensor()`, `getIndices()`, `getResult()`.
- A generated `parse()` + `print()` matching the `assemblyFormat`.
- Hooks for `build()`, `getEffects()`, etc.

We then write the body of `LoadOp::verify()` in C++ because the rank
check is not expressible in TableGen.

---

## 3. The TypeConverter

```cpp
void populateTensorExtTypeConverter(TypeConverter &typeConverter) {
  typeConverter.addConversion([](Type t) { return t; });             // identity
  typeConverter.addConversion([](ArrayType arr) -> Type {            // ours
    return MemRefType::get(arr.getShape(), arr.getElementType());
  });
  typeConverter.addSourceMaterialization(addUnrealizedCast);
  typeConverter.addTargetMaterialization(addUnrealizedCast);
}
```

The identity conversion is important: without it, types like `f32` and
`index` that appear around our ops would be rejected as "unconvertible"
by the DialectConversion driver.

**Source / target materializations** are the escape hatches that let
the driver insert `builtin.unrealized_conversion_cast` ops when an
operand or result is in the *old* type domain while the rest of the
program has moved to the *new* one. At the end of full conversion those
casts should have cancelled out; if any remain (e.g. a missing pattern)
the `--reconcile-unrealized-casts` pass in the final pipeline stage
removes the chains.

---

## 4. The five ConversionPatterns

Each pattern derives from `OpConversionPattern<OpType>`. The `OpAdaptor`
template gives us access to operands that have already been converted to
their new types — so `adaptor.getTensor()` on a `LoadOp` returns a
`Value` of builtin `memref` type even though the op was originally on
`!tensor_ext.array`.

### 4.1 `AllocOpLowering`

One line of real work:

```cpp
rewriter.replaceOpWithNewOp<memref::AllocOp>(op, toMemRef(arr));
```

`replaceOpWithNewOp` constructs the replacement *and* RAUWs the result,
so every consumer of the old op's result now sees the `memref` value.

### 4.2 `LoadOpLowering` / `StoreOpLowering`

Similarly trivial — forward the (already‑converted) operands to the
corresponding `memref` op.

### 4.3 `SliceOpLowering` (non‑trivial)

This one has to synthesize actual code, not just rewrite one op:

```
1. %dst = memref.alloc() : memref<sizes... x T>
2. build a rank-deep scf.for nest with:
     lb = 0, ub = sizes[d], step = 1
3. in the innermost body:
     srcIdx[d] = offsets[d] + iv[d]
     val       = memref.load  %src[srcIdx]
     memref.store val, %dst[iv]
4. replace the op with %dst
```

The nest is built with a recursive lambda that pushes induction vars
onto a `SmallVector<Value>` as it descends. `OpBuilder::InsertionGuard`
saves and restores the insertion point so the recursion doesn't have to
manually manage it. This same structure is reused in
`TransposeOpLowering`.

### 4.4 `TransposeOpLowering` (the trickiest)

For permutation `p`, the loop nest is over the **result** shape. The
source index on axis `p[k]` is `iv[k]`. Equivalently, if we let
`inv_p` be the inverse permutation, then `srcIdx[a] = iv[inv_p[a]]`.
We compute `inv_p` once and use it in the body:

```cpp
SmallVector<int64_t> invPerm(rank);
for (int64_t k = 0; k < rank; ++k)
  invPerm[perm[k]] = k;
SmallVector<Value> srcIdx(rank);
for (int64_t a = 0; a < rank; ++a)
  srcIdx[a] = ivs[invPerm[a]];
```

This is the same mathematical trick used by `linalg.generic`'s
indexing‑map inversion, just expressed directly.

### 4.5 Function signatures and calls

After our five op patterns run, the module can still contain
`func.func @foo(%A : !tensor_ext.array<...>)` signatures and
`func.return %x : !tensor_ext.array<...>` terminators. Three upstream
helpers handle those:

```cpp
populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, tc);
populateReturnOpTypeConversionPattern(patterns, tc);
populateCallOpTypeConversionPattern(patterns, tc);
```

They are declared in `mlir/Dialect/Func/Transforms/FuncConversions.h`.
Forgetting to include that header produces a very confusing "undefined
reference" at link time.

---

## 5. The ConversionTarget

```cpp
ConversionTarget target(*ctx);
target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect,
                       arith::ArithDialect, func::FuncDialect,
                       BuiltinDialect>();
target.addIllegalDialect<TensorExtDialect>();
target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  return typeConverter.isSignatureLegal(op.getFunctionType()) &&
         typeConverter.isLegal(&op.getBody());
});
target.addDynamicallyLegalOp<func::ReturnOp, func::CallOp>(
    [&](Operation *op) { return typeConverter.isLegal(op); });
```

Key subtlety: `func.func` starts out legal (it's in `func::FuncDialect`)
but becomes dynamically illegal if its signature or body still contains
`!tensor_ext.array`. This is what forces the function‑interface pattern
to fire.

---

## 6. The driver (`tensor-opt.cpp`)

30 lines of real code:

```cpp
mlir::registerAllPasses();
mlir::DialectRegistry registry;
mlir::registerAllDialects(registry);
registry.insert<mlir::tensor_ext::TensorExtDialect>();
mlir::tensor_ext::registerConvertTensorExtToMemRefPass();
return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "tensor-opt driver\n", registry));
```

`MlirOptMain` is the same entry point `mlir-opt` itself uses. It handles
`--help`, the pass pipeline parser, diagnostic verification
(`--verify-diagnostics`), `--split-input-file`, `--mlir-print-ir-after`,
and everything else upstream supports. We get all of that for free by
linking against `MLIROptLib`.

---

## 7. Build system: how TableGen participates

`src/include/TensorDialect/CMakeLists.txt` calls `mlir_tablegen` six
times on `TensorOps.td`:

```cmake
mlir_tablegen(TensorOpsDialect.h.inc   -gen-dialect-decls -dialect=tensor_ext)
mlir_tablegen(TensorOpsDialect.cpp.inc -gen-dialect-defs  -dialect=tensor_ext)
mlir_tablegen(TensorOps.h.inc          -gen-op-decls)
mlir_tablegen(TensorOps.cpp.inc        -gen-op-defs)
mlir_tablegen(TensorOpsTypes.h.inc     -gen-typedef-decls -typedefs-dialect=tensor_ext)
mlir_tablegen(TensorOpsTypes.cpp.inc   -gen-typedef-defs  -typedefs-dialect=tensor_ext)
add_public_tablegen_target(TensorExtOpsIncGen)
```

Each generator flag produces a different piece of the class hierarchy:

- `-gen-dialect-decls` / `-defs` — the `TensorExtDialect` class.
- `-gen-op-decls` / `-defs` — the `AllocOp`, `LoadOp`, … classes.
- `-gen-typedef-decls` / `-defs` — the `ArrayType` class.

The six generated `.inc` files land in
`build/src/include/TensorDialect/`. The dialect and conversion libraries
`DEPENDS` on the `TensorExtOpsIncGen` custom target, so CMake rebuilds
them whenever the `.td` changes.

The `add_mlir_dialect_library` and `add_mlir_library` macros wire up
`MLIRIR`, `MLIRSupport`, and the right include paths automatically. We
only have to list the *semantic* deps (`MLIRMemRefDialect`,
`MLIRSCFDialect`, etc.).

---

## 8. Linking

`tools/tensor-opt/CMakeLists.txt` pulls in three things:

1. **Every upstream dialect and conversion**, via the `MLIR_DIALECT_LIBS`
   and `MLIR_CONVERSION_LIBS` global properties. That's what lets us
   chain our pass with `--convert-scf-to-cf`, `--convert-func-to-llvm`,
   etc., without manually listing each library.
2. **Our two libraries**: `MLIRTensorExt` and `MLIRTensorExtConversion`.
3. **`MLIROptLib`**: the library containing `MlirOptMain`.

`mlir_check_all_link_libraries(tensor-opt)` is a diagnostic check that
verifies every listed library is actually linked — catches typos early.

---

## 9. Progressive lowering (what the teacher asked for)

Classic LLVM has a single IR. A frontend is responsible for lowering its
AST straight to that IR — if any abstraction is worth preserving (the
fact that something is a matrix multiply, say), the frontend has to do
so outside the IR, and no optimizer in the backend can see it.

MLIR inverts that. Each dialect is a level of abstraction, and passes
move programs **down** the dialect lattice in small, auditable steps.
In this project:

| Stage | Command                                                                 | What you see in the IR                            |
|-------|-------------------------------------------------------------------------|---------------------------------------------------|
| 0     | (source)                                                                | `tensor_ext.alloc/load/store/slice/transpose`     |
| 1     | `--convert-tensor-ext-to-memref`                                        | `memref.alloc/load/store` + `scf.for`             |
| 2     | `--convert-scf-to-cf`                                                   | `cf.br`, `cf.cond_br` — no structured loops      |
| 3     | `--convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts` | pure `llvm.*` dialect — ready for `mlir-translate` |

Each stage is a separate pass. Each pass operates on a pair of dialects
and preserves a well‑defined invariant. This means:

- You can **stop** at any level and hand the IR to another tool.
- You can **replace** a stage with a better implementation without
  touching the others (e.g. swap our slice lowering for a
  `linalg.generic`‑based one later).
- You can **observe** what happens at each level with
  `--mlir-print-ir-after-all`.

That incremental, dialect‑aware model is what makes MLIR a framework for
building *many* compilers rather than one fixed compiler.

---

## 10. Notes & gotchas encountered during implementation

1. `custom<...>` hooks: the free functions called by TableGen must be
   defined **above** the typedef `.inc` include, not below.
2. `FuncConversions.h` is a separate header from `FuncOps.h`. The link
   error is not obvious.
3. `ArrayRefParameter<int64_t>` generates a parameter of type
   `SmallVectorImpl<int64_t>&` in newer MLIR; older MLIR 18.0.x may use
   `ArrayRef<int64_t>`. Both signatures are supplied in
   `TensorDialect.cpp` so the code works across point releases.
4. `useDefaultTypePrinterParser = 1` + `assemblyFormat` on the TypeDef
   together are what makes the custom type print as
   `!tensor_ext.array<4x8xf32>`. Removing either piece breaks parsing.
5. The `scf.for` body must be populated with `OpBuilder::InsertionGuard`
   scoping — failing to restore the insertion point after the
   recursive call causes later patterns to insert into the wrong
   region.

These are all exactly the kinds of lessons you only learn by actually
building against live MLIR, and they're the reason this implementation
document exists.
