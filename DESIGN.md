# DESIGN.md — Design approach and alternatives

This document explains **why** the project is built the way it is, what
alternatives were considered, and what tradeoffs each choice involves.

---

## 1. Goals

The assignment asks for five things:

1. A custom MLIR dialect with high‑level multi‑dimensional array ops.
2. A TableGen definition giving those ops syntax and verification.
3. A conversion pass lowering the dialect to MLIR's `memref` dialect.
4. A test program demonstrating multi‑stage lowering.
5. A written discussion of how MLIR's progressive lowering differs from
   LLVM's single‑IR model.

Beyond those, we wanted the project to be **faithful to real MLIR
practice**, so that anyone reading it understands not just the toy
example but the architectural patterns they would see in a production
MLIR frontend (TensorFlow/MHLO, Torch‑MLIR, IREE, etc.).

---

## 2. Dialect naming — `tensor_ext`, not `tensor`

**Decision:** Call the dialect `tensor_ext`.

**Alternative:** Call it `tensor`.

**Why the decision:** Upstream MLIR already ships a builtin `tensor`
dialect. Registering another one with the same name produces a
duplicate‑dialect error as soon as `mlir::registerAllDialects` is
called. Using `tensor_ext` side‑steps the collision while keeping the
name obvious. This is the same pattern upstream uses for its own
out‑of‑tree examples (`toy`, `standalone`, etc.).

---

## 3. Custom type vs. reusing `tensor`/`memref`

**Decision:** Define our own type `!tensor_ext.array<DxD...xT>`.

**Alternatives:**

- Reuse the builtin `tensor<...>` type. Cleanest in principle, but
  defeats half the point of the assignment: showing how a dialect can
  introduce its own types and how `TypeConverter` translates them.
- Reuse `memref<...>`. This collapses the two abstraction levels into
  one — there would be nothing to lower, because the "high level" type
  already *is* the target type.

**Why the decision:** A separate type makes the lowering concrete.
After `--convert-tensor-ext-to-memref` runs, *every* `!tensor_ext.array`
must disappear from the IR — and the `TypeConverter` is the mechanism
that makes that happen even for values that thread through function
signatures and block arguments. That is the realistic workflow
production dialects use (e.g. the `torch` dialect's `!torch.tensor`
type is lowered to `tensor<...>` then to `memref<...>`).

---

## 4. Static vs. dynamic shapes

**Decision:** Static shapes only — every dimension must be a compile‑time
integer.

**Alternatives:**

- Support dynamic dims (`?`) like `memref` does. This would require
  `tensor_ext.alloc` to accept SSA `index` operands, slice/transpose
  patterns to thread dynamic sizes through the lowering, and dynamic
  sizes to survive until they become `memref.dim` calls or
  `memref.alloc(%n, %m)`.

**Why the decision:** Dynamic shapes roughly triple the complexity of
the lowering patterns and the verifier, without teaching anything new
about dialect design. Static shapes let every test case be fully
self‑contained and let the `scf.for` nests use `arith.constant` bounds.
This is the same simplification the upstream Toy tutorial makes.

---

## 5. Op set — why these five

The assignment lists `allocate, load, store, slice, transpose`. We kept
exactly that set for three reasons:

- `alloc` + `load` + `store` are the minimum needed to demonstrate
  a useful memory model.
- `slice` is the simplest *structural* op: it produces a new tensor
  of different shape, forcing the lowering to emit an `memref.alloc`
  and a copy loop.
- `transpose` demonstrates that structural ops can reason about
  **index permutations** — the lowering has to compute the inverse
  permutation and use it when emitting the load indices.

An obvious addition (matrix multiply, element‑wise add) was considered
but rejected: it would have grown the conversion patterns substantially
while only duplicating the `transpose`‑style lowering structure.

---

## 6. Slice / transpose lowering strategy

**Decision:** Emit an `memref.alloc` for the result, then a perfectly
nested `scf.for` loop of depth `rank` that copies each element
individually.

**Alternative 1:** Use `memref.subview` (for slice) and `memref.transpose`
(for transpose).

- Pro: far more compact IR — a single op instead of a loop nest.
- Con: those ops produce **views**, not owned buffers. The result
  aliases the source. That is a semantically different contract from
  what a user of a high‑level "slice" op usually expects, and it forces
  downstream passes to reason about aliasing. A production frontend
  typically lowers to "view" ops only after bufferization has
  determined that sharing storage is safe.
- Con: `memref.transpose` produces a memref with a non‑identity *layout
  map*, which then needs an additional pass (`--finalize-memref-to-llvm`
  with `--use-aligned-alloc`) and careful handling to reach LLVM. Loop
  lowering sidesteps that.

**Alternative 2:** Use `linalg.generic` / `linalg.copy`.

- Pro: even more declarative — tiling, fusion, and vectorization
  passes understand `linalg` deeply.
- Con: introduces a third dialect into the pipeline. The assignment
  explicitly asks for lowering to `memref`; `linalg` sits *between*
  our dialect and `memref`. That is worth showing in a report but not
  worth building as the default lowering.

**Why the decision:** Emitting `scf.for` keeps the output IR
**transparent** — any reader can see exactly what memory is being
touched — and it drops straight into the upstream pipeline without
needing the bufferization or `--finalize-memref-to-llvm` nuances that
views require. Future work: add a `--convert-tensor-ext-to-linalg`
pass as an alternative entry point.

---

## 7. Conversion framework — `DialectConversion` vs. `RewritePattern`

**Decision:** Use `applyFullConversion` with a `TypeConverter`,
`ConversionTarget`, and `OpConversionPattern`s.

**Alternative:** Use plain `RewritePattern` + `applyPatternsAndFoldGreedily`.

- Plain rewrites cannot easily rewrite **types** — so function
  signatures carrying `!tensor_ext.array` would be stuck half‑converted.
- Plain rewrites are greedy and can leave the IR in a partially
  converted state if a pattern fails. `applyFullConversion` guarantees
  that either the whole module is converted or the pass returns
  failure — a much stronger correctness property for a lowering.

`DialectConversion` is the framework every in‑tree `--convert-*-to-*`
pass uses (gpu→nvvm, async→llvm, vector→llvm, ...), so following it
means the code is immediately readable by anyone familiar with upstream
MLIR.

---

## 8. Legal/illegal dialect strategy

**Decision:**
- `tensor_ext` dialect is **illegal** after conversion.
- `memref`, `scf`, `arith`, `func`, `builtin` are **legal**.
- `func.func`, `func.return`, `func.call` are **dynamically legal**:
  legal iff they no longer carry `!tensor_ext.array` types.

The dynamic legality is what makes `populateFunctionOpInterfaceType-
ConversionPattern` kick in, rewriting function signatures. Without it,
a function returning `!tensor_ext.array<4x8xf32>` would remain illegal
forever because none of the op patterns touch function types.

---

## 9. Verifier placement — C++ vs. TableGen

TableGen offers simple assertions (`AllShapesMatch<...>`, `SameTypeOperands`)
that work fine for fixed‑arity ops. Our ops require:

- A **variadic** index list whose length must equal the tensor rank.
- A **permutation check** that is a proper bijection of `[0, rank)`.
- A **bounds check**: `offsets[i] + sizes[i] <= shape[i]`.

None of those are expressible as a TableGen predicate in any pleasant
way. We therefore set `let hasVerifier = 1;` on each op and wrote the
checks in C++. Upstream dialects (linalg, scf, tensor) follow the same
convention for this class of invariant.

---

## 10. Build system

**Decision:** Use MLIR's own `add_mlir_dialect_library`,
`add_mlir_library`, `mlir_tablegen` macros. Depend on a system‑installed
MLIR via `MLIR_DIR`/`LLVM_DIR`.

**Alternative:** Write a hand‑rolled Makefile or a CMake that uses
pure LLVM find_package.

**Why the decision:** MLIR's own CMake macros understand the TableGen
dependency graph, set the right include flags for the generated `.inc`
files, and participate correctly in the `MLIR_DIALECT_LIBS` and
`MLIR_CONVERSION_LIBS` globals. The result is a project layout that
mirrors `mlir/examples/toy/` almost exactly and that a reviewer can
drop into an LLVM monorepo if they want.

---

## 11. Non‑goals

Explicitly **not** attempted, to keep the scope tractable:

- Execution. The pipeline stops at LLVM dialect; actually translating
  to LLVM IR and running `lli` on the result is a one‑flag extension
  (`mlir-translate --mlir-to-llvmir`) but adds nothing conceptually.
- Bufferization. We skip the `tensor` → `memref` upstream bufferization
  step by not using the upstream `tensor` dialect at all.
- Optimization passes (canonicalization, CSE) over `tensor_ext`. These
  would be valuable in a real compiler but are out of scope for the
  assignment.
- Python bindings.

---

## 12. Summary of tradeoffs

| Decision                             | We get                                  | We give up                              |
|--------------------------------------|-----------------------------------------|-----------------------------------------|
| `tensor_ext` name                   | No collision with builtin               | Slightly wordier test files             |
| Custom array type                    | Real TypeConverter demonstration        | More boilerplate than reusing `memref`  |
| Static shapes                        | Simple verifier + patterns              | Can't handle unknown runtime sizes      |
| `scf.for` lowering (not subview)     | Transparent, self‑contained output IR   | More IR, slower than view‑based         |
| `DialectConversion`                  | Handles types + guarantees full rewrite | More API surface than greedy rewrites   |
| C++ verifiers                        | Expressive checks on permutations etc.  | Slightly more code than `.td` alone     |

Each tradeoff was taken in the direction that better illustrates
*how MLIR itself works*, because that is ultimately what the assignment
is asking the student to demonstrate.
