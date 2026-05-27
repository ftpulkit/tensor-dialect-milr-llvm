# DEMO.md — How to record the demo

The assignment brief asks for a **demo: video or screenshots (show
working + failure case)**. This document tells you exactly what to
capture and in what order, so the submission looks clean and
professional.

Pick one of the two options below.

---

## Option A — Screenshots (easiest, ~5 minutes)

Take 6 screenshots total and drop them in a top‑level `screenshots/`
folder. Name them `01_*.png`, `02_*.png`, etc.

### Screenshot 1 — Successful build (`./build.sh`)

```bash
./build.sh
```

Capture the **end** of the output, showing the `================ Build
complete =====` banner and the `tensor-opt binary: …/tensor-opt` line.

### Screenshot 2 — High‑level IR

```bash
cat testcases/03_pipeline.mlir
```

Shows what a user writes — short and high‑level, no loops.

### Screenshot 3 — After `--convert-tensor-ext-to-memref` (Stage 1)

```bash
./build/bin/tensor-opt --convert-tensor-ext-to-memref testcases/03_pipeline.mlir
```

Shows `memref.alloc`, `memref.store`, `memref.load`, and `scf.for` nests.
This is the main lowering the assignment asks for.

### Screenshot 4 — Full pipeline to LLVM dialect (Stage 3)

```bash
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

Capture enough of the output to see `llvm.mlir.constant`, `llvm.alloca`,
`llvm.store`, `llvm.br`, etc. This is the "multi‑stage progressive
lowering" demonstration.

### Screenshot 5 — Failure case

Create a broken file on the fly and try to parse it:

```bash
cat > /tmp/bad.mlir <<'EOF'
func.func @bad(%i: index) {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  %x = tensor_ext.load %A[%i] : !tensor_ext.array<4x8xf32>
  return
}
EOF
./build/bin/tensor-opt /tmp/bad.mlir
echo "exit code: $?"
```

Capture the red error message

```
/tmp/bad.mlir:3:8: error: 'tensor_ext.load' op expected 2 index operands, got 1
```

and the non‑zero exit code line. This is the required **failure mode**.

### Screenshot 6 — Full `./run.sh` summary

```bash
./run.sh | tail -20
```

Capture the final `SUMMARY` block showing `Passed: 7 / 7` and
`ALL TESTS PASSED` in green.

---

## Option B — Video (~2 minutes)

Record a single terminal screencast using `asciinema`, OBS, or your
OS's built‑in screen recorder.

```bash
# Install if you don't have it
sudo apt install -y asciinema

# Record
asciinema rec demo.cast
# ... now run the six commands above, in order ...
# Ctrl-D to stop
```

Or just record with OBS / your phone. Keep it under 3 minutes and
follow this script:

1. `cat README.md | head -30`  *("here's the project")*
2. `./build.sh`  *("one command to build")*
3. `cat testcases/03_pipeline.mlir`  *("this is what a user writes")*
4. `./build/bin/tensor-opt --convert-tensor-ext-to-memref testcases/03_pipeline.mlir` *("and this is the lowering")*
5. *Full LLVM pipeline command from Screenshot 4 above* *("all the way to LLVM")*
6. *The failure demo from Screenshot 5* *("and here's the verifier catching a bug")*
7. `./run.sh | tail -20` *("full suite, 7/7 pass")*

---

## Adding the demo to the repo

```bash
mkdir screenshots
mv *.png screenshots/
git add screenshots/ DEMO.md
git commit -m "Add demo screenshots"
git push
```

or for the video:

```bash
# Put it on YouTube/Drive and paste the link in README.md's top section:
echo "## Demo video: https://..." >> README.md
```

---

## Grading checklist — what the teacher will look at

- [x] `./build.sh` runs and produces `build/bin/tensor-opt`
- [x] `./run.sh` runs all test cases and reports `7 / 7 pass`
- [x] `./scripts/evaluation.sh` runs and produces `docs/evaluation_report.txt`
- [x] Screenshots or video demonstrate both the success path and a failure case
- [x] README, DESIGN, IMPLEMENTATION, EVALUATION are all present
- [x] ≥ 5 test cases in `testcases/` (we have 7 + baseline)

If every box above is checked, the submission meets the brief.
