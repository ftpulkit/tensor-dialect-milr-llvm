#!/usr/bin/env bash
# =============================================================================
#  scripts/evaluation.sh
#
#  Generates the numbers used in EVALUATION.md.  Metrics:
#
#    A) Source-level productivity:
#         lines / ops of tensor_ext workload vs. hand-written memref baseline.
#
#    B) Lowering fidelity:
#         ops at each stage of the progressive lowering pipeline.
#
#    C) Pass wall-clock time:
#         --mlir-timing of the lowering pass on each test case.
#
#    D) Correctness:
#         all 7 test cases in testcases/ exit 0 (or the expected code).
#
#  Output: a pretty-printed table to stdout and a copy written to
#          docs/evaluation_report.txt for inclusion in screenshots.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TESTS_DIR="${PROJECT_ROOT}/testcases"
OUT_FILE="${PROJECT_ROOT}/docs/evaluation_report.txt"
mkdir -p "${PROJECT_ROOT}/docs"

TENSOR_OPT="${BUILD_DIR}/bin/tensor-opt"
[[ -x "${TENSOR_OPT}" ]] || TENSOR_OPT="${BUILD_DIR}/src/tools/tensor-opt/tensor-opt"
if [[ ! -x "${TENSOR_OPT}" ]]; then
  echo "tensor-opt not built.  Run ./build.sh first." >&2
  exit 1
fi

# Redirect everything to both stdout and the report file.
exec > >(tee "$OUT_FILE") 2>&1

echo "=============================================================="
echo "  tensor_ext dialect - Evaluation Report"
echo "  Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Host:      $(uname -srvmo)"
echo "=============================================================="
echo

# -----------------------------------------------------------------------------
# Helper: count non-blank non-comment lines in a .mlir file.
# -----------------------------------------------------------------------------
loc() {
  grep -cvE '^[[:space:]]*(//|$)' "$1" || true
}

# Helper: count ops in a piece of IR (heuristic: count `DIALECT.NAME` tokens).
# Strips line comments first, and excludes our type name 'tensor_ext.array'
# (which matches the op-name pattern but is actually a type).
op_count() {
  sed 's|//.*||' | awk '
    {
      while (match($0, /[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*/)) {
        tok = substr($0, RSTART, RLENGTH);
        if (tok !~ /^(memref|scf|arith|tensor_ext|func|cf|llvm|builtin)\./) {
            $0 = substr($0, RSTART+RLENGTH);
            continue;
        }
        if (tok == "tensor_ext.array") {   # type, not an op
            $0 = substr($0, RSTART+RLENGTH);
            continue;
        }
        counts[tok]++;
        total++;
        $0 = substr($0, RSTART+RLENGTH);
      }
    }
    END {
      for (k in counts) printf "    %-30s %5d\n", k, counts[k] | "sort";
      close("sort");
      printf "    %-30s %5d\n", "TOTAL", total;
    }
  '
}

# Helper: total ops (an integer) for the input IR on stdin.
op_total() {
  sed 's|//.*||' | awk '
    {
      while (match($0, /[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*/)) {
        tok = substr($0, RSTART, RLENGTH);
        if (tok ~ /^(memref|scf|arith|tensor_ext|func|cf|llvm|builtin)\./ \
            && tok != "tensor_ext.array") n++;
        $0 = substr($0, RSTART+RLENGTH);
      }
    }
    END { print n+0 }
  '
}

# ==============================================================================
echo "## A) Source-level productivity"
echo "## ----------------------------"
echo "## Compare the same computation expressed in tensor_ext versus"
echo "## hand-written memref/scf/arith."
echo
# ==============================================================================

HI="${TESTS_DIR}/07_workload.mlir"
LO="${TESTS_DIR}/baseline_memref.mlir"

hi_loc=$(loc "$HI")
lo_loc=$(loc "$LO")
hi_ops=$(cat "$HI" | op_total)
lo_ops=$(cat "$LO" | op_total)

printf "  %-35s %10s %10s\n" "Metric" "tensor_ext" "baseline"
printf "  %-35s %10s %10s\n" "-----------------------------------" "----------" "----------"
printf "  %-35s %10d %10d\n" "Non-blank, non-comment SLOC" "$hi_loc" "$lo_loc"
printf "  %-35s %10d %10d\n" "Total MLIR ops in source" "$hi_ops" "$lo_ops"
reduction=$(awk -v a="$hi_loc" -v b="$lo_loc" 'BEGIN{ if (b==0) print 0; else printf "%.1f", 100*(1 - a/b) }')
echo
echo "  => tensor_ext source is ${reduction}% shorter than hand-written memref."
echo

# ==============================================================================
echo "## B) Lowering fidelity (op counts per stage)"
echo "## -----------------------------------------"
echo "## Stages:  0=source  1=+tensor_ext-to-memref  2=+scf-to-cf  3=full->llvm"
echo
# ==============================================================================

for tc in "${TESTS_DIR}"/03_pipeline.mlir "${TESTS_DIR}"/07_workload.mlir; do
  name=$(basename "$tc")
  echo "  File: $name"
  s0=$(cat "$tc" | op_total)
  s1=$("${TENSOR_OPT}" --convert-tensor-ext-to-memref "$tc" 2>/dev/null | op_total)
  s2=$("${TENSOR_OPT}" --convert-tensor-ext-to-memref --convert-scf-to-cf "$tc" 2>/dev/null | op_total)
  s3=$("${TENSOR_OPT}" --convert-tensor-ext-to-memref \
                       --convert-scf-to-cf \
                       --convert-arith-to-llvm \
                       --convert-func-to-llvm \
                       --finalize-memref-to-llvm \
                       --convert-cf-to-llvm \
                       --reconcile-unrealized-casts \
                       "$tc" 2>/dev/null | op_total)
  printf "    stage 0 (source tensor_ext) : %4d ops\n" "$s0"
  printf "    stage 1 (+tensor_ext->mem)  : %4d ops\n" "$s1"
  printf "    stage 2 (+scf->cf)          : %4d ops\n" "$s2"
  printf "    stage 3 (full -> LLVM)      : %4d ops\n" "$s3"
  echo
done

# ==============================================================================
echo "## C) Wall-clock pass timing"
echo "## -------------------------"
echo "## Time to execute --convert-tensor-ext-to-memref on each test case."
echo "## (Uses MLIR's built-in --mlir-timing infrastructure.)"
echo
# ==============================================================================

for tc in "${TESTS_DIR}"/0[1-7]_*.mlir; do
  name=$(basename "$tc")
  # Skip verifier-negative file (it deliberately fails).
  [[ "$name" == 04_verifier.mlir ]] && continue
  start=$(date +%s%N)
  "${TENSOR_OPT}" --convert-tensor-ext-to-memref "$tc" >/dev/null 2>&1 || true
  end=$(date +%s%N)
  ms=$(( (end - start) / 1000000 ))
  printf "    %-30s %6d ms\n" "$name" "$ms"
done
echo

# ==============================================================================
echo "## D) Correctness -- full test suite"
echo "## ---------------------------------"
echo
# ==============================================================================

pass=0; fail=0
run_expect() {
  local tc="$1" want="$2"; shift 2
  "${TENSOR_OPT}" "$@" "$tc" >/dev/null 2>&1 && got=0 || got=$?
  if [[ "$got" -eq "$want" ]]; then
    printf "    PASS  %-30s (rc=%d)\n" "$(basename "$tc")" "$got"
    pass=$((pass+1))
  else
    printf "    FAIL  %-30s (want rc=%d, got rc=%d)\n" \
       "$(basename "$tc")" "$want" "$got"
    fail=$((fail+1))
  fi
}
run_expect "${TESTS_DIR}/01_parse.mlir"    0
run_expect "${TESTS_DIR}/02_lower.mlir"    0 --convert-tensor-ext-to-memref
run_expect "${TESTS_DIR}/03_pipeline.mlir" 0 --convert-tensor-ext-to-memref
run_expect "${TESTS_DIR}/04_verifier.mlir" 0 --split-input-file --verify-diagnostics
run_expect "${TESTS_DIR}/05_composed.mlir" 0 --convert-tensor-ext-to-memref
run_expect "${TESTS_DIR}/06_rank3.mlir"    0 --convert-tensor-ext-to-memref
run_expect "${TESTS_DIR}/07_workload.mlir" 0 --convert-tensor-ext-to-memref

echo
echo "    -----------------------------------------"
echo "    Passed: $pass / $((pass+fail))"

echo
echo "Full report written to: $OUT_FILE"
