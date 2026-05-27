#!/usr/bin/env bash
# =============================================================================
#  run.sh — interactive demo / test suite for the tensor_ext MLIR dialect
#
#  Prereq: ./build.sh must have run successfully first.
#  Usage:  ./run.sh                  (interactive menu)
#          ./run.sh --all            (run all test cases non-interactively)
#          ./run.sh --eval           (run evaluation / metrics report)
#          ./run.sh --test <1-7>     (run a single numbered test directly)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC="${ROOT}/testcases"
BIN="${ROOT}/build/bin/tensor-opt"
FRONTEND="${ROOT}/scripts/frontend.py"

B="\033[1;34m"; G="\033[1;32m"; R="\033[1;31m"; Y="\033[1;33m"
C="\033[1;36m"; M="\033[1;35m"; W="\033[1;37m"; N="\033[0m"

if [[ ! -x "$BIN" ]]; then
  echo -e "${R}tensor-opt not found at $BIN.  Run ./build.sh first.${N}"
  exit 1
fi

PASS=0; FAIL=0

# ── helpers ──────────────────────────────────────────────────────────────────
hdr()   { echo; echo -e "${B}================================================================${N}";
          echo -e "${B}  $1${N}";
          echo -e "${B}================================================================${N}"; }
show()  { echo -e "${Y}\$ $*${N}"; }
pause() { echo; read -rp "  Press Enter to continue..." _; }

check() {
  local label="$1" want="$2"; shift 2
  local out; out=$(mktemp)
  if "$BIN" "$@" >"$out" 2>&1; then rc=0; else rc=$?; fi
  if [[ $rc -eq $want ]]; then
    echo -e "  ${G}✓ PASS${N}  $label"
    PASS=$((PASS+1))
  else
    echo -e "  ${R}✗ FAIL${N}  $label  (wanted rc=$want, got rc=$rc)"
    echo "  --- output ---"; cat "$out"; echo "  --- end ---"
    FAIL=$((FAIL+1))
  fi
  rm -f "$out"
}

# ── individual tests ──────────────────────────────────────────────────────────
run_test_1() {
  hdr "TEST 1 — Round-trip: parse → verify → print tensor_ext IR"
  echo -e "${C}--- Input (01_parse.mlir) ---${N}"
  cat "$TC/01_parse.mlir"
  echo -e "${C}--- Output ---${N}"
  show "$BIN $TC/01_parse.mlir"
  "$BIN" "$TC/01_parse.mlir"
  check "Round-trip of tensor_ext IR" 0 "$TC/01_parse.mlir"
}

run_test_2() {
  hdr "TEST 2 — Lower tensor_ext → memref + scf + arith"
  echo -e "${C}--- Input ---${N}"
  cat "$TC/02_lower.mlir"
  echo -e "${C}--- Output after --convert-tensor-ext-to-memref ---${N}"
  show "$BIN --convert-tensor-ext-to-memref $TC/02_lower.mlir"
  "$BIN" --convert-tensor-ext-to-memref "$TC/02_lower.mlir"
  check "Lower alloc/load/store/slice/transpose to memref" 0 \
    --convert-tensor-ext-to-memref "$TC/02_lower.mlir"
}

run_test_3() {
  hdr "TEST 3 — Progressive lowering pipeline (tensor_ext → LLVM dialect)"
  echo -e "${C}--- Stage 0: source ---${N}"
  cat "$TC/03_pipeline.mlir"
  echo -e "${C}--- Stage 1: +--convert-tensor-ext-to-memref ---${N}"
  "$BIN" --convert-tensor-ext-to-memref "$TC/03_pipeline.mlir"
  echo -e "${C}--- Stage 2: +--convert-scf-to-cf ---${N}"
  "$BIN" --convert-tensor-ext-to-memref --convert-scf-to-cf "$TC/03_pipeline.mlir"
  echo -e "${C}--- Stage 3: full lowering to LLVM dialect ---${N}"
  "$BIN" --convert-tensor-ext-to-memref \
         --convert-scf-to-cf \
         --convert-arith-to-llvm \
         --convert-func-to-llvm \
         --finalize-memref-to-llvm \
         --convert-cf-to-llvm \
         --reconcile-unrealized-casts \
         "$TC/03_pipeline.mlir"
  check "Full pipeline to LLVM dialect" 0 \
    --convert-tensor-ext-to-memref \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
    "$TC/03_pipeline.mlir"
}

run_test_4() {
  hdr "TEST 4 — Verifier: negative tests (errors are expected below)"
  echo -e "${C}Confirming that malformed inputs are correctly rejected...${N}"
  show "$BIN --split-input-file --verify-diagnostics $TC/04_verifier.mlir"
  "$BIN" --split-input-file --verify-diagnostics "$TC/04_verifier.mlir" || true
  check "Verifier rejects all malformed inputs" 0 \
    --split-input-file --verify-diagnostics "$TC/04_verifier.mlir"
}

run_test_5() {
  hdr "TEST 5 — Composed: alloc + store + slice + transpose in one function"
  echo -e "${C}--- Input ---${N}"
  cat "$TC/05_composed.mlir"
  echo -e "${C}--- Output ---${N}"
  "$BIN" --convert-tensor-ext-to-memref "$TC/05_composed.mlir"
  check "Composed ops lower correctly" 0 \
    --convert-tensor-ext-to-memref "$TC/05_composed.mlir"
}

run_test_6() {
  hdr "TEST 6 — Rank-3 tensors (rank-generic lowering)"
  echo -e "${C}--- Input ---${N}"
  cat "$TC/06_rank3.mlir"
  echo -e "${C}--- Output ---${N}"
  "$BIN" --convert-tensor-ext-to-memref "$TC/06_rank3.mlir"
  check "Rank-3 slice/transpose produce 3-level scf.for nests" 0 \
    --convert-tensor-ext-to-memref "$TC/06_rank3.mlir"
}

run_test_7() {
  hdr "TEST 7 — Verifier rejects rank-mismatched load (negative test)"
  BAD=$(mktemp --suffix=.mlir)
  trap 'rm -f "$BAD"' RETURN
  cat > "$BAD" << 'EOF'
func.func @rank_mismatch(%i: index) {
  %A = tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  // ERROR: rank-2 tensor needs 2 indices, not 1
  %x = tensor_ext.load %A[%i] : !tensor_ext.array<4x8xf32>
  return
}
EOF
  echo -e "${C}--- Bad input (should fail) ---${N}"
  cat "$BAD"
  echo
  echo -e "${C}--- tensor-opt output (error expected) ---${N}"
  "$BIN" "$BAD" 2>&1 || true
  check "Verifier correctly rejects rank-mismatched load" 1 "$BAD"
}

# ── manual input mode ─────────────────────────────────────────────────────────

run_manual() {
  hdr "MANUAL INPUT — High-Level C++ Frontend"
  echo
  echo -e "${W}Write high-level C++ style code. Supported constructs:${N}"
  echo
  echo -e "  ${Y}float A[4][8];${N}                       allocate a 4×8 tensor"
  echo -e "  ${Y}float x = A[i][j];${N}                   load element"
  echo -e "  ${Y}A[i][j] = value;${N}                     store element"
  echo -e "  ${Y}A[i][j] = 3.14;${N}                      store float literal"
  echo -e "  ${Y}float S[2][4] = slice(A, {0,2}, {2,4});${N}  slice (offset, size)"
  echo -e "  ${Y}float T[8][4] = transpose(A, {1,0});${N}     transpose"
  echo -e "  ${Y}return T;${N}                             return a tensor"
  echo
  echo -e "  Wrap in ${C}float funcname(float arg) { ... }${N} or just write statements."
  echo
  echo -e "${W}Examples you can try:${N}"
  echo -e "  ${C}float work(float v) { float A[4][8]; A[0][0] = v; float T[8][4] = transpose(A, {1,0}); return T; }${N}"
  echo -e "  ${C}float A[4][8]; float S[2][4] = slice(A, {0,2}, {2,4}); return S;${N}"
  echo
  echo -e "${W}Type your code below. Enter ${Y}END${W} on its own line when done.${N}"
  echo -e "${W}(Or press ${Y}Ctrl+C${W} to cancel)${N}"
  echo

  local tmpcode; tmpcode=$(mktemp --suffix=.cpp)
  local tmpmlir; tmpmlir=$(mktemp --suffix=.mlir)
  trap 'rm -f "$tmpcode" "$tmpmlir"' RETURN

  while IFS= read -r line; do
    [[ "$line" == "END" ]] && break
    echo "$line" >> "$tmpcode"
  done

  if [[ ! -s "$tmpcode" ]]; then
    echo -e "${R}No input received.${N}"
    return
  fi

  echo
  echo -e "${B}━━━━ Step 1: Your Code ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
  cat "$tmpcode"

  echo
  echo -e "${B}━━━━ Step 2: Generated tensor_ext MLIR ━━━━━━━━━━━━━━━━━━━━━━━━${N}"
  if ! python3 "$FRONTEND" "$tmpcode" > "$tmpmlir" 2>&1; then
    echo -e "${R}Frontend translation error:${N}"
    cat "$tmpmlir"
    return
  fi
  cat "$tmpmlir"

  echo
  echo -e "${M}Choose lowering stage:${N}"
  echo -e "  ${Y}[1]${N}  Stop here  (just show the tensor_ext MLIR)"
  echo -e "  ${Y}[2]${N}  Stage 1 — lower to memref/scf/arith"
  echo -e "  ${Y}[3]${N}  Stage 2 — + lower scf → cf"
  echo -e "  ${Y}[4]${N}  Stage 3 — full pipeline → LLVM dialect"
  echo
  read -rp "  Enter choice [1-4]: " pchoice

  case "$pchoice" in
    1)
      echo -e "${G}✓ MLIR generated successfully.${N}"
      ;;
    2)
      echo
      echo -e "${B}━━━━ Stage 1: tensor_ext → memref/scf/arith ━━━━━━━━━━━━━━━━━━━${N}"
      show "$BIN --convert-tensor-ext-to-memref <generated>"
      "$BIN" --convert-tensor-ext-to-memref "$tmpmlir" \
        && echo -e "\n${G}✓ Lowering succeeded.${N}" \
        || echo -e "\n${R}✗ Lowering failed (rc=$?). Check the MLIR above for errors.${N}"
      ;;
    3)
      echo
      echo -e "${B}━━━━ Stage 2: + scf → cf ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
      "$BIN" --convert-tensor-ext-to-memref \
             --convert-scf-to-cf \
             "$tmpmlir" \
        && echo -e "\n${G}✓ Lowering succeeded.${N}" \
        || echo -e "\n${R}✗ Lowering failed (rc=$?).${N}"
      ;;
    4)
      echo
      echo -e "${B}━━━━ Stage 3: Full → LLVM dialect ━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
      "$BIN" --convert-tensor-ext-to-memref \
             --convert-scf-to-cf \
             --convert-arith-to-llvm \
             --convert-func-to-llvm \
             --finalize-memref-to-llvm \
             --convert-cf-to-llvm \
             --reconcile-unrealized-casts \
             "$tmpmlir" \
        && echo -e "\n${G}✓ Full pipeline succeeded.${N}" \
        || echo -e "\n${R}✗ Pipeline failed (rc=$?).${N}"
      ;;
    *)
      echo -e "${R}Invalid choice.${N}"
      ;;
  esac
}

# ── run all tests ─────────────────────────────────────────────────────────────
run_all() {
  PASS=0; FAIL=0
  run_test_1; run_test_2; run_test_3; run_test_4
  run_test_5; run_test_6; run_test_7
  print_summary
}

print_summary() {
  hdr "SUMMARY"
  local TOTAL=$((PASS+FAIL))
  echo "  Tests run : $TOTAL"
  echo -e "  ${G}Passed    : $PASS${N}"
  if [[ $FAIL -gt 0 ]]; then
    echo -e "  ${R}Failed    : $FAIL${N}"; exit 1
  fi
  echo -e "  ${G}ALL TESTS PASSED ✓${N}"
}

# ── main menu ─────────────────────────────────────────────────────────────────
show_menu() {
  clear 2>/dev/null || true
  echo
  echo -e "${B}╔══════════════════════════════════════════════════════════════╗${N}"
  echo -e "${B}║        tensor_ext MLIR Dialect  —  Interactive Runner        ║${N}"
  echo -e "${B}╚══════════════════════════════════════════════════════════════╝${N}"
  echo
  echo -e "  ${W}Test Cases:${N}"
  echo -e "  ${Y}[1]${N}  TEST 1 — Round-trip: parse → verify → print"
  echo -e "  ${Y}[2]${N}  TEST 2 — Lower tensor_ext → memref/scf/arith"
  echo -e "  ${Y}[3]${N}  TEST 3 — Progressive lowering pipeline → LLVM dialect"
  echo -e "  ${Y}[4]${N}  TEST 4 — Verifier: reject malformed inputs"
  echo -e "  ${Y}[5]${N}  TEST 5 — Composed ops (alloc + store + slice + transpose)"
  echo -e "  ${Y}[6]${N}  TEST 6 — Rank-3 tensors (rank-generic lowering)"
  echo -e "  ${Y}[7]${N}  TEST 7 — Verifier rejects rank-mismatched load"
  echo
  echo -e "  ${W}Other:${N}"
  echo -e "  ${C}[a]${N}  Run ALL tests (full suite)"
  echo -e "  ${C}[e]${N}  Run EVALUATION (metrics report)"
  echo -e "  ${M}[m]${N}  Manual input — write C++ style code, get MLIR output"
  echo -e "  ${R}[q]${N}  Quit"
  echo
}

interactive_menu() {
  while true; do
    show_menu
    read -rp "  Select option: " choice
    echo
    PASS=0; FAIL=0
    case "$choice" in
      1) run_test_1; pause ;;
      2) run_test_2; pause ;;
      3) run_test_3; pause ;;
      4) run_test_4; pause ;;
      5) run_test_5; pause ;;
      6) run_test_6; pause ;;
      7) run_test_7; pause ;;
      a|A) run_all; pause ;;
      e|E) bash "${ROOT}/scripts/evaluation.sh"; pause ;;
      m|M) run_manual; pause ;;
      q|Q) echo -e "${G}Goodbye!${N}"; exit 0 ;;
      *)   echo -e "${R}Invalid option.${N}"; sleep 1 ;;
    esac
  done
}

# ── entry point ───────────────────────────────────────────────────────────────
case "${1:-}" in
  --all)  run_all ;;
  --eval) bash "${ROOT}/scripts/evaluation.sh" ;;
  --test)
    [[ -z "${2:-}" ]] && { echo -e "${R}Usage: $0 --test <1-7>${N}" >&2; exit 1; }
    PASS=0; FAIL=0
    case "$2" in
      1) run_test_1 ;; 2) run_test_2 ;; 3) run_test_3 ;;
      4) run_test_4 ;; 5) run_test_5 ;; 6) run_test_6 ;;
      7) run_test_7 ;;
      *) echo -e "${R}Unknown test: $2${N}" >&2; exit 1 ;;
    esac
    print_summary ;;
  "") interactive_menu ;;
  *)
    echo -e "${R}Unknown option: $1${N}" >&2
    echo "Usage: $0 [--all | --eval | --test <1-7>]" >&2
    exit 1 ;;
esac
