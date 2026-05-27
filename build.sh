#!/usr/bin/env bash
# =============================================================================
#  build.sh — build the tensor_ext MLIR dialect project
#
#  Usage:
#    ./build.sh                      # uses LLVM/MLIR 18 (default)
#    LLVM_VERSION=17 ./build.sh      # override LLVM version
#    JOBS=4 ./build.sh               # override parallelism
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${ROOT}/build"
VER="${LLVM_VERSION:-18}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

B="\033[1;34m"; G="\033[1;32m"; R="\033[1;31m"; Y="\033[1;33m"; N="\033[0m"
banner() { echo -e "${B}=== $1 ===${N}"; }

banner "tensor_ext MLIR dialect — build.sh"
echo "  Root      : $ROOT"
echo "  Build dir : $BUILD"
echo "  LLVM ver  : $VER"
echo "  Jobs      : $JOBS"
echo

# ---- 1. Check required apt packages ----------------------------------------
banner "Checking prerequisites"

PKGS=(cmake ninja-build "clang-${VER}" "llvm-${VER}-dev"
      "libmlir-${VER}-dev" "mlir-${VER}-tools")
MISSING=()
for p in "${PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done

if (( ${#MISSING[@]} )); then
  echo -e "${R}Missing packages: ${MISSING[*]}${N}"
  echo
  echo "Run:"
  echo -e "${Y}  sudo apt update${N}"
  echo -e "${Y}  sudo apt install -y ${MISSING[*]}${N}"
  echo
  echo "Then re-run ./build.sh"
  exit 1
fi
echo -e "${G}All packages present.${N}"

MLIR_CMAKE="/usr/lib/llvm-${VER}/lib/cmake/mlir"
LLVM_CMAKE="/usr/lib/llvm-${VER}/lib/cmake/llvm"
if [[ ! -d "$MLIR_CMAKE" ]]; then
  echo -e "${R}Cannot find $MLIR_CMAKE — is libmlir-${VER}-dev installed?${N}"
  exit 1
fi

# ---- 2. Configure ----------------------------------------------------------
banner "Configuring (CMake + Ninja)"

cmake -S "$ROOT" -B "$BUILD" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="clang-${VER}" \
      -DCMAKE_CXX_COMPILER="clang++-${VER}" \
      -DMLIR_DIR="$MLIR_CMAKE" \
      -DLLVM_DIR="$LLVM_CMAKE"

# ---- 3. Build --------------------------------------------------------------
banner "Building (j=${JOBS})"

cmake --build "$BUILD" -j "$JOBS"

BIN="${BUILD}/bin/tensor-opt"
echo
if [[ -x "$BIN" ]]; then
  echo -e "${G}Build succeeded.${N}"
  echo "  Binary: $BIN"
else
  echo -e "${R}Build finished but tensor-opt not found at $BIN${N}"
  exit 1
fi
echo
echo "Next steps:"
echo "  ./run.sh               — run the full test suite"
echo "  ./scripts/evaluation.sh — generate evaluation metrics"
