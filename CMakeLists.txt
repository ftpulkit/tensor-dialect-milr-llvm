# ==============================================================================
#  tensor_ext MLIR dialect project  —  top-level CMakeLists.txt
#
#  Quick start (Ubuntu 22.04 / 24.04):
#    sudo apt install cmake ninja-build clang-18 llvm-18-dev libmlir-18-dev mlir-18-tools
#    ./build.sh          # configures + builds
#    ./run.sh            # runs all tests
# ==============================================================================

cmake_minimum_required(VERSION 3.20)
project(TensorExtDialect LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Locate MLIR / LLVM ----------------------------------------------------

find_package(MLIR REQUIRED CONFIG)

message(STATUS "MLIR found at: ${MLIR_DIR}")
message(STATUS "LLVM found at: ${LLVM_DIR}")

# Place all executables in build/bin so scripts can find them predictably.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# ---- Include paths ---------------------------------------------------------

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# Source headers (our .h files).
include_directories(${PROJECT_SOURCE_DIR}/src/include)
# Generated headers (.inc files from mlir-tblgen).
include_directories(${PROJECT_BINARY_DIR}/src/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# ---- Sub-directories -------------------------------------------------------

add_subdirectory(src/include/TensorDialect)   # runs mlir-tblgen
add_subdirectory(src/lib/TensorDialect)        # MLIRTensorExt library
add_subdirectory(src/lib/Conversion)           # MLIRTensorExtConversion library
add_subdirectory(src/tools/tensor-opt)         # tensor-opt binary
