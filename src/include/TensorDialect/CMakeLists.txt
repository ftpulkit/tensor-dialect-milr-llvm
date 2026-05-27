# Run mlir-tblgen to generate .inc files from TensorOps.td
# Output lands in ${CMAKE_CURRENT_BINARY_DIR}/ which is on the include path
# because of `include_directories(${PROJECT_BINARY_DIR}/src/include)` in the
# top-level CMakeLists.txt.

set(LLVM_TARGET_DEFINITIONS TensorOps.td)

mlir_tablegen(TensorOpsDialect.h.inc   -gen-dialect-decls  -dialect=tensor_ext)
mlir_tablegen(TensorOpsDialect.cpp.inc -gen-dialect-defs   -dialect=tensor_ext)
mlir_tablegen(TensorOps.h.inc          -gen-op-decls)
mlir_tablegen(TensorOps.cpp.inc        -gen-op-defs)
mlir_tablegen(TensorOpsTypes.h.inc     -gen-typedef-decls  -typedefs-dialect=tensor_ext)
mlir_tablegen(TensorOpsTypes.cpp.inc   -gen-typedef-defs   -typedefs-dialect=tensor_ext)

add_public_tablegen_target(TensorExtOpsIncGen)
