# Collect all upstream dialect and conversion libraries so tensor-opt can run
# any standard pass alongside our custom one.
get_property(dialect_libs    GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs  GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(tensor-opt tensor-opt.cpp)
llvm_update_compile_flags(tensor-opt)

target_link_libraries(tensor-opt
  PRIVATE
    # All upstream dialects / conversions / extensions
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}

    # Our libraries
    MLIRTensorExt
    MLIRTensorExtConversion

    # MLIR tooling core
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIROptLib
    MLIRParser
    MLIRTransforms
)
