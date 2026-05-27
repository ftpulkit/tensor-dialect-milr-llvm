add_mlir_dialect_library(MLIRTensorExt
  TensorDialect.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/include/TensorDialect

  DEPENDS
    TensorExtOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces
)
