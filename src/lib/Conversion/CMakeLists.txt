add_mlir_library(MLIRTensorExtConversion
  TensorExtToMemRef.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/include/TensorDialect

  DEPENDS
    TensorExtOpsIncGen

  LINK_LIBS PUBLIC
    MLIRTensorExt
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRSupport
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRTransforms
)
