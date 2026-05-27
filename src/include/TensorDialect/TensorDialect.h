//===- TensorDialect.h - tensor_ext dialect header ----------------*- C++ -*-===//
#ifndef TENSOR_DIALECT_TENSORDIALECT_H
#define TENSOR_DIALECT_TENSORDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect class (generated)
#include "TensorDialect/TensorOpsDialect.h.inc"

// Type declarations (generated)
#define GET_TYPEDEF_CLASSES
#include "TensorDialect/TensorOpsTypes.h.inc"

// Op declarations (generated)
#define GET_OP_CLASSES
#include "TensorDialect/TensorOps.h.inc"

#endif // TENSOR_DIALECT_TENSORDIALECT_H