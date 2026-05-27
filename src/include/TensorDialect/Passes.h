//===- Passes.h - tensor_ext conversion pass declarations -------*- C++ -*-===//

#ifndef TENSOR_DIALECT_PASSES_H
#define TENSOR_DIALECT_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
class RewritePatternSet;
class TypeConverter;

namespace tensor_ext {

/// Populate tc with:  !tensor_ext.array<...> -> memref<...>  (plus identity).
void populateTensorExtTypeConverter(TypeConverter &tc);

/// Add the five ConversionPatterns + func-sig rewrite patterns to 'patterns'.
void populateTensorExtToMemRefPatterns(TypeConverter &tc,
                                       RewritePatternSet &patterns);

/// Create the --convert-tensor-ext-to-memref pass.
std::unique_ptr<Pass> createConvertTensorExtToMemRefPass();

/// Register the pass with the global pass registry (called from main).
void registerConvertTensorExtToMemRefPass();

} // namespace tensor_ext
} // namespace mlir

#endif // TENSOR_DIALECT_PASSES_H
