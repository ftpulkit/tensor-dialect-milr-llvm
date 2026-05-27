//===- TensorOps.td - tensor_ext dialect definition -------*- tablegen -*-===//
#ifndef TENSOR_OPS_TD
#define TENSOR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def TensorExt_Dialect : Dialect {
  let name = "tensor_ext";
  let summary = "High-level tensor dialect for teaching/demo.";
  let cppNamespace = "::mlir::tensor_ext";
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class TensorExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorExt_Dialect, mnemonic, traits>;

class TensorExt_Type<string name,
                     string typeMnemonic,
                     list<Trait> traits = []>
    : TypeDef<TensorExt_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ArrayType — !tensor_ext.array<4x8xf32>
//===----------------------------------------------------------------------===//

def TensorExt_ArrayType : TensorExt_Type<"Array", "array"> {
  let summary = "Statically-shaped N-D array.";

  let parameters = (ins
    ArrayRefParameter<"int64_t", "dimension sizes"> : $shape,
    "::mlir::Type" : $elementType
  );

  // Calls parseArrayShape / printArrayShape defined in TensorDialect.cpp
  let assemblyFormat =
      "`<` custom<ArrayShape>($shape, $elementType) `>`";

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }
    int64_t getNumElements() const;
  }];
}

def TensorExt_AnyArray : Type<
    CPred<"::llvm::isa<::mlir::tensor_ext::ArrayType>($_self)">,
    "a tensor_ext array",
    "::mlir::tensor_ext::ArrayType">;

//===----------------------------------------------------------------------===//
// tensor_ext.alloc
//===----------------------------------------------------------------------===//

def TensorExt_AllocOp
    : TensorExt_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {

  let summary = "Allocate a tensor.";

  let arguments = (ins);

  let results = (outs TensorExt_AnyArray:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// tensor_ext.load
//===----------------------------------------------------------------------===//

def TensorExt_LoadOp
    : TensorExt_Op<"load", [MemoryEffects<[MemRead]>]> {

  let summary = "Load one element from a tensor.";

  let arguments = (ins
    TensorExt_AnyArray:$tensor,
    Variadic<Index>:$indices
  );

  let results = (outs AnyType:$result);

  // tensor_ext.load %A[%i, %j]
  //   : !tensor_ext.array<4x8xf32> -> f32
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tensor_ext.store
//===----------------------------------------------------------------------===//

def TensorExt_StoreOp
    : TensorExt_Op<"store", [MemoryEffects<[MemWrite]>]> {

  let summary = "Store one element into a tensor.";

  let arguments = (ins
    AnyType:$value,
    TensorExt_AnyArray:$tensor,
    Variadic<Index>:$indices
  );

  let results = (outs);

  // tensor_ext.store %v, %A[%i, %j]
  //   : f32, !tensor_ext.array<4x8xf32>
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tensor_ext.slice
//===----------------------------------------------------------------------===//

def TensorExt_SliceOp
    : TensorExt_Op<"slice",
        [MemoryEffects<[MemRead, MemWrite]>]> {

  let useCustomPropertiesEncoding = 0;

  let summary = "Copy a sub-region into a fresh tensor.";

  let arguments = (ins
    TensorExt_AnyArray:$source,
    DenseI64ArrayAttr:$offsets,
    DenseI64ArrayAttr:$sizes
  );

  let results = (outs TensorExt_AnyArray:$result);

  let assemblyFormat = [{
    $source `offsets` `=` $offsets
    `sizes` `=` $sizes
    attr-dict `:` type($source)
    `to` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tensor_ext.transpose
//===----------------------------------------------------------------------===//

def TensorExt_TransposeOp
    : TensorExt_Op<"transpose",
        [MemoryEffects<[MemRead, MemWrite]>]> {

  let useCustomPropertiesEncoding = 0;

  let summary = "Permute the dimensions of a tensor.";

  let arguments = (ins
    TensorExt_AnyArray:$source,
    DenseI64ArrayAttr:$permutation
  );

  let results = (outs TensorExt_AnyArray:$result);

  let assemblyFormat = [{
    $source `permutation` `=` $permutation
    attr-dict `:` type($source)
    `to` type($result)
  }];

  let hasVerifier = 1;
}

#endif // TENSOR_OPS_TD