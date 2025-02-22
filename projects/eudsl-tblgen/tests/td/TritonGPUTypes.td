#ifndef TRITONGPU_TYPES
#define TRITONGPU_TYPES

include "AttrTypeBase.td"
include "BuiltinTypeInterfaces.td"
include "TritonGPUDialect.td"

class TTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonGPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def TTG_AsyncToken : TTG_TypeDef<"AsyncToken", "async.token", []> {
}

// Memory descriptor type.
def TTG_MemDescType : TTG_TypeDef<"MemDesc", "memdesc", [ShapedTypeInterface]> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "Attribute":$encoding,
    "Attribute":$memorySpace,
    "bool":$mutableMemory,
    ArrayRefParameter<"int64_t">:$allocShape
  );

  let extraClassDeclaration = [{
    MemDescType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                          Type elementType) const {
      return MemDescType::get(shape.value_or(getShape()), elementType, getEncoding(), getMemorySpace(), getMutableMemory(), getAllocShape());
    }

    bool hasRank() const { return true; }
  }];

  let builders = [
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "Type":$elementType,
            "Attribute":$encoding,
            "Attribute":$memorySpace
        ), [{
            return $_get(elementType.getContext(), shape, elementType, encoding, memorySpace, /*mutableMemory=*/false, /*allocShape=*/shape);
        }]>,
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "Type":$elementType,
            "Attribute":$encoding,
            "Attribute":$memorySpace,
            "bool":$mutableMemory
        ), [{
            return $_get(elementType.getContext(), shape, elementType, encoding, memorySpace, mutableMemory, /*allocShape=*/shape);
        }]>,
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "Type":$elementType,
            "Attribute":$encoding,
            "Attribute":$memorySpace,
            "bool":$mutableMemory,
            "llvm::ArrayRef<int64_t>":$allocShape
        ), [{
            return $_get(elementType.getContext(), shape, elementType, encoding, memorySpace, mutableMemory, allocShape);
        }]>

    ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


#endif
