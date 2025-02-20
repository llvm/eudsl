include "OpBase.td"
include "AttrTypeBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

class Test_Type<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<Test_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Test_Attr<string name, string attrMnemonic> : AttrDef<Test_Dialect, name> {
  let mnemonic = attrMnemonic;
}

class Test_Op<string mnemonic, list<Trait> traits = []>
    : Op<Test_Dialect, mnemonic, traits>;

def Test_SingletonAType : Test_Type<"SingletonAType", "singleton_a"> {}
def Test_SingletonBType : Test_Type<"SingletonBType", "singleton_b"> {}
def Test_SingletonCType : Test_Type<"SingletonCType", "singleton_c"> {}
def Test_TestAttr : Test_Attr<"Test", "test"> {}


def Test_AndOp : Test_Op<"and"> {
  let arguments = (ins AllOfType<[Test_SingletonAType, AnyType]>:$in);
}


def Test_AnyOp : Test_Op<"any"> {
  let arguments = (ins AnyType:$in);
}

def Test_AttributesOp : Test_Op<"attributes"> {
  let arguments = (ins I16Attr:$int_attr,
                       Test_TestAttr:$test_attr);
}

def Test_ConfinedOp : Test_Op<"confined"> {
  let arguments = (ins ConfinedType<AnyType, [CPred<"::llvm::isa<::mlir::TensorType>($_self)">]>:$tensor,
                       ConfinedType<AnyType, [And<[CPred<"::llvm::isa<::mlir::VectorType>($_self)">
                                             , CPred<"::llvm::cast<::mlir::VectorType>($_self).getRank() > 0">]>]>:$vector);
}

def Test_Integers : Test_Op<"integers"> {
  let arguments = (ins AnyI8:$any_int,
                       AnyInteger:$any_integer);
}

def Test_OrOp : Test_Op<"or"> {
  let arguments = (ins AnyTypeOf<[Test_SingletonAType, Test_SingletonBType, Test_SingletonCType]>:$in);
}

def Test_RegionsOp : Test_Op<"regions"> {
  let regions = (region AnyRegion:$any_region,
                        SizedRegion<1>:$single_block_region);
}

def Test_TypesOp : Test_Op<"types"> {
  let arguments = (ins I32:$a,
                       SI64:$b,
                       UI8:$c,
                       Index:$d,
                       F32:$e,
                       NoneType:$f,
                       Complex<F8E4M3FN>);
}

def Test_VariadicityOp : Test_Op<"variadicity"> {
  let arguments = (ins Variadic<Test_SingletonAType>:$variadic,
                       Optional<Test_SingletonBType>:$optional,
                       Test_SingletonCType:$required);
}

def TritonGPU_AttrTrait : AttrInterface<"TritonGPU_AttrTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let methods = [
  ];
}

class TritonGPU_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = Test_Dialect,
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, !listconcat([TritonGPU_AttrTrait], traits), baseCppClass> {
  let attrName = "triton.gpu." # attrMnemonic;
}

//===----------------------------------------------------------------------===//
// CTA Layout
//===----------------------------------------------------------------------===//

def CTALayoutAttr : TritonGPU_Attr<"CTALayout", "cta_layout"> {
  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$CTAsPerCGA,
    ArrayRefParameter<"unsigned">:$CTASplitNum,
    ArrayRefParameter<"unsigned">:$CTAOrder
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<unsigned>":$CTAsPerCGA,
                     "ArrayRef<unsigned>":$CTASplitNum,
                     "ArrayRef<unsigned>":$CTAOrder), [{
        if (llvm::all_of(CTAsPerCGA, [](unsigned x) { return x == 1; })) {
          SmallVector<unsigned> order;
          for (int i = CTAsPerCGA.size() - 1; i >= 0; --i)
            order.push_back(i);
          return $_get(context, CTAsPerCGA, CTASplitNum, order);
        }
        return $_get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
    }]>,
  ];

  let extraClassDeclaration = [{
    static CTALayoutAttr getDefault(MLIRContext *context, int rank) {
      SmallVector<unsigned> CTAsPerCGA(rank, 1);
      SmallVector<unsigned> CTASplitNum(rank, 1);
      SmallVector<unsigned> CTAOrder;
      for (int i = rank - 1; i >= 0; --i)
        CTAOrder.push_back(i);
      return get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
    }
  }];

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}