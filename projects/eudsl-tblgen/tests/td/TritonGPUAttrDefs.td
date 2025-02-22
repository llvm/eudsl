#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "AttrTypeBase.td"
include "TritonInterfaces.td"
include "TritonGPUDialect.td"

//===----------------------------------------------------------------------===//
// TritonGPU Attribute Definitions
//===----------------------------------------------------------------------===//
def TritonGPU_AttrTrait : AttrInterface<"TritonGPU_AttrTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let methods = [
  ];
}

class TritonGPU_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonGPU_Dialect,
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, !listconcat([TritonGPU_AttrTrait], traits), baseCppClass> {

  let attrName = "triton.gpu." # attrMnemonic;

  code extraBaseClassDeclaration = [{
  }];
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


def LayoutEncodingTrait : AttrInterface<"LayoutEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let description = [{
    Common trait for all TTGIR layouts.
  }];
  let methods = [
    InterfaceMethod<"Get the shape of the CTAs per CGA.",
                    "SmallVector<unsigned>",
                    "getCTAsPerCGA">,
    InterfaceMethod<"Get the order of the CTAs per CGA. The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getCTAOrder">,
    InterfaceMethod<"Each CTA processes 1/CTASplitNum of the tensor.",
                    "SmallVector<unsigned>",
                    "getCTASplitNum">,
  ];
}

//===----------------------------------------------------------------------===//
// Shared Layout Encoding
//===----------------------------------------------------------------------===//

def SharedEncodingTrait : AttrInterface<"SharedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let description = [{
    Common trait describing shared memory.
  }];
  let methods = [
    InterfaceMethod<"Return the default alignment for the layout.",
                    "int32_t",
                    "getAlignment">,
  ];
}

def SwizzledSharedEncodingAttr :
  TritonGPU_Attr<"SwizzledSharedEncoding", "swizzled_shared_encoding", [SharedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "swizzled_shared";

  // swizzle info: vec, perPhase, maxPhase
  // order: the fastest-changing axis first
  let parameters = (
    ins
    "unsigned":$vec,
    "unsigned":$perPhase,
    "unsigned":$maxPhase,
    ArrayRefParameter<"unsigned">:$order,
    "CTALayoutAttr":$CTALayout
  );

  let builders = [
    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "unsigned":$typeWidthInBit), [{
        bool needTrans = false; // default value
        return get(context, dotOpEnc, shape, order, CTALayout, typeWidthInBit, needTrans);
    }]>,

    // TODO(jlebar): This should not be an overload of
    // SwizzledSharedEncodingAttr::get().  It's misleading, because it does a bunch of
    // nontrivial work based on the given dotOpEnc.
    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "unsigned":$typeWidthInBit,
                     "bool":$needTrans), [{

    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CTALayout, bitwidth);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy,
                     "bool":$needTrans), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CTALayout, bitwidth, needTrans);
    }]>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    int32_t getAlignment() const;
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTAOrder() const;
    SmallVector<unsigned> getCTASplitNum() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def NVMMASharedEncodingAttr :
  TritonGPU_Attr<"NVMMASharedEncoding", "nvmma_shared_encoding", [SharedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "nvmma_shared";

  let parameters = (
    ins
    "unsigned":$swizzlingByteWidth,
    "bool":$transposed,
    "unsigned":$elementBitWidth,
    "bool":$fp4Padded,
    "CTALayoutAttr":$CTALayout
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy,
                     "bool": $fp4Padded), [{

    }]>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    int32_t getAlignment() const;
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTAOrder() const;
    SmallVector<unsigned> getCTASplitNum() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Distributed Layout Encoding
//===----------------------------------------------------------------------===//
def DistributedEncodingTrait : AttrInterface<"DistributedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let methods = [
    InterfaceMethod<"Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getRepOrder">,
    InterfaceMethod<"Return total element size per thread.",
                    "unsigned",
                    "getTotalElemsPerThread",
                     (ins "ArrayRef<int64_t>":$shape),
                     /*defaultImplementation=*/[{
                         return toLinearEncoding($_self, shape).getTotalElemsPerThread(shape);
                     }]>,
    InterfaceMethod<"Return element size per thread in each dimension.",
                    "SmallVector<unsigned>",
                    "getElemsPerThread",
                     (ins "ArrayRef<int64_t>":$shape),
                     /*defaultImplementation=*/[{
                         return toLinearEncoding($_self, shape).getElemsPerThread(shape);
                     }]>,
    // Interface for the meta information about the multiple thread hierarchy.
    InterfaceMethod<"Get the shape of the warps per CTA.",
                    "SmallVector<unsigned>",
                    "getWarpsPerCTA">,

    InterfaceMethod<"Get the order of the warps per CTA. The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getWarpOrder">,

    InterfaceMethod<"Get the shape of the threads per warp",
                    "SmallVector<unsigned>",
                    "getThreadsPerWarp">,

    InterfaceMethod<"Get the order of the threads per warp. The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getThreadOrder">,

    InterfaceMethod<"Get the shape of the values per thread.",
                    "SmallVector<unsigned>",
                    "getSizePerThread">,

    InterfaceMethod<"Gets the number of contiguous elements per thread.",
                    "SmallVector<unsigned>",
                    "getContigPerThread">,
    InterfaceMethod<"Convert to LinearLayout.",
                    "LinearLayout",
                    "toLinearLayout",
                    (ins "ArrayRef<int64_t>":$shape)>
  ];
}

class DistributedEncoding<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonGPU_Dialect>
  : TritonGPU_Attr<name, attrMnemonic, !listconcat([DistributedEncodingTrait, LayoutEncodingTrait], traits), dialect> {

  code extraDistributedDeclaration  = extraBaseClassDeclaration # [{
    // Implemented in subclasses
    SmallVector<unsigned> getRepOrder() const;
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTAOrder() const;
    SmallVector<unsigned> getCTASplitNum() const;
    SmallVector<unsigned> getWarpsPerCTA() const;
    SmallVector<unsigned> getWarpOrder() const;
    SmallVector<unsigned> getThreadsPerWarp() const;
    SmallVector<unsigned> getThreadOrder() const;

    SmallVector<unsigned> getSizePerThread() const;

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  }];
}

//===----------------------------------------------------------------------===//
// Linear Layout Encoding
//===----------------------------------------------------------------------===//

def LinearLayoutParam : AttrOrTypeParameter<"LinearLayout",
                                            "linear layout"> {
  let cppAccessorType = "const LinearLayout &";
}

def LinearEncodingAttr : DistributedEncoding<"LinearEncoding", "linear_encoding"> {
  let mnemonic = "linear";

  let description = [{
    See the docs in LinearLayout.h for the definition of linear layouts.
  }];

  let parameters = (ins LinearLayoutParam:$linearLayout);

  let extraClassDeclaration = extraDistributedDeclaration # [{
    // Generic distributed encoding methods
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const;

    SmallVector<unsigned> getContigPerThread() const;
    SmallVector<unsigned> getOrder() const;

    // Generalizes get{Warp,Thread,CTA}Order to linear layouts.
    // Returns the order of the dimensions `dimName` of the layout.
    // If more than dimension is of size one, it uses defaultOrder to determine
    // the order of the dimensions of size one.
    SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const;

    // Generalizes getThreadsPerWarp, getWarpsPerCTA, getCTAsPerCGA to linear layouts.
    // Returns the bases of the dimensions `dimName` of the layout.
    // If skipBroadcast is false, we count a base zero
    SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                      bool skipBroadcast = true) const;
  }];

  let genVerifyDecl = 1;
  // Example of assembly format:
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  let hasCustomAssemblyFormat = 1;
}


//===----------------------------------------------------------------------===//
// Blocked Layout Encoding
//===----------------------------------------------------------------------===//

def BlockedEncodingAttr : DistributedEncoding<"BlockedEncoding", "blocked_encoding"> {
  let mnemonic = "blocked";

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread__,
    ArrayRefParameter<"unsigned">:$threadsPerWarp__,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    ArrayRefParameter<"unsigned">:$order, // the fastest-changing axis first

    // CTALayout is optional in the textual IR.  If omitted, we infer it to be a
    // single CTA (so CTAsPerCGA = [1,...,1], CTASplitNum = [1,...,1],
    // CTAOrder=[n,n-1,...,0]).
    "CTALayoutAttr":$CTALayout
  );
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "CTALayoutAttr":$CTALayout), [{

    }]>,

    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "unsigned":$numCTAs), [{

    }]>
  ];

  let extraClassDeclaration = extraDistributedDeclaration # [{
    SmallVector<unsigned> getContigPerThread() {
      // Block encoding is dense stride layout. The elements per thread are contiguous.
      return getSizePerThread();
    };
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// MMA Layout Encoding
//===----------------------------------------------------------------------===//

def MmaEncodingTrait : AttrInterface<"MmaEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [

    InterfaceMethod<"Return whether the layout support reduction op.",
                    "bool",
                    "supportReduction">,

    InterfaceMethod<"Return size per thread for dot operands.",
                    "SmallVector<unsigned>",
                    "getSizePerThreadForOperand",
                    (ins "int":$opIdx,
                         "int":$kWidth)>,

    InterfaceMethod<"Return the number of threads per warp for dot operands.",
                    "SmallVector<unsigned>",
                    "getThreadsPerWarpForOperand",
                    (ins "int":$opIdx)>,

    InterfaceMethod<"Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getRepOrderForOperand",
                    (ins "int":$opIdx)>,
  ];
}

def AMDMfmaEncodingAttr : DistributedEncoding<"AMDMfmaEncoding", "amd_mfma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "amd_mfma";

  let parameters = (
    ins
    "unsigned": $versionMajor,
    "unsigned": $versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "unsigned":$MDim,
    "unsigned":$NDim,
    "bool":$isTransposed,
    "CTALayoutAttr":$CTALayout
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool supportReduction() const {
      return true;
    }
    SmallVector<unsigned> getSizePerThreadForOperand(int kWidth, int opIdx) const;
    SmallVector<int64_t> getInstrShapeForOperand(int kWidth, int opIdx) const;
    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> operandShape, int kWidth, int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
    SmallVector<unsigned> getThreadsPerWarpForOperand(int opIdx) const;

    SmallVector<unsigned> getContigPerThread() {
      auto rank = getWarpsPerCTA().size();
      SmallVector<unsigned> contigPerThread(rank, 1);
      if (getIsTransposed())
        contigPerThread[rank - 1] = 4;
      else
        contigPerThread[rank - 2] = 4;
      return contigPerThread;
    };

  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def AMDWmmaEncodingAttr : DistributedEncoding<"AMDWmmaEncoding", "amd_wmma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "amd_wmma";

  let parameters = (
    ins
    "unsigned": $version,
    "bool":$isTransposed,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "CTALayoutAttr":$CTALayout
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool supportReduction() const {
      return true;
    }
    SmallVector<unsigned> getSizePerThreadForOperand(int kWidth, int opIdx) const;
    SmallVector<int64_t> getElemsPerInstrForOperands() const;
    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> operandShape,
                                          Type elemType, int kWidth, int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
    SmallVector<unsigned> getThreadsPerWarpForOperand(int opIdx) const;
    static SmallVector<unsigned> getMNKDimPerInstr();

    SmallVector<unsigned> getContigPerThread() {
      auto rank = getWarpsPerCTA().size();
      assert(rank == 2 || rank == 3);
      SmallVector<unsigned> contigPerThread(rank, 1);
      if (getVersion() == 2) {
        contigPerThread[rank - 2] = 8;
      }
      return contigPerThread;
    };
  }];
}

def NvidiaMmaEncodingAttr : DistributedEncoding<"NvidiaMmaEncoding", "nvidia_mma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "nvidia_mma";

  let parameters = (
    ins
    "unsigned":$versionMajor,
    "unsigned":$versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "CTALayoutAttr":$CTALayout,
    ArrayRefParameter<"unsigned">:$instrShape
  );


  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool isVolta() const;
    bool isTuring() const;
    bool isAmpere() const;
    bool isHopper() const;

    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> shape,
                                          int bitwidth, int kWidth,
                                          int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
    SmallVector<unsigned> getThreadsPerWarpForOperand(int opIdx) const;

    bool supportReduction() const {
      if (isAmpere() || isHopper()) {
        return true;
      }
      return false;
    };
    SmallVector<unsigned> getSizePerThreadForOperand(int kWidth, int opIdx) const;

    SmallVector<unsigned> getContigPerThread() {
      assert(isAmpere() || isHopper());
      auto rank = getWarpsPerCTA().size();
      SmallVector<unsigned> contigPerThread(rank, 1);
      contigPerThread[rank - 1] = 2;
      return contigPerThread;
    };

  }];

  let hasCustomAssemblyFormat = 1;
}

def SliceEncodingAttr : DistributedEncoding<"SliceEncoding", "slice_encoding"> {
  let mnemonic = "slice";

  let parameters = (
    ins
    "unsigned":$dim,
    // TODO: constraint here to only take distributed encodings
    "Attribute":$parent
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    template<class T>
    SmallVector<T> paddedShape(ArrayRef<T> shape) const;

    SmallVector<unsigned> getContigPerThread() {
      auto parentLayout = mlir::cast<DistributedEncodingTrait>(getParent());
      auto parentContigPerThread = parentLayout.getContigPerThread();
      parentContigPerThread.erase(parentContigPerThread.begin() + getDim());
      return parentContigPerThread;
    };
  }];

  let hasCustomAssemblyFormat = 1;
}

def DotOperandEncodingAttr : DistributedEncoding<"DotOperandEncoding", "dot_operand_encoding"> {
  let mnemonic = "dot_op";

  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent,
    DefaultValuedParameter<"unsigned", "0">:$kWidth
  );

  let builders = [
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "Type":$eltTy), [{
      NvidiaMmaEncodingAttr parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent);
      if (!parentAttr || (!parentAttr.isAmpere() && !parentAttr.isHopper()))
        return $_get(context, opIdx, parent, 0);
      // For MMAV2 and V3
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned kWidth = 32 / bitwidth;
      return $_get(context, opIdx, parent, kWidth);
    }]>
  ];

  let assemblyFormat = "`<` `{` struct(params) `}` `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = extraDistributedDeclaration # [{
    SmallVector<unsigned> getContigPerThread() {
      auto rank = getWarpsPerCTA().size();
      assert(rank == 2 || rank == 3);
      SmallVector<unsigned> contigPerThread(rank, 1);
      auto kWidth = getKWidth();
      assert(kWidth != 0 && "Do not support kWidth=0");
      if (getOpIdx() == 0)
        contigPerThread[rank - 1] = kWidth;
      else
        contigPerThread[rank - 2] = kWidth;
      return contigPerThread;
    };
  }];
}

def TTG_SharedMemorySpace : AttrDef<TritonGPU_Dialect, "SharedMemorySpace"> {
  let mnemonic = "shared_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to shared memory.
  }];
}
#endif
