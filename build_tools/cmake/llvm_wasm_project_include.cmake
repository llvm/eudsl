# everything in here gets run at the end of the project loading
# https://github.com/emscripten-core/emscripten/issues/15276#issuecomment-1039349267
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
# __ELF__ so that the correct LLVM_ABI macros get set
# https://github.com/WebAssembly/tool-conventions/blob/main/BasicCABI.md#function-arguments-and-return-values
# Similarly, types can either be returned directly from WebAssembly functions or returned indirectly via a pointer parameter prepended to the parameter list.
# https://github.com/llvm/llvm-project/blob/6d973b4548e281d0b8e75e85833804bb45b6a0e8/clang/lib/CodeGen/Targets/WebAssembly.cpp#L135
# https://github.com/llvm/llvm-project/commit/c285307e1457c4db2346443a4336e672d7487111#diff-b83bb889340990fea25762060e144b5cd4b4652a6fa737aaea9555d456344219
# https://github.com/llvm/llvm-project/blame/63534779b4ef1816e2961246011e2ec3be110d27/clang/lib/CodeGen/TargetInfo.cpp
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ELF__")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ELF__")
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-sLINKABLE=1 -sEXPORT_ALL=1")
set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "-sLINKABLE=1 -sEXPORT_ALL=1")
# prevent duplicated libs from being linked
# will need to be renamed to DEDUPLICATION when CMake version catches up
# https://github.com/Kitware/CMake/commit/5617c34c3135f7ec203d5a48b803eb323f458bc3#diff-17fc647759070cdaddd99e9ad994c4478860d9c301ed9a9a9f061a8825c8b690L20
set(CMAKE_C_LINK_LIBRARIES_PROCESSING ORDER=FORWARD UNICITY=ALL)
set(CMAKE_CXX_LINK_LIBRARIES_PROCESSING ORDER=FORWARD UNICITY=ALL)

## hack to prevent -Bsymbolic
#set(LLVM_LINKER_IS_SOLARISLD_ILLUMOS ON)

set(CMAKE_SHARED_LIBRARY_SUFFIX ".wasm")
set(CMAKE_STRIP FALSE)
set(LLVM_NO_DEAD_STRIP ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_VISIBILITY_PRESET default)
