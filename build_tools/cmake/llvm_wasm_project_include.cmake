# everything in here gets run at the end of the project loading
# https://github.com/emscripten-core/emscripten/issues/15276#issuecomment-1039349267
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
# so that the correct LLVM_ABI macros get set
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ELF__")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ELF__")
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-sSIDE_MODULE=1 -sEXPORT_ALL=1 --emit-symbol-map")
set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "-sSIDE_MODULE=1 -sEXPORT_ALL=1 --emit-symbol-map")
# prevent duplicated libs from being linked
# will need to be renamed to DEDUPLICATION when CMake version catches up
# https://github.com/Kitware/CMake/commit/5617c34c3135f7ec203d5a48b803eb323f458bc3#diff-17fc647759070cdaddd99e9ad994c4478860d9c301ed9a9a9f061a8825c8b690L20
set(CMAKE_C_LINK_LIBRARIES_PROCESSING ORDER=FORWARD UNICITY=ALL)
set(CMAKE_CXX_LINK_LIBRARIES_PROCESSING ORDER=FORWARD UNICITY=ALL)
# hack to prevent -Bsymbolic
set(LLVM_LINKER_IS_SOLARISLD_ILLUMOS ON)
set(CMAKE_SHARED_LIBRARY_SUFFIX ".wasm")
set(CMAKE_STRIP FALSE)