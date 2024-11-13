set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(eudsl-tblgen EUDSL
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT EUDSL
  eudsl-tblgen.cpp
)

target_link_libraries(eudsl-tblgen
  PRIVATE
  MLIRTblgenLib)

mlir_check_all_link_libraries(eudsl-tblgen)

install(TARGETS eudsl-tblgen COMPONENT eudsl-tblgen)