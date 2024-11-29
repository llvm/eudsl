#include "mlir/IR/Action.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/InterfaceSupport.h"
#include "llvm/ADT/APSInt.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/unique_ptr.h>

using namespace llvm;
using namespace mlir;

namespace nb = nanobind;
using namespace nb::literals;

template <>
struct nb::detail::type_caster<StringRef> {
  NB_TYPE_CASTER(StringRef, const_name("str"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    Py_ssize_t size;
    const char *str = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
    if (!str) {
      PyErr_Clear();
      return false;
    }
    value = StringRef(str, (size_t)size);
    return true;
  }

  static handle from_cpp(StringRef value, rv_policy, cleanup_list *) noexcept {
    return PyUnicode_FromStringAndSize(value.data(), value.size());
  }
};

// hack to prevent nanobind from trying to detail::wrap_copy<T>
struct DialectRegistry_ : DialectRegistry {
  DialectRegistry_(const DialectRegistry_ &) = delete;
  DialectRegistry_() : DialectRegistry() {}
};

NB_MODULE(eudsl_ext, m) {
  nb::class_<TypeID>(m, "TypeID");
  nb::class_<LogicalResult>(m, "LogicalResult");

  nb::class_<AbstractAttribute>(m, "AbstractAttribute")
      .def_static("lookup", nb::overload_cast<TypeID, MLIRContext *>(
                                &AbstractAttribute::lookup))
      .def_static("lookup", nb::overload_cast<StringRef, MLIRContext *>(
                                &AbstractAttribute::lookup))
      //      .def("get",
      //           nb::overload_cast<Dialect &, mlir::detail::InterfaceMap &&,
      //                             AbstractAttribute::HasTraitFn &&,
      //                             AbstractAttribute::WalkImmediateSubElementsFn,
      //                             AbstractAttribute::ReplaceImmediateSubElementsFn,
      //                             TypeID,
      //                             StringRef>(&AbstractAttribute::get))
      .def_prop_ro("dialect", &AbstractAttribute::getDialect)
      //      .def("get_interface", &AbstractAttribute::getInterface)
      .def("has_interface", &AbstractAttribute::hasInterface)
      //      .def("has_trait",
      //      nb::overload_cast<TypeID>(&AbstractAttribute::hasTrait))
      .def("walk_immediate_sub_elements",
           &AbstractAttribute::walkImmediateSubElements)
      .def("replace_immediate_sub_elements",
           &AbstractAttribute::replaceImmediateSubElements)
      .def_prop_ro("type_id", &AbstractAttribute::getTypeID)
      .def("name", &AbstractAttribute::getName);

  nb::class_<Attribute>(m, "Attribute")
      //      .def("operator=", &Attribute::operator=)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("__bool__", &Attribute::operator bool)
      .def_prop_ro("type_id", &Attribute::getTypeID)
      .def_prop_ro("context", &Attribute::getContext)
      .def_prop_ro("dialect", &Attribute::getDialect)
      //      .def("print", &Attribute::print)
      //      .def("print", &Attribute::print)
      .def("dump", &Attribute::dump)
      //      .def("print_stripped", &Attribute::printStripped)
      //      .def("print_stripped", &Attribute::printStripped)
      //            .def("get_as_opaque_pointer",
      //            &Attribute::getAsOpaquePointer)
      .def("get_from_opaque_pointer", &Attribute::getFromOpaquePointer)
      //      .def("has_promise_or_implements_interface",
      //           &Attribute::hasPromiseOrImplementsInterface)
      //      .def("has_trait", &Attribute::hasTrait)
      .def_prop_ro("abstract_attribute", &Attribute::getAbstractAttribute)
      .def("walk_immediate_sub_elements", &Attribute::walkImmediateSubElements)
      .def("replace_immediate_sub_elements",
           &Attribute::replaceImmediateSubElements)
      //      .def("walk", &Attribute::walk)
      //      .def("replace", &Attribute::replace)
      .def("get_impl", &Attribute::getImpl);

  nb::class_<AffineExpr>(m, "AffineExpr");

  nb::class_<AffineMap>(m, "AffineMap")
      .def(nb::init<>())
      .def_static("get", nb::overload_cast<MLIRContext *>(&AffineMap::get))
      .def_static("get", nb::overload_cast<unsigned, unsigned, MLIRContext *>(
                             &AffineMap::get))
      .def_static("get", nb::overload_cast<unsigned, unsigned, AffineExpr>(
                             &AffineMap::get))
      .def_static("get",
                  nb::overload_cast<unsigned, unsigned, ArrayRef<AffineExpr>,
                                    MLIRContext *>(&AffineMap::get))
      .def_static("get_constant_map", &AffineMap::getConstantMap)
      .def_static("get_multi_dim_identity_map",
                  &AffineMap::getMultiDimIdentityMap)
      .def_static("get_minor_identity_map", &AffineMap::getMinorIdentityMap)
      .def_static("get_filtered_identity_map",
                  &AffineMap::getFilteredIdentityMap)
      .def_static("get_permutation_map",
                  nb::overload_cast<ArrayRef<unsigned>, MLIRContext *>(
                      &AffineMap::getPermutationMap))
      .def_static("get_permutation_map",
                  nb::overload_cast<ArrayRef<int64_t>, MLIRContext *>(
                      &AffineMap::getPermutationMap))
      .def_static("get_multi_dim_map_with_targets",
                  &AffineMap::getMultiDimMapWithTargets)
      .def_static(
          "infer_from_expr_list",
          nb::overload_cast<ArrayRef<ArrayRef<AffineExpr>>, MLIRContext *>(
              &AffineMap::inferFromExprList))
      .def_static(
          "infer_from_expr_list",
          nb::overload_cast<ArrayRef<SmallVector<AffineExpr, 4>>,
                            MLIRContext *>(&AffineMap::inferFromExprList))
      .def("get_context", &AffineMap::getContext)
      .def("operator bool", &AffineMap::operator bool)
      .def("operator==", &AffineMap::operator==)
      .def("operator!=", &AffineMap::operator!=)
      .def("is_identity", &AffineMap::isIdentity)
      .def("is_symbol_identity", &AffineMap::isSymbolIdentity)
      .def("is_minor_identity", &AffineMap::isMinorIdentity)
      .def("get_broadcast_dims", &AffineMap::getBroadcastDims)
      .def("is_minor_identity_with_broadcasting",
           &AffineMap::isMinorIdentityWithBroadcasting)
      .def("is_permutation_of_minor_identity_with_broadcasting",
           &AffineMap::isPermutationOfMinorIdentityWithBroadcasting)
      .def("is_empty", &AffineMap::isEmpty)
      .def("is_single_constant", &AffineMap::isSingleConstant)
      .def("is_constant", &AffineMap::isConstant)
      .def("get_single_constant_result", &AffineMap::getSingleConstantResult)
      .def("get_constant_results", &AffineMap::getConstantResults)
      .def("print", &AffineMap::print)
      .def("dump", &AffineMap::dump)
      .def("get_num_dims", &AffineMap::getNumDims)
      .def("get_num_symbols", &AffineMap::getNumSymbols)
      .def("get_num_results", &AffineMap::getNumResults)
      .def("get_num_inputs", &AffineMap::getNumInputs)
      .def("get_results", &AffineMap::getResults)
      .def("get_result", &AffineMap::getResult)
      .def("get_dim_position", &AffineMap::getDimPosition)
      .def("get_result_position", &AffineMap::getResultPosition)
      .def("is_function_of_dim", &AffineMap::isFunctionOfDim)
      .def("is_function_of_symbol", &AffineMap::isFunctionOfSymbol)
      .def("walk_exprs", &AffineMap::walkExprs)
      .def("replace_dims_and_symbols", &AffineMap::replaceDimsAndSymbols)
      .def("replace",
           nb::overload_cast<AffineExpr, AffineExpr, unsigned, unsigned>(
               &AffineMap::replace, nb::const_))
      .def("replace",
           nb::overload_cast<const llvm::DenseMap<AffineExpr, AffineExpr> &>(
               &AffineMap::replace, nb::const_))
      .def("replace",
           nb::overload_cast<const llvm::DenseMap<AffineExpr, AffineExpr> &,
                             unsigned, unsigned>(&AffineMap::replace,
                                                 nb::const_))
      .def("shift_dims", &AffineMap::shiftDims)
      .def("shift_symbols", &AffineMap::shiftSymbols)
      .def("drop_result", &AffineMap::dropResult)
      .def("drop_results", nb::overload_cast<ArrayRef<int64_t>>(
                               &AffineMap::dropResults, nb::const_))
      .def("drop_results", nb::overload_cast<const llvm::SmallBitVector &>(
                               &AffineMap::dropResults, nb::const_))
      .def("insert_result", &AffineMap::insertResult)
      .def("constant_fold", &AffineMap::constantFold)
      .def("partial_constant_fold", &AffineMap::partialConstantFold)
      .def("compose",
           nb::overload_cast<AffineMap>(&AffineMap::compose, nb::const_))
      .def("compose", nb::overload_cast<ArrayRef<int64_t>>(&AffineMap::compose,
                                                           nb::const_))
      .def("get_num_of_zero_results", &AffineMap::getNumOfZeroResults)
      .def("drop_zero_results", &AffineMap::dropZeroResults)
      .def("is_projected_permutation", &AffineMap::isProjectedPermutation)
      .def("is_permutation", &AffineMap::isPermutation)
      .def("get_sub_map", &AffineMap::getSubMap)
      .def("get_slice_map", &AffineMap::getSliceMap)
      .def("get_major_sub_map", &AffineMap::getMajorSubMap)
      .def("get_minor_sub_map", &AffineMap::getMinorSubMap)
      .def("get_largest_known_divisor_of_map_exprs",
           &AffineMap::getLargestKnownDivisorOfMapExprs)
      //      .def("hash_value", &AffineMap::hash_value)
      //      .def("get_as_opaque_pointer", &AffineMap::getAsOpaquePointer)
      .def_static("get_from_opaque_pointer", &AffineMap::getFromOpaquePointer);

  nb::class_<AffineMapAttr, Attribute>(m, "AffineMapAttr")
      .def_prop_ro("affine_map", &AffineMapAttr::getAffineMap)
      .def_ro_static("name", &AffineMapAttr::name)
      .def_ro_static("dialect_name", &AffineMapAttr::dialectName)
      .def_static("get", &AffineMapAttr::get)
      .def_prop_ro("value", &AffineMapAttr::getValue);

  nb::class_<ArrayAttr, Attribute>(m, "ArrayAttr")
      .def("__getitem__", &ArrayAttr::operator[])
      .def("begin", &ArrayAttr::begin)
      .def("end", &ArrayAttr::end)
      .def("size", &ArrayAttr::size)
      .def("empty", &ArrayAttr::empty)
      //      .def("get_as_range", &ArrayAttr::getAsRange)
      //      .def("get_as_value_range", &ArrayAttr::getAsValueRange)
      .def("get", &ArrayAttr::get)
      .def("get_value", &ArrayAttr::getValue);

  nb::class_<DenseElementsAttr, Attribute>(m, "DenseElementsAttr");

  nb::class_<DenseArrayAttr, Attribute>(m, "DenseArrayAttr")
      .def("size", &DenseArrayAttr::size)
      .def("empty", &DenseArrayAttr::empty)
      .def_ro_static("name", &DenseArrayAttr::name)
      .def_ro_static("dialect_name", &DenseArrayAttr::dialectName)
      .def_static("get",
                  nb::overload_cast<mlir::MLIRContext *, Type, int64_t,
                                    llvm::ArrayRef<char>>(&DenseArrayAttr::get))
      .def_static("get", nb::overload_cast<Type, unsigned, ArrayRef<char>>(
                             &DenseArrayAttr::get))
      .def_static("verify", &DenseArrayAttr::verify)
      .def_static("verify_invariants", &DenseArrayAttr::verifyInvariants)
      .def("get_element_type", &DenseArrayAttr::getElementType)
      .def("get_size", &DenseArrayAttr::getSize)
      .def("get_raw_data", &DenseArrayAttr::getRawData);

  nb::class_<DenseIntOrFPElementsAttr, DenseElementsAttr>(
      m, "DenseIntOrFPElementsAttr")
      .def_ro_static("name", &DenseIntOrFPElementsAttr::name)
      .def_ro_static("dialect_name", &DenseIntOrFPElementsAttr::dialectName)
      .def("empty", &DenseIntOrFPElementsAttr::empty)
      .def("get_num_elements", &DenseIntOrFPElementsAttr::getNumElements)
      .def("get_element_type", &DenseIntOrFPElementsAttr::getElementType)
      //      .def("get_values", &DenseIntOrFPElementsAttr::getValues)
      .def("is_splat", &DenseIntOrFPElementsAttr::isSplat)
      .def("size", &DenseIntOrFPElementsAttr::size)
      //      .def("value_begin", &DenseIntOrFPElementsAttr::value_begin)
      //      .def("try_value_begin_impl",
      //           &DenseIntOrFPElementsAttr::try_value_begin_impl)
      .def_static(
          "convert_endian_of_array_ref_for_b_emachine",
          &DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine);
  //      .def_static("convert_endian_of_char_for_b_emachine",
  //                  &DenseIntOrFPElementsAttr::convertEndianOfCharForBEmachine);

  nb::class_<DenseStringElementsAttr>(m, "DenseStringElementsAttr")
      .def("empty", &DenseStringElementsAttr::empty)
      .def("get_num_elements", &DenseStringElementsAttr::getNumElements)
      .def("get_element_type", &DenseStringElementsAttr::getElementType)
      //      .def("get_values", &DenseStringElementsAttr::getValues)
      .def("is_splat", &DenseStringElementsAttr::isSplat)
      .def("size", &DenseStringElementsAttr::size)
      //      .def("value_begin", &DenseStringElementsAttr::value_begin)
      //      .def("try_value_begin_impl",
      //           &DenseStringElementsAttr::try_value_begin_impl)
      .def("get", &DenseStringElementsAttr::get);

  nb::class_<DenseResourceElementsAttr, Attribute>(m,
                                                   "DenseResourceElementsAttr")
      .def_static("get",
                  nb::overload_cast<ShapedType, DenseResourceElementsHandle>(
                      &DenseResourceElementsAttr::get))
      //      .def_static("get",
      //                  nb::overload_cast<ShapedType, StringRef,
      //                  AsmResourceBlob>(
      //                      &DenseResourceElementsAttr::get))
      .def("get_type", &DenseResourceElementsAttr::getType)
      .def("get_raw_handle", &DenseResourceElementsAttr::getRawHandle);

  nb::class_<StringAttr, Attribute>(m, "StringAttr")
      .def_ro_static("name", &StringAttr::name)
      .def_ro_static("dialect_name", &StringAttr::dialectName)
      .def("get_referenced_dialect", &StringAttr::getReferencedDialect)
      .def("operator string_ref", &StringAttr::operator StringRef)
      .def("strref", &StringAttr::strref)
      .def("str", &StringAttr::str)
      //      .def("data", &StringAttr::data)
      .def("size", &StringAttr::size)
      .def("empty", &StringAttr::empty)
      .def("begin", &StringAttr::begin)
      .def("end", &StringAttr::end)
      .def("compare", &StringAttr::compare)
      .def_static("get",
                  nb::overload_cast<const Twine &, Type>(&StringAttr::get))
      .def_static("get", nb::overload_cast<mlir::MLIRContext *, const Twine &>(
                             &StringAttr::get))
      .def_static("get",
                  nb::overload_cast<mlir::MLIRContext *>(&StringAttr::get))
      .def("get_value", &StringAttr::getValue)
      .def("get_type", &StringAttr::getType);

  nb::class_<NamedAttribute>(m, "NamedAttribute")
      .def(nb::init<StringAttr, Attribute>())
      .def("get_name", &NamedAttribute::getName)
      .def("get_name_dialect", &NamedAttribute::getNameDialect)
      .def("get_value", &NamedAttribute::getValue)
      .def("set_name", &NamedAttribute::setName)
      .def("set_value", &NamedAttribute::setValue)
      .def("__lt__<", nb::overload_cast<const NamedAttribute &>(
                          &NamedAttribute::operator<, nb::const_))
      .def("__lt__<",
           nb::overload_cast<StringRef>(&NamedAttribute::operator<, nb::const_))
      .def(nb::self == nb::self)
      .def(nb::self != nb::self);

  nb::class_<Type>(m, "Type")
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("__bool__", &Type::operator bool)
      .def("get_type_id", &Type::getTypeID)
      .def("get_context", &Type::getContext)
      .def("get_dialect", &Type::getDialect)
      .def("is_index", &Type::isIndex)
      .def("is_float4_e2_m1_fn", &Type::isFloat4E2M1FN)
      .def("is_float6_e2_m3_fn", &Type::isFloat6E2M3FN)
      .def("is_float6_e3_m2_fn", &Type::isFloat6E3M2FN)
      .def("is_float8_e5_m2", &Type::isFloat8E5M2)
      .def("is_float8_e4_m3", &Type::isFloat8E4M3)
      .def("is_float8_e4_m3_fn", &Type::isFloat8E4M3FN)
      .def("is_float8_e5_m2_fnuz", &Type::isFloat8E5M2FNUZ)
      .def("is_float8_e4_m3_fnuz", &Type::isFloat8E4M3FNUZ)
      .def("is_float8_e4_m3_b11_fnuz", &Type::isFloat8E4M3B11FNUZ)
      .def("is_float8_e3_m4", &Type::isFloat8E3M4)
      .def("is_float8_e8_m0_fnu", &Type::isFloat8E8M0FNU)
      .def("is_bf16", &Type::isBF16)
      .def("is_f16", &Type::isF16)
      .def("is_tf32", &Type::isTF32)
      .def("is_f32", &Type::isF32)
      .def("is_f64", &Type::isF64)
      .def("is_f80", &Type::isF80)
      .def("is_f128", &Type::isF128)
      .def("is_integer", nb::overload_cast<>(&Type::isInteger, nb::const_))
      .def("is_integer",
           nb::overload_cast<unsigned>(&Type::isInteger, nb::const_))
      .def("is_signless_integer",
           nb::overload_cast<>(&Type::isSignlessInteger, nb::const_))
      .def("is_signless_integer",
           nb::overload_cast<unsigned>(&Type::isSignlessInteger, nb::const_))
      .def("is_signed_integer",
           nb::overload_cast<>(&Type::isSignedInteger, nb::const_))
      .def("is_signed_integer",
           nb::overload_cast<unsigned>(&Type::isSignedInteger, nb::const_))
      .def("is_unsigned_integer",
           nb::overload_cast<>(&Type::isUnsignedInteger, nb::const_))
      .def("is_unsigned_integer",
           nb::overload_cast<unsigned>(&Type::isUnsignedInteger, nb::const_))
      .def("get_int_or_float_bit_width", &Type::getIntOrFloatBitWidth)
      .def("is_signless_int_or_index", &Type::isSignlessIntOrIndex)
      .def("is_signless_int_or_index_or_float",
           &Type::isSignlessIntOrIndexOrFloat)
      .def("is_signless_int_or_float", &Type::isSignlessIntOrFloat)
      .def("is_int_or_index", &Type::isIntOrIndex)
      .def("is_int_or_float", &Type::isIntOrFloat)
      .def("is_int_or_index_or_float", &Type::isIntOrIndexOrFloat)
      //            .def("print", &Type::print)
      //      .def("print", &Type::print)
      .def("dump", &Type::dump)
      //            .def("get_as_opaque_pointer", &Type::getAsOpaquePointer)
      //      .def("get_from_opaque_pointer", &Type::getFromOpaquePointer)
      //      .def("has_promise_or_implements_interface",
      //           &Type::hasPromiseOrImplementsInterface)
      //      .def("has_trait", &Type::hasTrait)
      .def("get_abstract_type", &Type::getAbstractType)
      .def("get_impl", &Type::getImpl)
      .def("walk_immediate_sub_elements", &Type::walkImmediateSubElements)
      .def("replace_immediate_sub_elements",
           &Type::replaceImmediateSubElements);
  //            .def("walk", &Type::walk)
  //            .def("replace", &Type::replace);

  nb::class_<LocationAttr, Attribute>(m, "LocationAttr")
      .def("walk", &LocationAttr::walk);

  nb::class_<Location>(m, "Location")
      .def("get_context", &Location::getContext)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("dump", &Location::dump);

  nb::class_<Builder>(m, "Builder");
  nb::class_<OpBuilder, Builder>(m, "OpBuilder");

  nb::class_<RewritePatternSet>(m, "RewritePatternSet");

  nb::class_<DialectAsmParser>(m, "DialectAsmParser");

  nb::class_<DictionaryAttr, Attribute>(m, "DictionaryAttr")
      .def("get_named",
           nb::overload_cast<StringRef>(&DictionaryAttr::getNamed, nb::const_))
      .def("get_named",
           nb::overload_cast<StringAttr>(&DictionaryAttr::getNamed, nb::const_))
      .def("contains",
           nb::overload_cast<StringRef>(&DictionaryAttr::contains, nb::const_))
      .def("contains",
           nb::overload_cast<StringAttr>(&DictionaryAttr::contains, nb::const_))
      .def("begin", &DictionaryAttr::begin)
      .def("end", &DictionaryAttr::end)
      .def("empty", &DictionaryAttr::empty)
      .def("size", &DictionaryAttr::size)
      .def_static("sort", &DictionaryAttr::sort)
      .def_static("sort_in_place", &DictionaryAttr::sortInPlace)
      .def_static("find_duplicate", &DictionaryAttr::findDuplicate)
      //      .def("get_as", &DictionaryAttr::getAs)
      .def("get_attr",
           nb::overload_cast<StringRef>(&DictionaryAttr::get, nb::const_))
      .def("get_attr",
           nb::overload_cast<StringAttr>(&DictionaryAttr::get, nb::const_))
      .def_static(
          "get",
          nb::overload_cast<mlir::MLIRContext *, ArrayRef<NamedAttribute>>(
              &DictionaryAttr::get))
      .def("get_value", &DictionaryAttr::getValue)
      .def_ro_static("name", &DictionaryAttr::name)
      .def_ro_static("dialect_name", &DictionaryAttr::dialectName);

  nb::class_<FloatAttr, Attribute>(m, "FloatAttr")
      .def_ro_static("name", &FloatAttr::name)
      .def_ro_static("dialect_name", &FloatAttr::dialectName)
      //      .def("get_value_as_double",
      //           nb::overload_cast<>(&FloatAttr::getValueAsDouble))
      .def_static("get_value_as_double", nb::overload_cast<llvm::APFloat>(
                                             &FloatAttr::getValueAsDouble))
      .def_static("get",
                  nb::overload_cast<Type, const APFloat &>(&FloatAttr::get))
      .def_static("get", nb::overload_cast<Type, double>(&FloatAttr::get))
      .def_static("verify", &FloatAttr::verify)
      .def_static("verify_invariants", &FloatAttr::verifyInvariants)
      .def("get_type", &FloatAttr::getType)
      .def("get_value", &FloatAttr::getValue);

  nb::class_<IntegerAttr, Attribute>(m, "IntegerAttr")
      .def_ro_static("name", &IntegerAttr::name)
      .def_ro_static("dialect_name", &IntegerAttr::dialectName)
      .def("get_int", &IntegerAttr::getInt)
      .def("get_s_int", &IntegerAttr::getSInt)
      .def("get_u_int", &IntegerAttr::getUInt)
      .def("get_aps_int", &IntegerAttr::getAPSInt)
      .def_static("get",
                  nb::overload_cast<Type, const APInt &>(&IntegerAttr::get))
      .def_static("get", nb::overload_cast<mlir::MLIRContext *, const APSInt &>(
                             &IntegerAttr::get))
      .def_static("get", nb::overload_cast<Type, int64_t>(&IntegerAttr::get))
      .def_static("verify", &IntegerAttr::verify)
      .def_static("verify_invariants", &IntegerAttr::verifyInvariants)
      .def("get_type", &IntegerAttr::getType)
      .def("get_value", &IntegerAttr::getValue);

  nb::class_<IntegerSetAttr, Attribute>(m, "IntegerSetAttr")
      .def_ro_static("name", &IntegerSetAttr::name)
      .def_ro_static("dialect_name", &IntegerSetAttr::dialectName)
      .def_static("get", &IntegerSetAttr::get)
      .def("get_value", &IntegerSetAttr::getValue);

  nb::class_<OpaqueAttr, Attribute>(m, "OpaqueAttr")
      .def_ro_static("name", &OpaqueAttr::name)
      .def_ro_static("dialect_name", &OpaqueAttr::dialectName)
      .def_static("get", &OpaqueAttr::get)
      .def_static("verify", &OpaqueAttr::verify)
      .def_static("verify_invariants", &OpaqueAttr::verifyInvariants)
      .def("get_dialect_namespace", &OpaqueAttr::getDialectNamespace)
      .def("get_attr_data", &OpaqueAttr::getAttrData)
      .def("get_type", &OpaqueAttr::getType);

  nb::class_<SparseElementsAttr, Attribute>(m, "SparseElementsAttr")
      .def_ro_static("name", &SparseElementsAttr::name)
      .def_ro_static("dialect_name", &SparseElementsAttr::dialectName)
      //      .def("value_begin", &SparseElementsAttr::value_begin)
      //      .def("try_value_begin_impl",
      //      &SparseElementsAttr::try_value_begin_impl)
      //      .def("get_zero_value", &SparseElementsAttr::getZeroValue)
      //      .def("get_zero_value", &SparseElementsAttr::getZeroValue)
      //      .def("get_zero_value", &SparseElementsAttr::getZeroValue)
      //      .def("get_zero_value", &SparseElementsAttr::getZeroValue)
      //      .def("get_zero_value", &SparseElementsAttr::getZeroValue)
      //      .def("get_zero_value", &SparseElementsAttr::getZeroValue)
      .def_static("get", &SparseElementsAttr::get)
      .def_static("verify", &SparseElementsAttr::verify)
      .def_static("verify_invariants", &SparseElementsAttr::verifyInvariants)
      .def("get_type", &SparseElementsAttr::getType)
      .def("get_indices", &SparseElementsAttr::getIndices);

  nb::class_<StridedLayoutAttr, Attribute>(m, "StridedLayoutAttr")
      .def_ro_static("name", &StridedLayoutAttr::name)
      .def_ro_static("dialect_name", &StridedLayoutAttr::dialectName)
      .def("has_static_layout", &StridedLayoutAttr::hasStaticLayout)
      .def_static("get", &StridedLayoutAttr::get)
      .def_static("verify", &StridedLayoutAttr::verify)
      .def_static("verify_invariants", &StridedLayoutAttr::verifyInvariants)
      .def("get_offset", &StridedLayoutAttr::getOffset)
      .def("get_strides", &StridedLayoutAttr::getStrides)
      .def("get_affine_map", &StridedLayoutAttr::getAffineMap)
      .def("verify_layout", &StridedLayoutAttr::verifyLayout);

  nb::class_<SymbolRefAttr, Attribute>(m, "SymbolRefAttr")
      .def_ro_static("name", &SymbolRefAttr::name)
      .def_ro_static("dialect_name", &SymbolRefAttr::dialectName)
      .def_static("get", nb::overload_cast<MLIRContext *, StringRef,
                                           ArrayRef<FlatSymbolRefAttr>>(
                             &SymbolRefAttr::get))
      .def_static("get", nb::overload_cast<StringAttr>(&SymbolRefAttr::get))
      .def_static("get", nb::overload_cast<MLIRContext *, StringRef>(
                             &SymbolRefAttr::get))
      .def_static("get", nb::overload_cast<Operation *>(&SymbolRefAttr::get))
      .def("get_leaf_reference", &SymbolRefAttr::getLeafReference)
      .def_static("get",
                  nb::overload_cast<StringAttr, ArrayRef<FlatSymbolRefAttr>>(
                      &SymbolRefAttr::get))
      .def("get_root_reference", &SymbolRefAttr::getRootReference)
      .def("get_nested_references", &SymbolRefAttr::getNestedReferences);

  nb::class_<TypeAttr, Attribute>(m, "TypeAttr")
      .def_ro_static("name", &TypeAttr::name)
      .def_ro_static("dialect_name", &TypeAttr::dialectName)
      .def_static("get", &TypeAttr::get)
      .def("get_value", &TypeAttr::getValue);

  nb::class_<UnitAttr, Attribute>(m, "UnitAttr")
      .def_ro_static("name", &UnitAttr::name)
      .def_ro_static("dialect_name", &UnitAttr::dialectName)
      .def_static("get", &UnitAttr::get);

  nb::class_<Region>(m, "Region")
      .def(nb::init<>())
      .def("get_context", &Region::getContext)
      .def("get_loc", &Region::getLoc)
      .def("get_blocks", &Region::getBlocks)
      .def("emplace_block", &Region::emplaceBlock)
      .def("begin", &Region::begin)
      .def("end", &Region::end)
      .def("rbegin", &Region::rbegin)
      .def("rend", &Region::rend)
      .def("empty", &Region::empty)
      .def("push_back", &Region::push_back)
      .def("push_front", &Region::push_front)
      .def("back", &Region::back)
      .def("front", &Region::front)
      .def("has_one_block", &Region::hasOneBlock)
      .def_static("get_sublist_access", &Region::getSublistAccess)
      .def("get_arguments", &Region::getArguments)
      .def("get_argument_types", &Region::getArgumentTypes)
      .def("args_begin", &Region::args_begin)
      .def("args_end", &Region::args_end)
      .def("args_rbegin", &Region::args_rbegin)
      .def("args_rend", &Region::args_rend)
      .def("args_empty", &Region::args_empty)
      .def("add_argument", &Region::addArgument)

      .def("insert_argument",
           nb::overload_cast<Region::args_iterator, Type, Location>(
               &Region::insertArgument))
      .def("insert_argument",
           nb::overload_cast<unsigned, Type, Location>(&Region::insertArgument))
      .def("add_arguments", &Region::addArguments)
      .def("erase_argument", &Region::eraseArgument)
      .def("get_num_arguments", &Region::getNumArguments)
      .def("get_argument", &Region::getArgument)

      //      .def("op_begin", &Region::op_begin)
      //      .def("op_end", &Region::op_end)
      //      .def("get_ops", &Region::getOps)

      //      .def("op_begin", &Region::op_begin)
      //      .def("op_end", &Region::op_end)
      //      .def("get_ops", &Region::getOps)

      .def("get_parent_region", &Region::getParentRegion)
      .def("get_parent_op", &Region::getParentOp)
      //      .def("get_parent_of_type", &Region::getParentOfType)
      .def("get_region_number", &Region::getRegionNumber)
      .def("is_proper_ancestor", &Region::isProperAncestor)
      .def("is_ancestor", &Region::isAncestor)
      .def("clone_into",
           nb::overload_cast<Region *, IRMapping &>(&Region::cloneInto))
      .def("clone_into",
           nb::overload_cast<Region *, Region::iterator, IRMapping &>(
               &Region::cloneInto))
      .def("take_body", &Region::takeBody)
      .def("find_ancestor_block_in_region", &Region::findAncestorBlockInRegion)
      .def("find_ancestor_op_in_region", &Region::findAncestorOpInRegion)
      .def("drop_all_references", &Region::dropAllReferences)
      //      .def("walk", &Region::walk)
      //      .def("view_graph", &Region::viewGraph)
      .def("view_graph", nb::overload_cast<const Twine &>(&Region::viewGraph));

  nb::class_<Value>(m, "Value")
      .def("__bool__", &Value::operator bool)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("get_type", &Value::getType)
      .def("get_context", &Value::getContext)
      .def("set_type", &Value::setType)
      //      .def("get_defining_op", &Value::getDefiningOp)
      //      .def("get_defining_op", &Value::getDefiningOp)
      .def("get_loc", &Value::getLoc)
      .def("set_loc", &Value::setLoc)
      .def("get_parent_region", &Value::getParentRegion)
      .def("get_parent_block", &Value::getParentBlock)
      .def("drop_all_uses", &Value::dropAllUses)
      .def("replace_all_uses_with", &Value::replaceAllUsesWith)
      .def("replace_all_uses_except",
           nb::overload_cast<Value, const SmallPtrSetImpl<Operation *> &>(
               &Value::replaceAllUsesExcept))
      .def("replace_all_uses_except",
           nb::overload_cast<Value, Operation *>(&Value::replaceAllUsesExcept))
      .def("replace_uses_with_if", &Value::replaceUsesWithIf)
      .def("is_used_outside_of_block", &Value::isUsedOutsideOfBlock)
      .def("shuffle_use_list", &Value::shuffleUseList)
      .def("use_begin", &Value::use_begin)
      .def("use_end", &Value::use_end)
      .def("get_uses", &Value::getUses)
      .def("has_one_use", &Value::hasOneUse)
      .def("use_empty", &Value::use_empty)
      .def("user_begin", &Value::user_begin)
      .def("user_end", &Value::user_end)
      .def("get_users", &Value::getUsers)
      //            .def("print", &Value::print)
      //      .def("print", &Value::print)
      //      .def("print", &Value::print)
      .def("dump", &Value::dump)
      //      .def("print_as_operand", &Value::printAsOperand)
      //      .def("print_as_operand", &Value::printAsOperand)
      .def("get_as_opaque_pointer", &Value::getAsOpaquePointer)
      .def("get_from_opaque_pointer", &Value::getFromOpaquePointer)
      .def("get_impl", &Value::getImpl);

  nb::class_<BlockArgument, Value>(m, "BlockArgument")
      .def("get_owner", &BlockArgument::getOwner)
      .def("get_arg_number", &BlockArgument::getArgNumber)
      .def("get_loc", &BlockArgument::getLoc)
      .def("set_loc", &BlockArgument::setLoc);

  nb::class_<Block>(m, "Block")
      .def(nb::init<>())
      .def("clear", &Block::clear)
      .def("get_parent", &Block::getParent)
      .def("get_parent_op", &Block::getParentOp)
      .def("is_entry_block", &Block::isEntryBlock)
      .def("insert_before", &Block::insertBefore)
      .def("insert_after", &Block::insertAfter)
      .def("move_before", nb::overload_cast<Block *>(&Block::moveBefore))
      //      .def("move_before", &Block::moveBefore)
      .def("erase", &Block::erase)
      .def("get_arguments", &Block::getArguments)
      .def("get_argument_types", &Block::getArgumentTypes)
      .def("args_begin", &Block::args_begin)
      .def("args_end", &Block::args_end)
      .def("args_rbegin", &Block::args_rbegin)
      .def("args_rend", &Block::args_rend)
      .def("args_empty", &Block::args_empty)
      .def("add_argument", &Block::addArgument)
      .def("insert_argument",
           nb::overload_cast<Block::args_iterator, Type, Location>(
               &Block::insertArgument))
      .def("insert_argument",
           nb::overload_cast<unsigned, Type, Location>(&Block::insertArgument))
      .def("add_arguments", &Block::addArguments)
      .def("erase_argument", &Block::eraseArgument)
      .def("erase_arguments",
           nb::overload_cast<unsigned, unsigned>(&Block::eraseArguments))
      .def("erase_arguments",
           nb::overload_cast<const BitVector &>(&Block::eraseArguments))
      .def("erase_arguments",
           nb::overload_cast<mlir::function_ref<bool(BlockArgument)>>(
               &Block::eraseArguments))
      .def("get_num_arguments", &Block::getNumArguments)
      .def("get_argument", &Block::getArgument)
      .def("get_operations", &Block::getOperations)
      .def("begin", &Block::begin)
      .def("end", &Block::end)
      .def("rbegin", &Block::rbegin)
      .def("rend", &Block::rend)
      .def("empty", &Block::empty)
      .def("push_back", &Block::push_back)
      .def("push_front", &Block::push_front)
      .def("back", &Block::back)
      .def("front", &Block::front)
      .def("find_ancestor_op_in_block", &Block::findAncestorOpInBlock)
      .def("drop_all_references", &Block::dropAllReferences)
      .def("drop_all_defined_value_uses", &Block::dropAllDefinedValueUses)
      .def("is_op_order_valid", &Block::isOpOrderValid)
      .def("invalidate_op_order", &Block::invalidateOpOrder)
      .def("verify_op_order", &Block::verifyOpOrder)
      .def("recompute_op_order", &Block::recomputeOpOrder)
      //            .def("get_ops", &Block::getOps)
      //            .def("op_begin", &Block::op_begin)
      //      .def("op_end", &Block::op_end)
      .def("without_terminator", &Block::without_terminator)
      .def("get_terminator", &Block::getTerminator)
      .def("might_have_terminator", &Block::mightHaveTerminator)
      .def("pred_begin", &Block::pred_begin)
      .def("pred_end", &Block::pred_end)
      .def("get_predecessors", &Block::getPredecessors)
      .def("has_no_predecessors", &Block::hasNoPredecessors)
      .def("has_no_successors", &Block::hasNoSuccessors)
      .def("get_single_predecessor", &Block::getSinglePredecessor)
      .def("get_unique_predecessor", &Block::getUniquePredecessor)
      .def("get_num_successors", &Block::getNumSuccessors)
      .def("get_successor", &Block::getSuccessor)
      .def("succ_begin", &Block::succ_begin)
      .def("succ_end", &Block::succ_end)
      .def("get_successors", &Block::getSuccessors)
      .def("is_reachable", &Block::isReachable)
      //            .def("walk", &Block::walk)
      //            .def("walk", &Block::walk)
      .def("split_block",
           nb::overload_cast<Block::iterator>(&Block::splitBlock))
      .def("split_block", nb::overload_cast<Operation *>(&Block::splitBlock))
      .def("get_sublist_access", &Block::getSublistAccess)
      //            .def("print", &Block::print)
      //            .def("print", &Block::print)
      .def("dump", &Block::dump);
  //            .def("print_as_operand", &Block::printAsOperand)
  //            .def("print_as_operand", &Block::printAsOperand)

  nb::class_<OpFoldResult>(m, "OpFoldResult");

  nb::class_<Operation>(m, "Operation")
      .def_static(
          "create",
          nb::overload_cast<Location, OperationName, TypeRange, ValueRange,
                            NamedAttrList &&, OpaqueProperties, BlockRange,
                            unsigned>(&Operation::create))
      .def_static(
          "create",
          nb::overload_cast<Location, OperationName, TypeRange, ValueRange,
                            DictionaryAttr, OpaqueProperties, BlockRange,
                            unsigned>(&Operation::create))
      .def_static("create",
                  nb::overload_cast<const OperationState &>(&Operation::create))
      .def_static(
          "create",
          nb::overload_cast<Location, OperationName, TypeRange, ValueRange,
                            NamedAttrList &&, OpaqueProperties, BlockRange,
                            RegionRange>(&Operation::create))
      .def("get_name", &Operation::getName)
      .def("get_registered_info", &Operation::getRegisteredInfo)
      .def("is_registered", &Operation::isRegistered)
      .def("erase", &Operation::erase)
      .def("remove", &Operation::remove)
      .def("clone", nb::overload_cast<IRMapping &, Operation::CloneOptions>(
                        &Operation::clone))
      .def("clone",
           nb::overload_cast<Operation::CloneOptions>(&Operation::clone))
      .def("clone_without_regions",
           nb::overload_cast<IRMapping &>(&Operation::cloneWithoutRegions))
      .def("clone_without_regions",
           nb::overload_cast<>(&Operation::cloneWithoutRegions))
      .def("get_block", &Operation::getBlock)
      .def("get_context", &Operation::getContext)
      .def("get_dialect", &Operation::getDialect)
      .def("get_loc", &Operation::getLoc)
      .def("set_loc", &Operation::setLoc)
      .def("get_parent_region", &Operation::getParentRegion)
      .def("get_parent_op", &Operation::getParentOp)
      //            .def("get_parent_of_type", &Operation::getParentOfType)
      //            .def("get_parent_with_trait",
      //            &Operation::getParentWithTrait)
      .def("is_proper_ancestor", &Operation::isProperAncestor)
      .def("is_ancestor", &Operation::isAncestor)
      .def("replace_uses_of_with", &Operation::replaceUsesOfWith)
      //            .def("replace_all_uses_with",
      //            &Operation::replaceAllUsesWith) .def("replace_uses_with_if",
      //            &Operation::replaceUsesWithIf)
      .def("destroy", &Operation::destroy)
      .def("drop_all_references", &Operation::dropAllReferences)
      .def("drop_all_defined_value_uses", &Operation::dropAllDefinedValueUses)
      .def("move_before",
           nb::overload_cast<Operation *>(&Operation::moveBefore))
      //      .def("move_before", &Operation::moveBefore)
      .def("move_after", nb::overload_cast<Operation *>(&Operation::moveAfter))
      .def("move_after", nb::overload_cast<Operation *>(&Operation::moveAfter))
      .def("is_before_in_block", &Operation::isBeforeInBlock)
      //            .def("print", &Operation::print)
      //      .def("print", &Operation::print)
      .def("dump", &Operation::dump)
      .def("set_operands",
           nb::overload_cast<ValueRange>(&Operation::setOperands))
      .def("set_operands", nb::overload_cast<unsigned, unsigned, ValueRange>(
                               &Operation::setOperands))
      .def("insert_operands", &Operation::insertOperands)
      .def("get_num_operands", &Operation::getNumOperands)
      .def("get_operand", &Operation::getOperand)
      .def("set_operand", &Operation::setOperand)
      .def("erase_operand",
           nb::overload_cast<unsigned>(&Operation::eraseOperand))
      .def("erase_operands",
           nb::overload_cast<unsigned, unsigned>(&Operation::eraseOperands))
      .def("erase_operands",
           nb::overload_cast<const BitVector &>(&Operation::eraseOperands))
      .def("operand_begin", &Operation::operand_begin)
      .def("operand_end", &Operation::operand_end)
      .def("get_operands", &Operation::getOperands)
      .def("get_op_operands", &Operation::getOpOperands)
      .def("get_op_operand", &Operation::getOpOperand)
      .def("operand_type_begin", &Operation::operand_type_begin)
      .def("operand_type_end", &Operation::operand_type_end)
      .def("get_operand_types", &Operation::getOperandTypes)
      .def("get_num_results", &Operation::getNumResults)
      .def("get_result", &Operation::getResult)
      .def("result_begin", &Operation::result_begin)
      .def("result_end", &Operation::result_end)
      .def("get_results", &Operation::getResults)
      .def("get_op_results", &Operation::getOpResults)
      .def("get_op_result", &Operation::getOpResult)
      .def("result_type_begin", &Operation::result_type_begin)
      .def("result_type_end", &Operation::result_type_end)
      .def("get_result_types", &Operation::getResultTypes)
      .def("get_inherent_attr", &Operation::getInherentAttr)
      .def("set_inherent_attr", &Operation::setInherentAttr)
      .def("get_discardable_attr",
           nb::overload_cast<StringRef>(&Operation::getDiscardableAttr))
      .def("get_discardable_attr",
           nb::overload_cast<StringAttr>(&Operation::getDiscardableAttr))
      .def("set_discardable_attr", nb::overload_cast<StringAttr, Attribute>(
                                       &Operation::setDiscardableAttr))
      .def("set_discardable_attr", nb::overload_cast<StringRef, Attribute>(
                                       &Operation::setDiscardableAttr))
      .def("remove_discardable_attr",
           nb::overload_cast<StringAttr>(&Operation::removeDiscardableAttr))
      .def("remove_discardable_attr",
           nb::overload_cast<StringRef>(&Operation::removeDiscardableAttr))
      .def("get_discardable_attrs", &Operation::getDiscardableAttrs)
      .def("get_discardable_attr_dictionary",
           &Operation::getDiscardableAttrDictionary)
      .def("get_raw_dictionary_attrs", &Operation::getRawDictionaryAttrs)
      .def("get_attrs", &Operation::getAttrs)
      .def("get_attr_dictionary", &Operation::getAttrDictionary)
      .def("set_attrs", nb::overload_cast<DictionaryAttr>(&Operation::setAttrs))
      .def("set_attrs",
           nb::overload_cast<ArrayRef<NamedAttribute>>(&Operation::setAttrs))
      .def("set_discardable_attrs",
           nb::overload_cast<DictionaryAttr>(&Operation::setDiscardableAttrs))
      .def("set_discardable_attrs", nb::overload_cast<ArrayRef<NamedAttribute>>(
                                        &Operation::setDiscardableAttrs))
      .def("get_attr", nb::overload_cast<StringAttr>(&Operation::getAttr))
      .def("get_attr", nb::overload_cast<StringRef>(&Operation::getAttr))
      //            .def("get_attr_of_type", &Operation::getAttrOfType)
      //            .def("get_attr_of_type", &Operation::getAttrOfType)
      .def("has_attr", nb::overload_cast<StringAttr>(&Operation::hasAttr))
      .def("has_attr", nb::overload_cast<StringRef>(&Operation::hasAttr))
      //            .def("has_attr_of_type", &Operation::hasAttrOfType)
      .def("set_attr",
           nb::overload_cast<StringAttr, Attribute>(&Operation::setAttr))
      .def("set_attr",
           nb::overload_cast<StringRef, Attribute>(&Operation::setAttr))
      .def("remove_attr", nb::overload_cast<StringAttr>(&Operation::removeAttr))
      .def("remove_attr", nb::overload_cast<StringRef>(&Operation::removeAttr))
      .def("get_dialect_attrs", &Operation::getDialectAttrs)
      .def("dialect_attr_begin", &Operation::dialect_attr_begin)
      .def("dialect_attr_end", &Operation::dialect_attr_end)
      //            .def("set_dialect_attrs", &Operation::setDialectAttrs)
      .def("populate_default_attrs", &Operation::populateDefaultAttrs)
      .def("get_num_regions", &Operation::getNumRegions)
      .def("get_regions", &Operation::getRegions)
      .def("get_region", &Operation::getRegion)
      .def("get_block_operands", &Operation::getBlockOperands)
      .def("successor_begin", &Operation::successor_begin)
      .def("successor_end", &Operation::successor_end)
      .def("get_successors", &Operation::getSuccessors)
      .def("has_successors", &Operation::hasSuccessors)
      .def("get_num_successors", &Operation::getNumSuccessors)
      .def("get_successor", &Operation::getSuccessor)
      .def("set_successor", &Operation::setSuccessor)
      .def("fold",
           nb::overload_cast<ArrayRef<Attribute>,
                             SmallVectorImpl<OpFoldResult> &>(&Operation::fold))
      .def("fold",
           nb::overload_cast<SmallVectorImpl<OpFoldResult> &>(&Operation::fold))
      //            .def("has_promise_or_implements_interface",
      //                 &Operation::hasPromiseOrImplementsInterface)
      //            .def("has_trait", &Operation::hasTrait)
      //            .def("might_have_trait", &Operation::mightHaveTrait)
      //            .def("walk", &Operation::walk)
      //      .def("walk", &Operation::walk)
      .def("drop_all_uses", &Operation::dropAllUses)
      .def("use_begin", &Operation::use_begin)
      .def("use_end", &Operation::use_end)
      .def("get_uses", &Operation::getUses)
      .def("has_one_use", &Operation::hasOneUse)
      .def("use_empty", &Operation::use_empty)
      .def("is_used_outside_of_block", &Operation::isUsedOutsideOfBlock)
      .def("user_begin", &Operation::user_begin)
      .def("user_end", &Operation::user_end)
      .def("get_users", &Operation::getUsers)
      .def("emit_op_error", &Operation::emitOpError)
      .def("emit_error", &Operation::emitError)
      .def("emit_warning", &Operation::emitWarning)
      .def("emit_remark", &Operation::emitRemark)
      .def("get_properties_storage_size", &Operation::getPropertiesStorageSize)
      //            .def("get_properties_storage",
      //            &Operation::getPropertiesStorage)
      //      .def("get_properties_storage", &Operation::getPropertiesStorage)
      .def("get_properties_storage_unsafe",
           &Operation::getPropertiesStorageUnsafe)
      .def("get_properties_as_attribute", &Operation::getPropertiesAsAttribute)
      .def("set_properties_from_attribute",
           &Operation::setPropertiesFromAttribute)
      .def("copy_properties", &Operation::copyProperties)
      .def("hash_properties", &Operation::hashProperties);

  nb::class_<OperationName>(m, "OperationName");
  nb::class_<DialectInterface>(m, "DialectInterface");

  nb::class_<Dialect>(m, "Dialect")
      .def("is_valid_namespace", &Dialect::isValidNamespace)
      .def("get_context", &Dialect::getContext)
      .def("get_namespace", &Dialect::getNamespace)
      .def("get_type_id", &Dialect::getTypeID)
      .def("allows_unknown_operations", &Dialect::allowsUnknownOperations)
      .def("allows_unknown_types", &Dialect::allowsUnknownTypes)
      .def("get_canonicalization_patterns",
           &Dialect::getCanonicalizationPatterns)
      .def("materialize_constant", &Dialect::materializeConstant)
      .def("parse_attribute", &Dialect::parseAttribute)
      .def("print_attribute", &Dialect::printAttribute)
      .def("parse_type", &Dialect::parseType)
      .def("print_type", &Dialect::printType)
      .def("get_parse_operation_hook", &Dialect::getParseOperationHook)
      .def("get_operation_printer", &Dialect::getOperationPrinter)
      .def("verify_region_arg_attribute", &Dialect::verifyRegionArgAttribute)
      .def("verify_region_result_attribute",
           &Dialect::verifyRegionResultAttribute)
      .def("verify_operation_attribute", &Dialect::verifyOperationAttribute)
      .def("get_registered_interface",
           [](Dialect &self, TypeID interfaceID) {
             return self.getRegisteredInterface(interfaceID);
           })
      .def("get_registered_interface_for_op",
           [](Dialect &self, TypeID interfaceID, OperationName opName) {
             return self.getRegisteredInterfaceForOp(interfaceID, opName);
           })
      .def("add_interface",
           [](Dialect &self, std::unique_ptr<DialectInterface> interface) {
             return self.addInterface(std::move(interface));
           })
      .def("handle_use_of_undefined_promised_interface",
           &Dialect::handleUseOfUndefinedPromisedInterface)
      .def("handle_addition_of_undefined_promised_interface",
           &Dialect::handleAdditionOfUndefinedPromisedInterface)
      .def("has_promised_interface",
           [](Dialect &self, TypeID interfaceRequestorID, TypeID interfaceID) {
             return self.hasPromisedInterface(interfaceRequestorID,
                                              interfaceID);
           });

  nb::class_<DialectRegistry_>(m, "DialectRegistry")
      .def(nb::init<>())
      .def(
          "insert",
          [](DialectRegistry_ &self, TypeID typeID, StringRef name) {
            self.insert(typeID, name,
                        static_cast<DialectAllocatorFunction>(
                            ([&name](MLIRContext *ctx) {
                              return ctx->getOrLoadDialect(name);
                            })));
          },
          "type_id"_a, "name"_a);

  nb::enum_<MLIRContext::Threading>(m, "Threading")
      .value("DISABLED", MLIRContext::Threading::DISABLED)
      .value("ENABLED", MLIRContext::Threading::ENABLED);

  nb::class_<MLIRContext>(m, "MLIRContext")
      .def(nb::init<MLIRContext::Threading>(),
           "multithreading"_a = MLIRContext::Threading::ENABLED)
      .def(nb::init<const DialectRegistry_ &, MLIRContext::Threading>(),
           "registry"_a, "multithreading"_a = MLIRContext::Threading::ENABLED)
      .def("get_loaded_dialects", &MLIRContext::getLoadedDialects)
      .def("get_dialect_registry", &MLIRContext::getDialectRegistry)
      .def("append_dialect_registry", &MLIRContext::appendDialectRegistry)
      .def("get_available_dialects", &MLIRContext::getAvailableDialects)
      .def("get_loaded_dialect",
           [](MLIRContext &self, StringRef name) {
             return self.getLoadedDialect(name);
           })
      //      .def("load_dialect", &MLIRContext::loadDialect)
      //      .def("get_or_load_dynamic_dialect",
      //      &MLIRContext::getOrLoadDynamicDialect)
      .def("load_all_available_dialects",
           &MLIRContext::loadAllAvailableDialects)
      .def("get_or_load_dialect",
           [](MLIRContext &self, StringRef name) {
             return self.getOrLoadDialect(name);
           })
      .def("allows_unregistered_dialects",
           &MLIRContext::allowsUnregisteredDialects)
      .def("allow_unregistered_dialects",
           &MLIRContext::allowUnregisteredDialects)
      .def("is_multithreading_enabled", &MLIRContext::isMultithreadingEnabled)
      .def("disable_multithreading", &MLIRContext::disableMultithreading)
      .def("enable_multithreading", &MLIRContext::enableMultithreading)
      //      .def("set_thread_pool", &MLIRContext::setThreadPool)
      .def("get_num_threads", &MLIRContext::getNumThreads)
      //      .def("get_thread_pool", &MLIRContext::getThreadPool)
      .def("should_print_op_on_diagnostic",
           &MLIRContext::shouldPrintOpOnDiagnostic)
      .def("print_op_on_diagnostic", &MLIRContext::printOpOnDiagnostic)
      .def("should_print_stack_trace_on_diagnostic",
           &MLIRContext::shouldPrintStackTraceOnDiagnostic)
      .def("print_stack_trace_on_diagnostic",
           &MLIRContext::printStackTraceOnDiagnostic)
      .def("get_registered_operations", &MLIRContext::getRegisteredOperations)
      .def("get_registered_operations_by_dialect",
           &MLIRContext::getRegisteredOperationsByDialect)
      .def("is_operation_registered", &MLIRContext::isOperationRegistered)
      //        .def("get_impl", &MLIRContext::getImpl)
      .def("get_diag_engine", &MLIRContext::getDiagEngine)
      .def("get_affine_uniquer", &MLIRContext::getAffineUniquer)
      .def("get_type_uniquer", &MLIRContext::getTypeUniquer)
      .def("get_attribute_uniquer", &MLIRContext::getAttributeUniquer)
      .def("enter_multi_threaded_execution",
           &MLIRContext::enterMultiThreadedExecution)
      .def("exit_multi_threaded_execution",
           &MLIRContext::exitMultiThreadedExecution)
      //        .def("get_or_load_dialect", &MLIRContext::getOrLoadDialect)
      .def("get_registry_hash", &MLIRContext::getRegistryHash)
      .def("register_action_handler", &MLIRContext::registerActionHandler)
      .def("has_action_handler", &MLIRContext::hasActionHandler)
      .def("execute_action",
           [](MLIRContext &self, mlir::function_ref<void()> actionFn,
              const tracing::Action &action) {
             self.executeAction(actionFn, action);
           });
}
