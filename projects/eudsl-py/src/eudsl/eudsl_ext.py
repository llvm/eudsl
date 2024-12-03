from collections.abc import Iterable, Iterator
from typing import TypeAlias, overload

from . import dialects as dialects, ir as ir


class APFloat:
    pass

class APInt:
    pass

class APSInt:
    pass

class ArrayRefOfType:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: VectorOfType, /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Type]: ...

    def __getitem__(self, arg: int, /) -> ir.Type: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Type, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Type, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[APFloat]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[APFloat]: ...

    def __getitem__(self, arg: int, /) -> APFloat: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: APFloat, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: APFloat, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[APInt]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[APInt]: ...

    def __getitem__(self, arg: int, /) -> APInt: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: APInt, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: APInt, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[AffineExpr]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.AffineExpr]: ...

    def __getitem__(self, arg: int, /) -> ir.AffineExpr: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.AffineExpr, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.AffineExpr, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[AffineMap]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.AffineMap]: ...

    def __getitem__(self, arg: int, /) -> ir.AffineMap: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.AffineMap, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.AffineMap, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[Attribute]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Attribute]: ...

    def __getitem__(self, arg: int, /) -> ir.Attribute: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Attribute, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Attribute, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[BlockArgument]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.BlockArgument]: ...

    def __getitem__(self, arg: int, /) -> ir.BlockArgument: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.BlockArgument, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.BlockArgument, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[Block]:
    pass

class ArrayRef[FlatSymbolRefAttr]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.FlatSymbolRefAttr]: ...

    def __getitem__(self, arg: int, /) -> ir.FlatSymbolRefAttr: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.FlatSymbolRefAttr, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.FlatSymbolRefAttr, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[IRUnit]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.IRUnit]: ...

    def __getitem__(self, arg: int, /) -> ir.IRUnit: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.IRUnit, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.IRUnit, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[Location]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Location]: ...

    def __getitem__(self, arg: int, /) -> ir.Location: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Location, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Location, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[NamedAttribute]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.NamedAttribute]: ...

    def __getitem__(self, arg: int, /) -> ir.NamedAttribute: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.NamedAttribute, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.NamedAttribute, /) -> int:
        """Return number of occurrences of `arg`."""

ArrayRef[OpAsmParser.Argument]: TypeAlias = ArrayRef[OpAsmParser.Argument]

ArrayRef[OpAsmParser.UnresolvedOperand]: TypeAlias = ArrayRef[OpAsmParser.UnresolvedOperand]

class ArrayRef[OperationName]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OperationName]: ...

    def __getitem__(self, arg: int, /) -> ir.OperationName: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.OperationName, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.OperationName, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[Operation]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Operation]: ...

    def __getitem__(self, arg: int, /) -> ir.Operation: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Operation, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Operation, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[Region]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Region]: ...

    def __getitem__(self, arg: int, /) -> ir.Region: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Region, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Region, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[RegisteredOperationName]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.RegisteredOperationName]: ...

    def __getitem__(self, arg: int, /) -> ir.RegisteredOperationName: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.RegisteredOperationName, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.RegisteredOperationName, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[SymbolTable]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.SymbolTable]: ...

    def __getitem__(self, arg: int, /) -> ir.SymbolTable: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.SymbolTable, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.SymbolTable, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[Value]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Value]: ...

    def __getitem__(self, arg: int, /) -> ir.Value: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Value, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Value, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[bool]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[bool]: ...

    def __getitem__(self, arg: int, /) -> bool: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: bool, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: bool, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[char]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[str]: ...

    def __getitem__(self, arg: int, /) -> str: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: str, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: str, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[double]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[float]: ...

    def __getitem__(self, arg: int, /) -> float: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: float, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: float, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[float]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[float]: ...

    def __getitem__(self, arg: int, /) -> float: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: float, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: float, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[int16_t]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[int32_t]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[int64_t]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[long]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[signed_char]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[str]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[str]: ...

    def __getitem__(self, arg: int, /) -> str: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: str, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: str, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[unsigned]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[unsigned_long]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

class AsmParser:
    pass

class AsmResourcePrinter:
    pass

class AttrTypeSubElementReplacements[Attribute]:
    pass

class AttrTypeSubElementReplacements[Type]:
    pass

class BitVector:
    pass

class DataLayoutSpecInterface:
    pass

class DialectBytecodeReader:
    pass

class DialectBytecodeWriter:
    pass

class DialectResourceBlobHandle[BuiltinDialect]:
    pass

class FailureOr[AffineMap]:
    pass

class FailureOr[AsmDialectResourceHandle]:
    pass

class FailureOr[AsmResourceBlob]:
    pass

class FailureOr[ElementsAttrIndexer]:
    pass

class FailureOr[OperationName]:
    pass

class FailureOr[StringAttr]:
    pass

class FailureOr[bool]:
    pass

class IRObjectWithUseList[BlockOperand]:
    pass

class IRObjectWithUseList[OpOperand]:
    pass

class IntegerValueRange:
    pass

class InterfaceMap:
    pass

class LogicalResult:
    pass

class MutableArrayRef[BlockArgument]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.BlockArgument]: ...

    def __getitem__(self, arg: int, /) -> ir.BlockArgument: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.BlockArgument, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.BlockArgument, /) -> int:
        """Return number of occurrences of `arg`."""

class MutableArrayRef[BlockOperand]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.BlockOperand]: ...

    def __getitem__(self, arg: int, /) -> ir.BlockOperand: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.BlockOperand, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.BlockOperand, /) -> int:
        """Return number of occurrences of `arg`."""

class MutableArrayRef[DiagnosticArgument]:
    pass

class MutableArrayRef[Dialect]:
    pass

class MutableArrayRef[OpOperand]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OpOperand]: ...

    def __getitem__(self, arg: int, /) -> ir.OpOperand: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.OpOperand, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.OpOperand, /) -> int:
        """Return number of occurrences of `arg`."""

class MutableArrayRef[Region]:
    pass

class MutableArrayRef[char]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[str]: ...

    def __getitem__(self, arg: int, /) -> str: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: str, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: str, /) -> int:
        """Return number of occurrences of `arg`."""

class ParseResult:
    pass

class SmallBitVector:
    pass

class SmallPtrSetImpl[Operation]:
    pass

class SmallVectorImpl[Attribute]:
    pass

class SmallVectorImpl[DiagnosticArgument]:
    pass

class SmallVectorImpl[NamedAttribute]:
    pass

SmallVectorImpl[OpAsmParser.Argument]: TypeAlias = SmallVectorImpl[OpAsmParser.Argument]

SmallVectorImpl[OpAsmParser.UnresolvedOperand]: TypeAlias = SmallVectorImpl[OpAsmParser.UnresolvedOperand]

class SmallVectorImpl[OpFoldResult]:
    pass

class SmallVectorImpl[Operation]:
    pass

class SmallVectorImpl[Type]:
    pass

class SmallVectorImpl[Value]:
    pass

class SmallVectorImpl[int]:
    pass

class SourceMgr:
    pass

class StorageUniquer:
    pass

class TargetSystemSpecInterface:
    pass

class ThreadPoolInterface:
    pass

class TypeID:
    pass

class ValueTypeRange[OperandRange]:
    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Type]: ...

    def __getitem__(self, arg: int, /) -> ir.Type: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Type, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Type, /) -> int:
        """Return number of occurrences of `arg`."""

class ValueTypeRange[ResultRange]:
    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Type]: ...

    def __getitem__(self, arg: int, /) -> ir.Type: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Type, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Type, /) -> int:
        """Return number of occurrences of `arg`."""

class ValueTypeRange[ValueRange]:
    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Type]: ...

    def __getitem__(self, arg: int, /) -> ir.Type: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Type, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Type, /) -> int:
        """Return number of occurrences of `arg`."""

class ValueUseIterator[BlockOperand]:
    pass

class ValueUseIterator[OpOperand]:
    pass

class VectorOfDialect:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: VectorOfDialect) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.Dialect], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Dialect]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.Dialect: ...

    @overload
    def __getitem__(self, arg: slice, /) -> VectorOfDialect: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.Dialect, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.Dialect, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.Dialect:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: VectorOfDialect, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.Dialect, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: VectorOfDialect, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Dialect, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Dialect, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.Dialect, /) -> None:
        """Remove first occurrence of `arg`."""

class VectorOfStringRef:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: VectorOfStringRef) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[str], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[str]: ...

    @overload
    def __getitem__(self, arg: int, /) -> str: ...

    @overload
    def __getitem__(self, arg: slice, /) -> VectorOfStringRef: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: str, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: str, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> str:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: VectorOfStringRef, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: str, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: VectorOfStringRef, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: str, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: str, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: str, /) -> None:
        """Remove first occurrence of `arg`."""

class VectorOfType:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: VectorOfType) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.Type], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Type]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.Type: ...

    @overload
    def __getitem__(self, arg: slice, /) -> VectorOfType: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.Type, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.Type, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.Type:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: VectorOfType, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.Type, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: VectorOfType, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Type, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Type, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.Type, /) -> None:
        """Remove first occurrence of `arg`."""

class hash_code:
    pass

class initializer_list[Block]:
    pass

class initializer_list[Type]:
    pass

class initializer_list[Value]:
    pass

class iplist[Block]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Block]: ...

    def __getitem__(self, arg: int, /) -> ir.Block: ...

class iplist[Operation]:
    def __init__(self) -> None:
        """Default constructor"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Operation]: ...

    def __getitem__(self, arg: int, /) -> ir.Operation: ...

class iterator_range[BlockArgument]:
    pass

iterator_range[Operation.dialect_attr_iterator]: TypeAlias = iterator_range[Operation.dialect_attr_iterator]

class iterator_range[PredecessorIterator]:
    pass

iterator_range[Region.OpIterator]: TypeAlias = iterator_range[Region.OpIterator]

iterator_range[ResultRange.UseIterator]: TypeAlias = iterator_range[ResultRange.UseIterator]

class raw_ostream:
    pass

class reverse_iterator[BlockArgument]:
    pass
