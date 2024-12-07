from collections.abc import Iterable, Iterator
from typing import TypeAlias, TypeVar, overload

from . import dialects as dialects, ir as ir


class APFloat:
    pass

class APInt:
    pass

class APSInt:
    pass

class ArrayRef[T]:
    @overload
    def __init__(self, arg: SmallVector[b], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[f], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[i], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[c], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[d], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[l], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[s], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[i], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[x], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[t], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[j], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[y], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir4TypeE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir8LocationE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir9AttributeE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir10AffineExprE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir9AffineMapE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir6IRUnitE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir23RegisteredOperationNameE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4llvm5APIntE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4llvm7APFloatE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir5ValueE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir10StringAttrE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir13OperationNameE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[PN4mlir6RegionE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[PN4mlir11SymbolTableE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[PN4mlir9OperationE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir12OpFoldResultE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir14NamedAttributeE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir17FlatSymbolRefAttrE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir13BlockArgumentE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4llvm8ArrayRefIPN4mlir5BlockEEE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4llvm9StringRefE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir18DiagnosticArgumentE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir11OpAsmParser8ArgumentE], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[N4mlir11OpAsmParser17UnresolvedOperandE], /) -> None: ...

class ArrayRef[N4llvm5APIntE]:
    def __init__(self, arg: SmallVector[N4llvm5APIntE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4llvm7APFloatE]:
    def __init__(self, arg: SmallVector[N4llvm7APFloatE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4llvm8ArrayRefIPN4mlir5BlockEEE]:
    def __init__(self, arg: SmallVector[N4llvm8ArrayRefIPN4mlir5BlockEEE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator["llvm::ArrayRef<mlir::Block*>"]: ...

    def __getitem__(self, arg: int, /) -> "llvm::ArrayRef<mlir::Block*>": ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: "llvm::ArrayRef<mlir::Block*>", /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: "llvm::ArrayRef<mlir::Block*>", /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[N4llvm9StringRefE]:
    def __init__(self, arg: SmallVector[N4llvm9StringRefE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir10AffineExprE]:
    def __init__(self, arg: SmallVector[N4mlir10AffineExprE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir10StringAttrE]:
    def __init__(self, arg: SmallVector[N4mlir10StringAttrE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.StringAttr]: ...

    def __getitem__(self, arg: int, /) -> ir.StringAttr: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.StringAttr, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.StringAttr, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[N4mlir11OpAsmParser17UnresolvedOperandE]:
    def __init__(self, arg: SmallVector[N4mlir11OpAsmParser17UnresolvedOperandE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OpAsmParser.UnresolvedOperand]: ...

    def __getitem__(self, arg: int, /) -> ir.OpAsmParser.UnresolvedOperand: ...

class ArrayRef[N4mlir11OpAsmParser8ArgumentE]:
    def __init__(self, arg: SmallVector[N4mlir11OpAsmParser8ArgumentE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OpAsmParser.Argument]: ...

    def __getitem__(self, arg: int, /) -> ir.OpAsmParser.Argument: ...

class ArrayRef[N4mlir12OpFoldResultE]:
    def __init__(self, arg: SmallVector[N4mlir12OpFoldResultE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OpFoldResult]: ...

    def __getitem__(self, arg: int, /) -> ir.OpFoldResult: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.OpFoldResult, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.OpFoldResult, /) -> int:
        """Return number of occurrences of `arg`."""

class ArrayRef[N4mlir13BlockArgumentE]:
    def __init__(self, arg: SmallVector[N4mlir13BlockArgumentE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir13OperationNameE]:
    def __init__(self, arg: SmallVector[N4mlir13OperationNameE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir14NamedAttributeE]:
    def __init__(self, arg: SmallVector[N4mlir14NamedAttributeE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir17FlatSymbolRefAttrE]:
    def __init__(self, arg: SmallVector[N4mlir17FlatSymbolRefAttrE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir18DiagnosticArgumentE]:
    def __init__(self, arg: SmallVector[N4mlir18DiagnosticArgumentE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.DiagnosticArgument]: ...

    def __getitem__(self, arg: int, /) -> ir.DiagnosticArgument: ...

class ArrayRef[N4mlir23RegisteredOperationNameE]:
    def __init__(self, arg: SmallVector[N4mlir23RegisteredOperationNameE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir4TypeE]:
    def __init__(self, arg: SmallVector[N4mlir4TypeE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir5ValueE]:
    def __init__(self, arg: SmallVector[N4mlir5ValueE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir6IRUnitE]:
    def __init__(self, arg: SmallVector[N4mlir6IRUnitE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir8LocationE]:
    def __init__(self, arg: SmallVector[N4mlir8LocationE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir9AffineMapE]:
    def __init__(self, arg: SmallVector[N4mlir9AffineMapE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[N4mlir9AttributeE]:
    def __init__(self, arg: SmallVector[N4mlir9AttributeE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[PN4mlir11SymbolTableE]:
    def __init__(self, arg: SmallVector[PN4mlir11SymbolTableE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[PN4mlir6RegionE]:
    def __init__(self, arg: SmallVector[PN4mlir6RegionE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[PN4mlir9OperationE]:
    def __init__(self, arg: SmallVector[PN4mlir9OperationE], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[b]:
    def __init__(self, arg: SmallVector[b], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[c]:
    def __init__(self, arg: SmallVector[c], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[d]:
    def __init__(self, arg: SmallVector[d], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[f]:
    def __init__(self, arg: SmallVector[f], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[i]:
    @overload
    def __init__(self, arg: SmallVector[i], /) -> None: ...

    @overload
    def __init__(self, arg: SmallVector[i], /) -> None: ...

    @overload
    def __len__(self) -> int: ...

    @overload
    def __len__(self) -> int: ...

    @overload
    def __bool__(self) -> bool: ...

    @overload
    def __bool__(self) -> bool: ...

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __eq__(self, arg: object, /) -> bool: ...

    @overload
    def __eq__(self, arg: object, /) -> bool: ...

    @overload
    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    @overload
    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    @overload
    def count(self, arg: int, /) -> int: ...

class ArrayRef[j]:
    def __init__(self, arg: SmallVector[j], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[l]:
    def __init__(self, arg: SmallVector[l], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[s]:
    def __init__(self, arg: SmallVector[s], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[t]:
    def __init__(self, arg: SmallVector[t], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[x]:
    def __init__(self, arg: SmallVector[x], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class ArrayRef[y]:
    def __init__(self, arg: SmallVector[y], /) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...

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

class MutableArrayRef[T]:
    pass

class ParseResult:
    pass

class SmallBitVector:
    pass

class SmallPtrSetImpl[Operation]:
    pass

class SmallVector[T]:
    @staticmethod
    def nparray(arg: object, /) -> None: ...

class SmallVector[N4llvm5APIntE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4llvm5APIntE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[APInt], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[APInt]: ...

    @overload
    def __getitem__(self, arg: int, /) -> APInt: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4llvm5APIntE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: APInt, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: APInt, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> APInt:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4llvm5APIntE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: APInt, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4llvm5APIntE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: APInt, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: APInt, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: APInt, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4llvm7APFloatE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4llvm7APFloatE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[APFloat], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[APFloat]: ...

    @overload
    def __getitem__(self, arg: int, /) -> APFloat: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4llvm7APFloatE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: APFloat, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: APFloat, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> APFloat:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4llvm7APFloatE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: APFloat, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4llvm7APFloatE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: APFloat, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: APFloat, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: APFloat, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4llvm8ArrayRefIPN4mlir5BlockEEE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4llvm8ArrayRefIPN4mlir5BlockEEE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable["llvm::ArrayRef<mlir::Block*>"], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator["llvm::ArrayRef<mlir::Block*>"]: ...

    @overload
    def __getitem__(self, arg: int, /) -> "llvm::ArrayRef<mlir::Block*>": ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4llvm8ArrayRefIPN4mlir5BlockEEE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: "llvm::ArrayRef<mlir::Block*>", /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: "llvm::ArrayRef<mlir::Block*>", /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> "llvm::ArrayRef<mlir::Block*>":
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4llvm8ArrayRefIPN4mlir5BlockEEE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: "llvm::ArrayRef<mlir::Block*>", /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4llvm8ArrayRefIPN4mlir5BlockEEE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: "llvm::ArrayRef<mlir::Block*>", /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: "llvm::ArrayRef<mlir::Block*>", /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: "llvm::ArrayRef<mlir::Block*>", /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4llvm9StringRefE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4llvm9StringRefE]) -> None:
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
    def __getitem__(self, arg: slice, /) -> SmallVector[N4llvm9StringRefE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: str, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: str, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> str:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4llvm9StringRefE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: str, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4llvm9StringRefE], /) -> None: ...

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

class SmallVector[N4mlir10AffineExprE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir10AffineExprE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.AffineExpr], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.AffineExpr]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.AffineExpr: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir10AffineExprE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.AffineExpr, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.AffineExpr, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.AffineExpr:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir10AffineExprE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.AffineExpr, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir10AffineExprE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.AffineExpr, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.AffineExpr, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.AffineExpr, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir10StringAttrE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir10StringAttrE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.StringAttr], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.StringAttr]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.StringAttr: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir10StringAttrE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.StringAttr, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.StringAttr, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.StringAttr:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir10StringAttrE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.StringAttr, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir10StringAttrE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.StringAttr, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.StringAttr, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.StringAttr, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir11OpAsmParser17UnresolvedOperandE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir11OpAsmParser17UnresolvedOperandE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.OpAsmParser.UnresolvedOperand], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OpAsmParser.UnresolvedOperand]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.OpAsmParser.UnresolvedOperand: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir11OpAsmParser17UnresolvedOperandE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.OpAsmParser.UnresolvedOperand, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.OpAsmParser.UnresolvedOperand, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.OpAsmParser.UnresolvedOperand:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir11OpAsmParser17UnresolvedOperandE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.OpAsmParser.UnresolvedOperand, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir11OpAsmParser17UnresolvedOperandE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

class SmallVector[N4mlir11OpAsmParser8ArgumentE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir11OpAsmParser8ArgumentE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.OpAsmParser.Argument], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OpAsmParser.Argument]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.OpAsmParser.Argument: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir11OpAsmParser8ArgumentE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.OpAsmParser.Argument, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.OpAsmParser.Argument, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.OpAsmParser.Argument:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir11OpAsmParser8ArgumentE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.OpAsmParser.Argument, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir11OpAsmParser8ArgumentE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

class SmallVector[N4mlir12OpFoldResultE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir12OpFoldResultE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.OpFoldResult], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OpFoldResult]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.OpFoldResult: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir12OpFoldResultE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.OpFoldResult, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.OpFoldResult, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.OpFoldResult:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir12OpFoldResultE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.OpFoldResult, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir12OpFoldResultE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.OpFoldResult, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.OpFoldResult, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.OpFoldResult, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir13BlockArgumentE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir13BlockArgumentE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.BlockArgument], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.BlockArgument]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.BlockArgument: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir13BlockArgumentE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.BlockArgument, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.BlockArgument, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.BlockArgument:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir13BlockArgumentE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.BlockArgument, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir13BlockArgumentE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.BlockArgument, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.BlockArgument, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.BlockArgument, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir13OperationNameE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir13OperationNameE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.OperationName], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.OperationName]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.OperationName: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir13OperationNameE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.OperationName, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.OperationName, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.OperationName:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir13OperationNameE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.OperationName, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir13OperationNameE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.OperationName, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.OperationName, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.OperationName, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir14NamedAttributeE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir14NamedAttributeE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.NamedAttribute], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.NamedAttribute]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.NamedAttribute: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir14NamedAttributeE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.NamedAttribute, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.NamedAttribute, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.NamedAttribute:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir14NamedAttributeE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.NamedAttribute, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir14NamedAttributeE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.NamedAttribute, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.NamedAttribute, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.NamedAttribute, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir17FlatSymbolRefAttrE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir17FlatSymbolRefAttrE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.FlatSymbolRefAttr], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.FlatSymbolRefAttr]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.FlatSymbolRefAttr: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir17FlatSymbolRefAttrE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.FlatSymbolRefAttr, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.FlatSymbolRefAttr, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.FlatSymbolRefAttr:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir17FlatSymbolRefAttrE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.FlatSymbolRefAttr, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir17FlatSymbolRefAttrE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.FlatSymbolRefAttr, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.FlatSymbolRefAttr, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.FlatSymbolRefAttr, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir18DiagnosticArgumentE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir18DiagnosticArgumentE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.DiagnosticArgument], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.DiagnosticArgument]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.DiagnosticArgument: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir18DiagnosticArgumentE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.DiagnosticArgument, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.DiagnosticArgument, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.DiagnosticArgument:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir18DiagnosticArgumentE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.DiagnosticArgument, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir18DiagnosticArgumentE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

class SmallVector[N4mlir23RegisteredOperationNameE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir23RegisteredOperationNameE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.RegisteredOperationName], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.RegisteredOperationName]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.RegisteredOperationName: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir23RegisteredOperationNameE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.RegisteredOperationName, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.RegisteredOperationName, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.RegisteredOperationName:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir23RegisteredOperationNameE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.RegisteredOperationName, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir23RegisteredOperationNameE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.RegisteredOperationName, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.RegisteredOperationName, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.RegisteredOperationName, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir4TypeE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir4TypeE]) -> None:
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
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir4TypeE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.Type, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.Type, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.Type:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir4TypeE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.Type, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir4TypeE], /) -> None: ...

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

class SmallVector[N4mlir5ValueE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir5ValueE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.Value], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Value]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.Value: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir5ValueE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.Value, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.Value, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.Value:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir5ValueE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.Value, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir5ValueE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Value, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Value, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.Value, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir6IRUnitE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir6IRUnitE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.IRUnit], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.IRUnit]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.IRUnit: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir6IRUnitE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.IRUnit, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.IRUnit, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.IRUnit:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir6IRUnitE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.IRUnit, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir6IRUnitE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.IRUnit, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.IRUnit, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.IRUnit, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir8LocationE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir8LocationE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.Location], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Location]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.Location: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir8LocationE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.Location, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.Location, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.Location:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir8LocationE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.Location, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir8LocationE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Location, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Location, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.Location, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir9AffineMapE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir9AffineMapE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.AffineMap], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.AffineMap]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.AffineMap: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir9AffineMapE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.AffineMap, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.AffineMap, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.AffineMap:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir9AffineMapE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.AffineMap, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir9AffineMapE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.AffineMap, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.AffineMap, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.AffineMap, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[N4mlir9AttributeE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[N4mlir9AttributeE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.Attribute], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Attribute]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.Attribute: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[N4mlir9AttributeE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.Attribute, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.Attribute, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.Attribute:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[N4mlir9AttributeE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.Attribute, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[N4mlir9AttributeE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Attribute, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Attribute, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.Attribute, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[PN4mlir11SymbolTableE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[PN4mlir11SymbolTableE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.SymbolTable], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.SymbolTable]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.SymbolTable: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[PN4mlir11SymbolTableE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.SymbolTable, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.SymbolTable, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.SymbolTable:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[PN4mlir11SymbolTableE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.SymbolTable, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[PN4mlir11SymbolTableE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.SymbolTable, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.SymbolTable, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.SymbolTable, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[PN4mlir6RegionE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[PN4mlir6RegionE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.Region], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Region]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.Region: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[PN4mlir6RegionE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.Region, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.Region, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.Region:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[PN4mlir6RegionE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.Region, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[PN4mlir6RegionE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Region, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Region, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.Region, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[PN4mlir9OperationE]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[PN4mlir9OperationE]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[ir.Operation], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Operation]: ...

    @overload
    def __getitem__(self, arg: int, /) -> ir.Operation: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[PN4mlir9OperationE]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: ir.Operation, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: ir.Operation, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> ir.Operation:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[PN4mlir9OperationE], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: ir.Operation, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[PN4mlir9OperationE], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: ir.Operation, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: ir.Operation, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: ir.Operation, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[b]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[b]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[bool], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[bool]: ...

    @overload
    def __getitem__(self, arg: int, /) -> bool: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[b]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: bool, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: bool, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> bool:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[b], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: bool, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[b], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: bool, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: bool, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: bool, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[c]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[c]) -> None:
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
    def __getitem__(self, arg: slice, /) -> SmallVector[c]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: str, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: str, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> str:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[c], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: str, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[c], /) -> None: ...

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

class SmallVector[d]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[d]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[float], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[float]: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[d]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: float, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: float, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> float:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[d], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[d], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: float, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: float, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: float, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[f]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[f]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[float], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[float]: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[f]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: float, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: float, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> float:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[f], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[f], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: float, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: float, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: float, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[i]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[i]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[i]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[i], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[i], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[j]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[j]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[j]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[j], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[j], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[l]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[l]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[l]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[l], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[l], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[s]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[s]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[s]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[s], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[s], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[t]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[t]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[t]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[t], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[t], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[x]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[x]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[x]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[x], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[x], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class SmallVector[y]:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SmallVector[y]) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> SmallVector[y]: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: SmallVector[y], /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: SmallVector[y], /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class SourceMgr:
    pass

class StorageUniquer:
    pass

T = TypeVar("T")

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

class hash_code:
    pass

class initializer_list[Block]:
    pass

class initializer_list[Type]:
    pass

class initializer_list[Value]:
    pass

class iplist[Block]:
    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[ir.Block]: ...

    def __getitem__(self, arg: int, /) -> ir.Block: ...

class iplist[Operation]:
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
