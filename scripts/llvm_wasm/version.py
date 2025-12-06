from __future__ import annotations

import os

__all__ = ["dynamic_metadata"]


def __dir__() -> list[str]:
    return __all__


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
    _project: dict[str, object] = None,
) -> str:
    if field != "version":
        msg = "Only the 'version' field is supported"
        raise ValueError(msg)

    if settings:
        msg = "No inline configuration is supported"
        raise ValueError(msg)

    return os.getenv("WHEEL_VERSION", "0.0.0")
