from pathlib import Path


def bin_dir() -> str:
    return str(Path(__file__).parent / "bin")


def cmake_dir() -> str:
    return str(Path(__file__).parent / "lib" / "cmake")


def include_dir() -> str:
    return str(Path(__file__).parent / "include")


def lib_dir() -> str:
    return str(Path(__file__).parent / "lib")


def root_dir() -> str:
    return str(Path(__file__).parent)


__all__ = ("bin_dir", "cmake_dir", "include_dir", "lib_dir", "root_dir")
