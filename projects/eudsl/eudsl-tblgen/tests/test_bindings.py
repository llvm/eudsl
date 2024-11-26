import pytest


def test_RecTy():
    from eudsl_tblgen import RecTyKind, RecTy

    r = RecTyKind(RecTyKind.BitRecTyKind)
    print(r)
