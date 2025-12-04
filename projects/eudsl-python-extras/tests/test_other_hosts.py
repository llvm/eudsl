import pytest


def jax_not_installed():
    try:
        from jaxlib import mlir

        # don't skip
        return False

    except ImportError:
        # skip
        return True


@pytest.mark.skipif(jax_not_installed(), reason="jax not installed")
def test_jax_trampolines_smoke():
    # noinspection PyUnresolvedReferences
    from jaxlib.mlir import ir

    # noinspection PyUnresolvedReferences
    from jaxlib.mlir.extras import context

    # noinspection PyUnresolvedReferences
    from jaxlib.mlir.extras.runtime import passes
