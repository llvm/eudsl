__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#pkgutil-style-namespace-packages

from .ast.py_type import PyTypeObject


def make_nanobind_metaclass_inheritable():
    from .. import ir

    # Hack to allow us to inherit from nanobind's metaclass type
    # https://github.com/wjakob/nanobind/pull/836
    nb_meta_cls = type(ir.Value)
    _Py_TPFLAGS_BASETYPE = 1 << 10
    PyTypeObject.from_object(nb_meta_cls).tp_flags |= _Py_TPFLAGS_BASETYPE


make_nanobind_metaclass_inheritable()
