__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#pkgutil-style-namespace-packages

from .. import ir
from .ast.py_type import PyTypeObject

# Hack to allow us to inherit from nanobind's metaclass type
# https://github.com/wjakob/nanobind/pull/836
nb_meta_cls = type(ir.Value)
_Py_TPFLAGS_BASETYPE = 1 << 10
PyTypeObject.from_object(nb_meta_cls).tp_flags |= _Py_TPFLAGS_BASETYPE
