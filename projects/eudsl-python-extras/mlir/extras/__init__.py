__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#pkgutil-style-namespace-packages

from .util import make_nanobind_metaclass_inheritable

make_nanobind_metaclass_inheritable()
