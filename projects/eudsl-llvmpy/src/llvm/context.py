import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from . import (
    ContextRef,
    ModuleRef,
    BuilderRef,
    print_module_to_string,
    context_create,
    module_create_with_name_in_context,
    create_memory_buffer_with_memory_range,
    parse_ir_in_context,
    create_builder_in_context,
    dispose_builder,
)


@dataclass
class Context:
    context: ContextRef
    module: ModuleRef
    builder: BuilderRef

    def __str__(self):
        return print_module_to_string(self.module)


__tls = threading.local()


def current_context() -> Context:
    return __tls.current_context


def set_current_context(ctx: Context):
    __tls.current_context = ctx


def reset_current_context():
    ctx = current_context()
    set_current_context(None)
    dispose_builder(ctx.builder)


@contextmanager
def context(
    mod_name: Optional[str] = None, src: Optional[str] = None, buffer_name="<src>"
):
    ctx = context_create()
    if mod_name is None:
        mod_name = buffer_name
    mod = module_create_with_name_in_context(mod_name, ctx)
    if src is not None:
        buf = create_memory_buffer_with_memory_range(src, len(src), buffer_name, True)
        parse_ir_in_context(ctx, buf, mod)
    builder = create_builder_in_context(ctx)
    _ctx = Context(ctx, mod, builder)
    set_current_context(_ctx)
    yield _ctx
    reset_current_context()
