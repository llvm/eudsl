#  Part of the  Project, under the Apache License v2.0 with  Exceptions.
#  See https:#llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH -exception
#  Copyright (c) 2024.
from textwrap import dedent

from llvm import (
    ContextRef,
    MemoryBufferRef,
    ModuleRef,
    context_create,
    parse_ir_in_context,
    create_memory_buffer_with_memory_range,
    ModuleRef,
    dump_module,
    print_module_to_string,
    create_builder,
    int32_type,
    function_type,
    add_function,
    append_basic_block,
    position_builder_at_end,
    get_param,
    build_add,
    build_ret,
    dispose_builder,
    module_create_with_name_in_context,
)


def test_smoke():
    src = dedent(
        """
    declare i32 @foo()                                             
    declare i32 @bar()                                             
    define i32 @entry(i32 %argc) {                                 
    entry:                                                         
      %and = and i32 %argc, 1                                      
      %tobool = icmp eq i32 %and, 0                                
      br i1 %tobool, label %if.end, label %if.then                 
    if.then:                                                       
      %call = tail call i32 @foo()                                 
      br label %return                                             
    if.end:                                                        
      %call1 = tail call i32 @bar()                                
      br label %return                                             
    return:                                                        
      %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.end ] 
      ret i32 %retval.0                                            
    }                                                              
    """
    )
    ctx = context_create()
    buf = create_memory_buffer_with_memory_range(src, len(src), "<src>", True)
    mod = ModuleRef()
    parse_ir_in_context(ctx, buf, mod)
    print(print_module_to_string(mod))


def test_builder():
    ctx = context_create()
    mod = module_create_with_name_in_context("demo", ctx)

    # Add a "sum" function":
    #  - Create the function type and function instance.
    param_types = [int32_type(), int32_type()]
    sum_function_type = function_type(int32_type(), param_types, 2, 0)
    sum_function = add_function(mod, "sum", sum_function_type)

    #  - Add a basic block to the function.
    entry_bb = append_basic_block(sum_function, "entry")

    #  - Add an IR builder and point it at the end of the basic block.
    builder = create_builder()
    position_builder_at_end(builder, entry_bb)

    #  - Get the two function arguments and use them co construct an "add"
    #    instruction.
    sum_arg_0 = get_param(sum_function, 0)
    sum_arg_1 = get_param(sum_function, 1)
    result = build_add(builder, sum_arg_0, sum_arg_1, "result")

    #  - Build the return instruction.
    build_ret(builder, result)

    #  - Free the builder.
    dispose_builder(builder)

    print(print_module_to_string(mod))


if __name__ == "__main__":
    # test_smoke()
    test_builder()
