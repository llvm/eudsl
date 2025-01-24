#  Part of the  Project, under the Apache License v2.0 with  Exceptions.
#  See https:#llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH -exception
#  Copyright (c) 2024.
from textwrap import dedent

from llvm import types_ as T
from llvm.context import context
from llvm.function import function
from llvm.instructions import add, ret


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
    with context(src=src, buffer_name="test_smoke") as ctx:
        print(ctx)


def test_builder():
    with context(mod_name="test_builder") as ctx:

        @function
        def bob(a: T.int32, b: T.int32) -> T.int32: ...

        @function(emit=True)
        def sum(a: T.int32, b: T.int32) -> T.int32:
            result = add(a, b)
            ret(result)

        print(ctx)


if __name__ == "__main__":
    test_smoke()
    test_builder()
