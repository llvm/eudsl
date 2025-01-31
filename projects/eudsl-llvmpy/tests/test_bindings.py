#  Part of the  Project, under the Apache License v2.0 with  Exceptions.
#  See https:#llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH -exception
#  Copyright (c) 2024.
from textwrap import dedent

from llvm import types_ as T
from llvm.context import context
from llvm.function import function
from llvm.instructions import add, ret
import llvm.amdgcn
from mlir import ir


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

        @function(emit=True)
        def sum(a: T.int32, b: T.int32, c: T.float) -> T.int32:
            e = llvm.amdgcn.cvt_pk_i16(a, b)
            f = llvm.amdgcn.frexp_mant(c)
            result = add(a, b)
            ret(result)

        mod_str = str(ctx)

    correct = dedent(
        """\
    ; ModuleID = 'test_builder'
    source_filename = "test_builder"

    define i32 @sum(i32 %0, i32 %1, float %2) {
    entry:
      %3 = call <2 x i16> @llvm.amdgcn.cvt.pk.i16(i32 %0, i32 %1)
      %4 = call float @llvm.amdgcn.frexp.mant.f32(float %2)
      %5 = add i32 %0, %1
      ret i32 %5
    }

    ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
    declare <2 x i16> @llvm.amdgcn.cvt.pk.i16(i32, i32) #0

    ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
    declare float @llvm.amdgcn.frexp.mant.f32(float) #0

    attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
    """
    )

    assert correct == mod_str


if __name__ == "__main__":
    test_smoke()
    test_builder()
