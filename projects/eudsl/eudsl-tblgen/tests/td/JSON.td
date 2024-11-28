// adapted from llvm/test/TableGen/JSON.td

class Base {}
class Intermediate : Base {}
class Derived : Intermediate {}

def D : Intermediate {}

def ExampleDagOp;

def FieldKeywordTest {
    int a;
    field int b;
}

class Variables {
    int i;
    string s;
    bit b;
    bits<8> bs;
    code c;
    list<int> li;
    Base base;
    dag d;
}
def VarNull : Variables {}
def VarPrim : Variables {
    int i = 3;
    int enormous_pos = 9123456789123456789;
    int enormous_neg = -9123456789123456789;
    string s = "hello, world";
    bit b = 0;
    bits<8> bs = { 0,0,0,1,0,1,1,1 };
    code c = [{ void  }];
    list<int> li = [ 1, 2, 3, 4 ];
}
def VarObj : Variables {
    Base base = D;
    dag d = (ExampleDagOp 22, "hello":$foo);
    int undef_int;
    field int ref_int = undef_int;
    bits<2> undef_bits;
    bits<4> ref_bits;
    let ref_bits{3...2} = 0b10;
    let ref_bits{1...0} = undef_bits{1...0};
    field int complex_ref_int = !add(undef_int, 2);
}

def Named { int AnonTestField = 1; }
def { int AnonTestField = 2; }
