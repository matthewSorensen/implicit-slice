
import sympy.utilities.codegen as gen

from sympy import symbols
from sympy.utilities.codegen import codegen
from sympy.abc import x, y, z
from StringIO import StringIO

class CUDAGen(gen.CCodeGen):
    def get_prototype(self, routine):
        """ We must modify all prototypes to include __device__ """
        return "__device__ " + super(CUDAGen,self).get_prototype(routine)

    def _preprocessor_statements(self, prefix):
        return ["#include <math.h>\n","#include \"cuda_runtime.h\"\n"]

    def write(self,routines, header = False, empty = True):
        contents = StringIO()
        self.dump_c(routines, contents, ".cu", header, empty)
        return contents.getvalue()
        
# then we must write out a function that actually does all of the work
"""
__device__ kernel(float* out, int length, int width, float dx, float dy, float sx, float sy, z){
    int x, int y;
    float fx, float fy;
    float sample = f(fx,fy,z);
    out[y * length + x] = sample;
}

"""

def cudagen(expr):
    code_gen = CUDAGen("implicit-kernel")
    routines = []
    routines.append(gen.Routine("f", expr, [x,y,z]))
    return code_gen.write(routines)

def test():
    code = cudagen(x + y * x / z)
    print code
