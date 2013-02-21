import sympy.utilities.codegen as gen

from sympy import symbols
from sympy.utilities.codegen import codegen
from sympy.abc import x, y, z
from StringIO import StringIO

import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule

class CUDAGen(gen.CCodeGen):
    def get_prototype(self, routine):
        """ We must modify all prototypes to include __device__ """
        return "__device__ " + super(CUDAGen,self).get_prototype(routine)

    def _preprocessor_statements(self, prefix):
        return []
    
    def write(self,routines, header = False, empty = True):
        contents = StringIO()
        self.dump_c(routines, contents, ".cu", header, empty)
        return contents.getvalue() + self.boilerplate()
        
    def boilerplate(self):
        return """
__global__ void kernel(float* out, int length, int width, float z){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(length <= x || width <= y)
        return;

    const float fx = x / (float) length;
    const float fy = y / (float) width;

    out[y * length + x] = f(fx,fy,z);
}

"""

def cudagen(expr):
    code_gen = CUDAGen("implicit-kernel")
    routines = []
    routines.append(gen.Routine("f", expr, [x,y,z]))
    return code_gen.write(routines)

def test():
    code = cudagen(x + y )
    return code

mod = SourceModule(test())

f = mod.get_function("kernel")
dest = numpy.zeros((10,10)).astype(numpy.float32)

f(drv.Out(dest),numpy.int32(10),numpy.int32(10),numpy.float32(1), block=(10,10,1), grid=(1,1))
print dest
