import sympy.utilities.codegen as gen

from sympy import symbols
from sympy.utilities.codegen import codegen
from sympy.abc import x, y, z
from StringIO import StringIO

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
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

def compile_expr(expr):
    codegen = CUDAGen("implicit-kernel")
    routines = []
    routines.append(gen.Routine("f", expr, [x,y,z]))
    module = SourceModule(codegen.write(routines))
    kernel = module.get_function("kernel")

    def f(dim,z):
        x, y = dim
        dest = np.zeros(dim).astype(np.float32)

        gx = int((x + 31) / 32)
        gy = int((y + 31) / 32)

        kernel(drv.Out(dest),np.int32(x),np.int32(y),np.float32(z), block = (32,32,1), grid = (int(gx), int(gy), 1))

        return dest

    return f
