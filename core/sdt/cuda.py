import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from os.path import abspath, dirname, join

import pycuda.gpuarray as gpu

# Load and compile the files containing all kernels needed for
# sdt.
modulepath = join(dirname(abspath(__file__)), "kernels.cu")
modulefile = open(modulepath, "r")
module = SourceModule(modulefile.read())
modulefile.close()

# Pull out a number of functions that we need
binarize = module.get_function("binarize")
edt_pass = module.get_function("edt_pass")
sqrt = module.get_function("signed_sqrt")

def blocks(shape):
    size = 32
    x, y = shape
    return int((x + size -1) / size),int((y + size -1) / size), size

def bits(i):
    b = 0
    while i > 0:
        b += i & 1
        i = i >> 1
    return b

def sdt(sample, implicit = True):
    x,y = sample.shape

    if x > 512 or y > 512 or 1 != bits(x) or 1 != bits(y):
        raise ValueError("all array dimensions must be a power of two <= 512")

    zeros = gpu.empty(sample.shape,np.int32)

    binarize(drv.In(sample),zeros, np.int32(x), np.int32(y), block = (16,16,1), grid = (x >> 4, y >> 4))
    edt_pass(zeros,np.int32(x),np.int32(y), np.int32(0), block = (x,1,1), grid = (1,y))
    edt_pass(zeros,np.int32(x),np.int32(y), np.int32(1), block = (1,y,1), grid = (x,1))

    out = np.zeros(sample.shape).astype(np.float32)
    sqrt(zeros,drv.Out(out), np.int32(x), np.int32(y), block = (16,16,1), grid = (x >> 4, y >> 4))

    return out

