import pycuda.driver as drv
import pycuda.gpuarray as gpu
import numpy as np

import kernel

module = kernel.load_module("sdt.cu")

# Pull out a number of functions that we need
edt_pass = module.get_function("edt_pass")
sqrt = kernel.map_over_first(module.get_function("signed_sqrt"), 16)
binarize = kernel.map_over_first(module.get_function("binarize"), 16)


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
    binarize(zeros, drv.In(sample))

    edt_pass(zeros,np.int32(x),np.int32(y), np.int32(0), block = (x,1,1), grid = (1,y))
    edt_pass(zeros,np.int32(x),np.int32(y), np.int32(1), block = (1,y,1), grid = (x,1))

    out = np.zeros(sample.shape).astype(np.float32)
    sqrt(zeros, drv.Out(out))

    return out

