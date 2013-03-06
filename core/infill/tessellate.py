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

def extract_gpumap(module,name):
    f = module.get_function(name)

    def wrap(array,*rest):
        x,y = array.shape
        dx = (x + 15) / 16
        dy = (y + 15) / 16

        return f(array,*rest, block = (16,16,1), grid = (dx,dy))
    return wrap

def dim(v):
    return gpu.vec.make_int2(*v.shape)

gpu_tile = extract_gpumap(module,"tile")
gpu_fuse = extract_gpumap(module,"fuse")

def tessellate_infill(model,offset,pattern,displace):

    infill = gpu.empty(model.shape,np.float32)

    gpu_tile(infill, drv.In(pattern), dim(model), dim(pattern), gpu.vec.make_int2(*displace))

    gpu_fuse(infill,drv.In(model), dim(infill), np.float32(offset))

    return infill.get()

