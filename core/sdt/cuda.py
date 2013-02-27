import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from os.path import abspath, dirname, join

import pycuda.gpuarray as gpu

# Load and compile the files containing all kernels needed for
# sdt.
modulepath = join(dirname(abspath(__file__)), "sdt_kernel.cu")
modulefile = open(modulepath, "r")
module = SourceModule(modulefile.read())
modulefile.close()

# Pull out a number of functions that we need
implicit_first_pass = module.get_function("implicit_first_pass")
voxel_first_pass = module.get_function("voxel_first_pass")
second_pass = module.get_function("second_pass")
sign_and_sqrt = module.get_function("sign_and_sqrt")


def blocks(shape):
    size = 32
    x, y = shape
    return int((x + size -1) / size),int((y + size -1) / size), size


def sdt(sample, implicit = True):
    blsx, blsy, threads = blocks(sample.shape)
    sgpu = gpu.to_gpu(sample)

    width, height = sample.shape
    iwidth, iheight = np.int32(width), np.int32(height)

    oneblock = (threads,1,1)
    twoblock = (threads,threads,1)

    if implicit:
        implicit_first_pass(sgpu, iwidth, iheight,block = oneblock, grid=(blsx,1))
    else:
        voxel_first_pass(sgpu, iwidth, iheight, block = oneblock, grid = (blsx,1))

    # Now the GPU version of samples is a set of 2D signed distances.
    # Then, allocate GPU temporary storage for bounds and vertexes
    bounds = gpu.empty((width+1,height),np.float32)
    verts  = gpu.empty((width,height),np.int32)
    unsigned = gpu.empty((width,height),np.float32)
    
    second_pass(sgpu,bounds,verts,unsigned, iwidth, iheight, block = oneblock, grid = (blsy, 1))
    sign_and_sqrt(sgpu,unsigned, drv.Out(sample), iwidth, iheight, block = twoblock, grid = (blsx, blsy))
    # ensure that sgpu, bounds, verts, unsigned are free. No idea how to do that...
    return sample
