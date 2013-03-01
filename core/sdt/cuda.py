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
extract_zeros = module.get_function("extract_zeros")
edt_pass = module.get_function("edt_pass")

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

    zeros = np.zeros(sample.shape,np.int32)

    extract_zeros(drv.In(sample),drv.Out(zeros), np.int32(x), np.int32(y), block = (16,16,1), grid = (x >> 4, y >> 4))

    edt_pass(drv.InOut(zeros),np.int32(x),np.int32(y), block = (x,1,1), grid = (1,y))

    return zeros

# blsx, blsy, threads = blocks(sample.shape)
 #   sgpu = gpu.to_gpu(sample)

  #  width, height = sample.shape
#    iwidth, iheight = np.int32(width), np.int32(height)

 #   oneblock = (threads,1,1)
  #  twoblock = (threads,threads,1)

   # if implicit:
    #    implicit_first_pass(sgpu, iwidth, iheight,block = oneblock, grid=(blsx,1))
   # else:
 #       voxel_first_pass(sgpu, iwidth, iheight, block = oneblock, grid = (blsx,1))
#
    # Now the GPU version of samples is a set of 2D signed distances.
    # Then, allocate GPU temporary storage for bounds and vertexes
 #   bounds = gpu.empty((width+1,height),np.float32)
  #  verts  = gpu.empty((width,height),np.int32)
   # unsigned = gpu.empty((width,height),np.float32)
    
   # second_pass(sgpu,bounds,verts,unsigned, iwidth, iheight, block = oneblock, grid = (blsy, 1))
    #sign_and_sqrt(sgpu,unsigned, drv.Out(sample), iwidth, iheight, block = twoblock, grid = (blsx, blsy))
    # ensure that sgpu, bounds, verts, unsigned are free. No idea how to do that...
   # return sample
