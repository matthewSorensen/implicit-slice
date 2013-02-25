import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from os.path import abspath, dirname, join

# Load and compile the files containing all kernels needed for
# sdt.
modulepath = join(dirname(abspath(__file__)), "sdt_kernel.cu")
modulefile = open(modulepath, "r")
module = SourceModule(modulefile.read())
modulefile.close()

# Pull out a number of functions that we need
implicit_first_pass = module.get_function("implicit_first_pass")
voxel_first_pass = module.get_function("voxel_first_pass")
