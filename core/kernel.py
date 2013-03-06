
""" This module simplifies the interface for loading CUDA kernels as executable
python functions. (C) 2013, MDS """

from os.path import abspath, dirname, join
import pycuda.gpuarray as gpu
import pycuda.autoinit
from pycuda.compiler import SourceModule

basedir = dirname(abspath(__file__))

def load_module(name):
    modulefile = open(join(basedir, "kernels", name), "r")
    module = SourceModule(modulefile.read())
    modulefile.close()
    return module

def dim(array):
    """ Returns the size of a 2D array as a CUDA int2. Takes either an array or a 2-tuple """
    try:
        array = array.shape
    except AttributeError:
        None # We we thus assume array is a valid shape
    return gpu.vec.make_int2(*array)

def blockgrid(block,size):
    """ Takes a 1 or 2D block size and the 2D size of the number of threads to run, and 
    returns the block-size extended to 3D and the grid size """

    x,y = (None,None)
    try:
        x,y = block
    except TypeError:
        x = y = block

    sx, sy = size
    return (x,y,1), ((sx + x -1) / x,(sy + y - 1) / y)

def map_over_first(function, block):
    """ Takes a kernel and sets the block and grid size such that enough threads are launched
    to associate a thread with each element of the first argument. The kernel is invoked with
    the first argument unchanged, the second set to the dimension of the first argument, the third
    kernel argument as the second python argument etc. """

    def wrapped(array,*args):
        thisblock, grid = blockgrid(block, array.shape)
        return function(array, dim(array), *args, block = thisblock, grid = grid)

    return wrapped

# Is there something we can do to make setting up types simple?
