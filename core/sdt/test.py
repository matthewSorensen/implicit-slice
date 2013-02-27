import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from pylab import *

dat = [1,2,3,1,2,3,1,0,-1,-1,-1,1,1,1]

test = np.array(dat).astype(np.float32)
file = open("fast_parsdt.cu","r")
mod = SourceModule(file.read())
sdt = mod.get_function("fast_parsdt")


data = [128 * 128 for x in range(0,128)]
data[20] = 0
data[64] = 0
data[82] = 240
data[100] = 0
data = np.array(data).astype(np.int32)

sdt(drv.InOut(data),np.int32(128),np.int32(0), block = (128,1,1), grid = (1,1))
plot(data)
show()
