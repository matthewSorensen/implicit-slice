import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

dat = [1,2,3,1,2,3,1,0,-1,-1,-1,1,1,1]

test = np.array(dat).astype(np.float32)
file = open("first.cu","r")
mod = SourceModule(file.read())
horizontal = mod.get_function("horizontal")
csgns = mod.get_function("copysign_and_sqrt")

dest = np.zeros((len(dat),1)).astype(np.float32)

horizontal(drv.In(test),drv.Out(dest),np.int32(len(dat)),np.int32(0),np.int32(10), block=(len(dat),1,1), grid=(1,1))
print "computed 1-d sdt: ", dest

