import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from pylab import *

first = open("fast_firstpass.cu","r")
first = SourceModule(first.read())
first = first.get_function("fast_firstpass")

edt = open("fast_parsdt.cu","r")
edt = SourceModule(edt.read())
edt = edt.get_function("fast_parsdt")

data = [(lambda x: x*x - 0.0625)((x-64.0)/128.0) for x in range(0,128)]
data = np.array(data).astype(np.float32)

output = np.zeros((128,1)).astype(int32)

first(drv.In(data),drv.Out(output),np.int32(128),np.int32(1), block = (16,16,1), grid = (8,1))
edt(drv.InOut(output),np.int32(128),np.int32(1), block = (128,1,1), grid = (1,1))

plot(output)
show()
