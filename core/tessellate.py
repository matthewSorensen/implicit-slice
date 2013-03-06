import pycuda.driver as drv
import numpy as np
import pycuda.gpuarray as gpu
import kernel

module = kernel.load_module("tessellate.cu")

tile = kernel.map_over_first(module.get_function("tile"), 16)
fuse = kernel.map_over_first(module.get_function("fuse"), 16)

def tessellate_infill(model,offset,pattern,displace):

    infill = gpu.empty(model.shape,np.float32)

    tile(infill, drv.In(pattern), kernel.dim(pattern), kernel.dim(displace))
    fuse(infill, drv.In(model), np.float32(offset))

    return infill.get()

