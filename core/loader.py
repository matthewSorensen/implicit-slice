import numpy as np
from math import ceil


def gpu_eval():
    return None

def cpu_eval(maxes, res, dz, func):
    z = 0
    shape = ceil(maxes[0] / res), ceil(maxes[1] / res)
    sx, sy = shape
    sx = 1 / sx
    sy = 1 / sy
    while z <= maxes[2]:
        array = np.fromfunction(lambda x,y: func(x * sx, y * sy, z), shape)
        yield array, z
        z += dz

def from_file(res,dz,f):
    x, y, z = res
    for i in range(z):
        raw = f.read( 4 * x * y) # read x*y 32-bit chunks
        array = np.fromstring(raw, dtype = np.float32, count = -1, sep = '')
        yield array.reshape((x,y)), i * dz

