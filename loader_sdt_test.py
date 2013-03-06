from core.sdt import sdt
from core.tessellate import tessellate_infill

from  core.loader import cpu_eval
import matplotlib.pyplot as plt

def square(x):
    return x*x

s = cpu_eval((1,1,1),1/512.0,1/512.0,lambda x,y,z: square(x-0.5) + square(y-0.5) - 0.25)

v, z = s.next()

dat = sdt(v, implicit = True)

infill,_ = cpu_eval((1,1,1),1/128.0,1/128.0,lambda x,y,z: square(x-0.5) + square(y-0.5) - (0.51 * 0.50)).next()

infill = sdt(tessellate_infill(dat,0.1,infill,(0,0)), implicit = True)

plt.clf()
plt.contour(infill,levels = [0])
plt.show()
