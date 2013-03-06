from core.sdt import sdt
from  core.loader import cpu_eval
import matplotlib.pyplot as plt

def square(x):
    return x*x

s = cpu_eval((1,1,1),1/512.0,1/512.0,lambda x,y,z: square(x-0.5) + square(y-0.5) - 0.0025)

v, z = s.next()

dat = sdt(v, implicit = True)

plt.clf()
plt.contour(dat)
plt.colorbar()
plt.show()
