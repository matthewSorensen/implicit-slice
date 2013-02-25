from core.sdt.pure import sdt
from  core.loader import cpu_eval
import matplotlib.pyplot as plt

def square(x):
    return x*x

s = cpu_eval((1,1,1),0.01,0.01,lambda x,y,z: square(x-0.5) + square(y-0.5) - 0.025)

v, z = s.next()

dat = sdt(v, implicit = True)

plt.clf()
plt.imshow(dat)
plt.colorbar()
plt.show()
