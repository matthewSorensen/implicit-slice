#!/usr/bin/env python
from core.loader import gpu_eval
import matplotlib.pyplot as plt
dat,_ = gpu_eval((128,128),0.1,1,"sin(100 * x) * cos(10*y)").next()
plt.imshow(dat)
plt.show()

