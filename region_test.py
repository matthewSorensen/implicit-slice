import core.region as region
import numpy as np
import matplotlib.pyplot as plt
import math 

def at(array,i):
    x, y = i
    return array[x][y]

def sgn(x):
    return math.copysign(1,x)

def flip(v):
    return np.array([v[1],v[0]])

def pred(sdf,r):
    center  = r.center()
    dcenter = at(sdf,center)
    scenter = sgn(dcenter)
    dcenter = abs(dcenter)

    for corner in r.corners():
        dc = at(sdf,corner)
        if scenter != sgn(dc):
            return True    
        c = math.sqrt(2) * np.linalg.norm(corner - center)
        if c <= dcenter + abs(dc):
            return True
    return False

def transfer(sdf,r):
    corners = r.corners()
    first = sgn(at(sdf,corners.next()))
    for c in corners:
        if sgn(at(sdf,c)) != first:
            return [r]
    return None

def merge(x,y):
    if x is None:
        return y
    if y is None:
        return x
    return x + y

sdf = np.loadtxt("test.dat")
r = region.SpatialRegion(region.zeros, 127 * region.ones)

plt.contour(sdf, levels = [0])
for p in r.quadtree_recursion_scheme(lambda x: pred(sdf,x),lambda x: transfer(sdf,x),merge):
    x,y = p.span
    plt.gca().add_patch(plt.Rectangle(flip(p.ll),x,y,color='r'))

plt.show()

