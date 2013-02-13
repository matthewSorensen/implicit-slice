import math
import numpy as np

def at(array,i):
    x, y = i
    return array[x][y]

def sgn(x):
    return math.copysign(1,x)

ihat = np.array([1,0])
jhat = np.array([0,1])
ones = np.array([1,1])
zeros = np.array([0,0])

def forward(x):
    return (x + 1) % 4

def backward(x):
    return (x + 3) % 4

def lerp(a,b,t):
    return a + t * (b -a)

def comps(vect):
    """ Project a vector onto i and j """
    x,y = vect
    return ihat * x, jhat * y
