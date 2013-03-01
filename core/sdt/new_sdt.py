from pylab import *
import numpy as np
import math

def run_threads(data):
    l = len(data)
    verts  = np.zeros(l)
    coeffs = np.zeros(l)
    signs  = np.zeros(l)

    threads = [go(i,data,verts,coeffs,signs) for i in range(0,len(data))]
    try:
        while True:
            for thread in threads:
                thread.next()
    except Exception as e:
        return data
    return data
 
def square(x):
    return x*x

def sign(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1

def go(i,data,verts,coeffs,signs):
    l = len(data)
    # First we write the value that this thread starts with
    # to the shared memory and synchronize with the other threads.
    out = data[i]
    sgn = sign(out)
    out = abs(out)
    coeffs[i] = out
    signs[i] = sgn * i
  
    yield
    # Then perform the initial reduction step - each pair of threads
    # computes the most local distance transform
    otherindex = i ^ 1
    otherdata  = coeffs[otherindex]
    othersign  = signs[otherindex]
    
    if 0 > othersign * sgn:
        if sgn == -1:
            out = 0
            coeffs[i] = 0
            verts[i] = i
            signs[i] = 0
        else:
            out = 1
            coeffs[i] = 1
            verts[i] = otherindex
    elif otherdata < out:
        coeffs[i] = otherdata + 1
        verts[i] = otherindex
    else:
        verts[i] = i
    yield
    # Now start gradually building bigger and bigger regions of local distance transforms
    # j is the index of the bottom of the previous regions in verts and coeffs
    j = i & ~1 
    # mask is region size - 1
    mask = 3
    while l > 0: 

        l = l >> 1 # Hello O(log(n))!

        # Find the index of the lower bound of the lower of the two regions being merged
        base = j & ~3
        # The bottom of the merged region lives half-way down the array
        dest = base >> 1

        offset = base ^ j
        half = offset >> 1
        offset = offset | half

        lowvertex = verts[base + 1]
        lowcoeff  = coeffs[base + 1]

        highvertex = verts[base + 2]
        highcoeff  = coeffs[base + 2]

        lowsgn = signs[base + 1]
        highsgn = signs[base + 2]

        if 0 > lowsgn * highsgn and l > 1:
            lowvertex = abs(min(lowsgn,highsgn))
            lowcoeff  = 0   

        low = square(i - lowvertex) + lowcoeff
        high = square(i - highvertex) + highcoeff
        extreme = square(i - verts[base + offset]) + coeffs[base + offset]

        out = min(out,high,low,extreme)
        
        # The first thread in the new region writes out the lower bound
        # and the last thread writes the upper bound.
        par = i & mask
        
        if par == 0 or par == mask:
            vertex = None
            coefficient = None

            if high < extreme or low < extreme:
                if high < low:
                    vertex = highvertex
                    coefficient = highcoeff
                else:
                    vertex = lowvertex
                    coefficient = lowcoeff
            else:
                vertex = verts[base + offset]
                coefficient = coeffs[base + offset]

            s = signs[base + 3 * half]
            yield
            signs[dest+ half] = s
            coeffs[dest + half] = coefficient;
            verts[dest + half] = vertex;
            yield

        else:
            yield
            yield        

        j = dest
        mask = (mask << 1) + 1;    

    data[i] = sgn * out
    yield

size = 128

data = np.array([(lambda x: min(x*x - 0.001,(x-0.25)*(x-0.25) - 0.001))(x / (1.0 * size) - 0.5) for x in range(0,size)])
binarized = np.array([size * size * sign(x) for x in data])
sdt = run_threads(binarized)

plot(data * size)
#plot(binarized)
#plot(run_threads(binarized) * 10)
plot([sign(x) * math.sqrt(abs(x)) for x in sdt])
plot([0 for x in data])
show()
