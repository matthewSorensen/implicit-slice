import math
import numpy as np

def square(x):
    return x * x

def copysign(x,y):
    if x == 0 or y == 0:
        return 0
    else:
        return math.copysign(x,y)

def sgn(x):
    if x == 0:
        return 0
    else:
        return math.copysign(1,x)

def horizontal(samples, replace):
    psgn = sgn(samples[0])
    last = None

    for i, s in enumerate(samples[:]):
        ssgn = sgn(s)
        if ssgn != psgn:
            if psgn == 0:
                i = i - 1
            if last is None:
                for x in range(0,i):
                    samples[x] = copysign(square(i - x), samples[x])
            else:
                for x in range(last,i+1):
                    samples[x] = copysign(square(min([x - last,i - x])),samples[x])
            last = i
        elif ssgn == 0: # by previous, psgn also is 0
            last = i + 1

        psgn = ssgn

    if last is None:
        value = copysign(replace + 1, samples[0])
        for i in range(0,len(samples)):
            samples[i] = value
    else:
        # Otherwise, fill in the last parabola segment
        for i in range(last,len(samples)):
            samples[i] = copysign(square(i - last), samples[i])
            
    return samples

inf = float("inf")
ninf = -1 * inf

def vertical(samples):
    """ This function is a direct implementation of the distance transform
    described in 'Distance Transforms of Sampled Functions' """

    k = 0
    verts = [0 for x in samples]
    bounds = [ninf,inf] + [0 for x in samples]

    for q, s in enumerate(samples[1:]):
        q += 1
        
        ss = s + square(q)

        inter = ss - samples[verts[k]] - square(verts[k])
        inter = 0.5 * inter / (q + verts[k])
        
        while inter <= bounds[k]:
            verts[k] = 0
            k -= 1
        
            inter = ss - samples[verts[k]] - square(verts[k])
            inter = 0.5 * inter / (q + verts[k])

        k += 1
        verts[k]  = q
        bounds[k] = inter
        bounds[k+1] = inf
     
    k = 0
    for q in range(0,len(samples)):
        while bounds[k+1] < q:
            k += 1

        dsquared = square(q - verts[k]) + samples[verts[k]]
        samples[q] = copysign(math.sqrt(dsquared), samples[q])

    return samples

def sdt(sample):
    width, height = sample.shape
    for y in range(0,height):
        horizontal(sample[...,y], square(max(width,height) * 2))
    for x in range(0, width):
        vertical(sample[x,...])

    return sample
