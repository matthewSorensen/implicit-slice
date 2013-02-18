
import math
from util import sgn

def square(x):
    return x * x

def reconstruct(coeff,size):
    minx, maxx = coeff

    if minx is None:
        return [square(x - maxx) for x in range(0,size)]
    if maxx is None:
        return [square(x - minx) for x in range(0,size)]
    return [min(square(x - minx),square(x - maxx)) for x in range(0,size)]

def first_pass(samples):
    # we want to turn samples into a piece-wise set of quadratics
    # coeffs contains (min, max) for each quadratic
    psgn = sgn(samples[0])
    coeffs = [(None,None)]

    for i, s in enumerate(samples):
        ssgn = sgn(s)
        if ssgn != psgn:
            # close the current coeffient and start a new one
            oldmin, oldmax = coeffs[-1]
            coeffs[-1] = (oldmin, i)
            coeffs.append((i,None))
        psgn = ssgn
        
    # so now we reconstruct the sample as a bunch of distances
    ls = len(samples)
        
    output = [ls * ls] * ls

    for c in coeffs:
        v = reconstruct(c,ls)
        for i, s in enumerate(v):
            output[i] = min([s, output[i]])

    for i, s in enumerate(output):
        output[i] = math.copysign(s, samples[i])

    return output



