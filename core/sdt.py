
import math
from util import sgn

def square(x):
    return x * x

def spmin(sample,dt):
    """ Returns a number such that sgn(spmin(x,y)) = sgn(x) and |spmin(x,y)| = min(|x|,|y|) """
    mag = min(abs(sample),abs(dt))
    return math.copysign(mag, sample)

def reconstruct(coeff,size, old):
    minx, maxx = coeff

    if minx is None:
        for i in range(0,maxx):
            old[i] = spmin(old[i], square(i - maxx))
    elif maxx is None:
        for i in range(minx,size):          
            old[i] = spmin(old[i], square(i - minx))
    else:
        for i in range(minx,maxx+1):
            old[i] = spmin(old[i], min([square(i - minx), square(i - maxx)]))

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

    for i, s in enumerate(samples):
        samples[i] = math.copysign(ls * ls, s)

    for c in coeffs:
        reconstruct(c, ls, samples)

    return samples



