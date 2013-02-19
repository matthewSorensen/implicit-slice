import math

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

def first_pass(samples):
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
        ls = len(samples)
        value = copysign(square(ls), samples[0])
        for i in range(0,ls):
            samples[i] = value
    else:
        # Otherwise, fill in the last parabola segment
        for i in range(last,len(samples)):
            samples[i] = copysign(square(i - last), samples[i])
            
    return samples
