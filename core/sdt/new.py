



def run_threads(data):
    verts  = [0 for x in data]
    coeffs = [0 for x in data]

    threads = [go(i,data,verts,coeffs) for i in range(0,len(data))]

    try:
        while True:
            for thread in threads:
                thread.next()
            print verts

    except Exception as e:
        print e, data
 
def square(x):
    return x*x

def go(i,data,verts,coeffs):
    l = len(data)

    # Perform the first set of reductions.
    otheri = i - 1 if i & 1 else i + 1
    other = data[otheri]

    out = data[i]

    if out > other:
        out = other + 1
        verts[i] = otheri
        coeffs[i] = other
    else:
        verts[i]  = i
        coeffs[i] = out

    yield

    j = i & ~1 # points to the bottom of this frame

    mask = 1
    size = 4

    while l > 0: 
        l = l >> 1 # Yes, this means we have a logarithmic algorithm =)
        out = min(out, square(i - verts[j]) + coeffs[j], square(i - verts[j+1]) + coeffs[j+1])
        
        yield
        dest = (j>>1) & ~1

        # choose two threads to write the top and bottom of the new frame
        if 0 == (i % size):
            print "thread",i,"is merging bottom"

            existingv = verts[j]
            existingc = coeffs[j]
            otherv = verts[j+2]
            otherc = coeffs[j+2]

            if (square(i -otherv) + otherc) < (square(i -existingv) + existingc):
                verts[dest] = otherv
                coeffs[dest] = otherc
            else:
                verts[dest] = existingv
                coeffs[dest] = existingc

        elif  (size-1) == (i % size):

            print "thread",i,"is merging top"

            existingv = verts[j-1]
            existingc = coeffs[j-1]
            otherv = verts[j+1]
            otherc = coeffs[j+1]

            if (square(i -otherv) + otherc) < (square(i -existingv) + existingc):
                verts[dest+1] = otherv
                coeffs[dest+1] = otherc
            else:
                verts[dest+1] = existingv
                coeffs[dest+1] = existingc
                        
        yield
        j = dest
        size = size << 1
        
    data[i] = out
    yield
