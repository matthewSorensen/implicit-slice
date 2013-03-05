



def run_threads(data):
    verts  = [0 for x in data]
    coeffs = [0 for x in data]

    threads = [go(i,data,verts,coeffs) for i in range(0,len(data))]

    try:
        while True:
            for thread in threads:
                thread.next()

    except Exception as e:
        print e, data
 
def square(x):
    return x*x

def go(i,data,verts,coeffs):
    l = len(data)
    # First we write the value that this thread starts with
    # to the shared memory and synchronize with the other threads.
    out = data[i]
    coeffs[i] = out
    yield
    # Then perform the initial reduction step - each pair of threads
    # computes the most local distance transform
    otherindex = i ^ 1
    otherdata  = coeffs[otherindex]
    if otherdata < out:
        coeffs[i] = otherdata + 1
        verts[i] = otherindex
    else:
        verts[i] = i

    yield
    # Now start gradually building bigger and bigger chunks of local distance transforms
    j = i & ~1 # points to the bottom of this chunk
    mask = 3
    while l > 0: 
        l = l >> 1 # Yes, this means we have a logarithmic algorithm =)

        base = j & ~3;
        dest = (j >> 1) & ~1
        par = i & mask
        offset = base ^ j
        half = offset >> 1
        offset = offset | half
        antioffset = (2 + offset) & 3

        low = square(i - verts[base + 1]) + coeffs[base + 1]
        high = square(i - verts[base + 2]) + coeffs[base + 2]
        extreme = square(i - verts[base + offset]) + coeffs[base + offset]

        out = min(out,high,low,extreme)
        
        vertex = None
        coefficient = None

        if par == 0 or par == mask:
            if high < extreme or low < extreme:
                vertex = verts[base + antioffset]
                coefficient = coeffs[base + antioffset]
            else:
                vertex = verts[base + offset]
                coefficient = coeffs[base + offset]
        
        yield
        
        if not vertex is None:
            coeffs[dest + half] = coefficient;
            verts[dest + half] = vertex;
    
        yield

        j = dest
        mask = (mask << 1) + 1;    

    data[i] = out
    yield
