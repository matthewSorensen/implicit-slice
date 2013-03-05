def run_threads(data):
    l = len(data)

    shared = [0 for x in range(0,l + 2)]
    threads = [go(i,data,shared, l * l) for i in range(0,len(data))]

    try:
        while True:
            for thread in threads:
                thread.next()

    except Exception as e:
        print e, data
 
def sgn(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1

def go(i,data,shared,replace):
    l = len(data)

    this = sgn(data[i])
    shared[i+1] = this
    # either overfetch or duplicate - in the one block case, we just duplicate
    if i == 0:
        shared[i] = this
    if i == l - 1:
        shared[i+2] = this
    yield
   
    if this == 1 and (shared[i] == -1 or shared[i+2] == -1):
        data[i] = 0
    else:
        data[i] = this * replace

    yield


    
