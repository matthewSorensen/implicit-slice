import core.topo as topo
import core.region as region
import numpy as np
import matplotlib.pyplot as plt

sdf = np.loadtxt("test.dat")
r = region.SpatialRegion(region.zeros, 127 * region.ones)

result = r.quadtree_recursion_scheme(lambda x: topo.pred(sdf,x),lambda x: topo.transfer(sdf,x),topo.merge)

for p in result.closed[0]:
    print "{", p[0], "," , p[1] , "},"
