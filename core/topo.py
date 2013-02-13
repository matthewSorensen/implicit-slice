from region import *
from polyline import *
import util

class TopologicalImpossibility(Exception):
    pass

def replace_references(op, old, new):
    for i in range(len(op)):
        curve, direction = op[i]
        if curve == old:
            op[i] = (new, direction)

class OccupiedRegion:
    def __init__(self,region,sdf):
        self.region = region
        self.closed = []
        
        line = from_sdf_corners(list(region.corners()),sdf)        

        self.segs = [line]
        self.open = [[],[],[],[]]
        self.open[self.region.index(line.head())] = [(0,True)]
        self.open[self.region.index(line.tail())] = [(0,False)]

        self.ambi = [0 == d for d in sdf]
        # this needs to be parameterized
        self.epsilon = 0.001

    def merge_empty(self,other):
        """ Merges this occupied region with an empty region, resolving potentially ambiguous corners """

        side = self.region.index(other.center())
        self.region.merge(other)
        prev_side = backward(side)
        next_side = forward(side)

        if self.ambi[side]:
            self.open[prev_side].append(self.open[side].pop(0))
            self.ambi[side] = False

        if self.ambi[forward]:
            self.open[next_side].insert(0, self.open[side].pop())
            self.ambi[next_side] = False

        if self.open[side]:        
            raise TopologicalImpossibility()

    def merge(self,other):
        """ Merges this occupied region with a second occupied region """

        side = self.region.index(other.region.center())
        to_join = other.open[invert(side)]
        to_join.reverse()

        if not len(to_join) == len(self.open[side]):
            # this is where I would implement
            raise TopologicalImpossibility()

        # Now we walk along self.open[side] and to_join in lockstep
        for i in range(len(self.open[side])):
            seg = self.open[side][i]
            indext, dirt = seg
            indexo, diro = to_join[i]
            # Fuse the two segments. if dirto is None, we may have a closure
            if not diro is None:
                self.segs[indext].join(dirt, other.segs[indexo], diro, self.epsilon)
                other.segs[indexo] = None
            elif not indext == indexo:
                # we're joining segments in the same object, but not closing a curve
                self.segs[indexo].join(False, other.segs[indext], diro, self.epsilon)
                # all references to indexo now must be references to indext
                replace_references(self.open[side],indexo, indext)
            else:
                self.closed.append(self.segs[indext].close())
                self.segs[indext] = None
            # If the next index in to_join refers to the same curve,           
            if i+1 < len(to_join) and not diro is None and indexo == to_join[i+1][0]:
                # place this entire curve into to_join
                to_join[i+1] = (indext, None)

        self.region.merge(other.region)

        self.segs = [seg for seg in self.segs + other.segs if not seg is None]
        # This isn't the best way to rebuild open, but it works

        self.open = [[],[],[],[]]
        for i, curve in enumerate(self.segs): # This ordering is entirely wrong and such
            self.open[self.region.index(curve.head())].append((i,True))
            self.open[self.region.index(curve.tail())].append((i,False))

        self.closed.extend(other.closed)

def merge(ra,rb):
    if isinstance(ra,OccupiedRegion):
        if isinstance(rb,OccupiedRegion):
            ra.merge(rb)
        else:
            ra.merge_empty(rb)
        return ra
    elif isinstance(rb,OccupiedRegion):
        rb.merge_empty(ra)
    else:
        rb.merge(ra)
    return rb

def transfer(sdf,region):
    dcorners = [util.at(sdf,point) for point in region.corners()]
    sn = util.sgn(dcorners[0])
    for d in dcorners[1:]:
        if util.sgn(d) != sn:
            return OccupiedRegion(region,dcorners)
    return region

def pred(sdf,r):
    center  = r.center()
    dcenter = util.at(sdf,center)
    scenter = util.sgn(dcenter)
    dcenter = abs(dcenter)

    for corner in r.corners():
        dc = util.at(sdf,corner)
        if scenter != util.sgn(dc):
            return True    
        c = math.sqrt(2) * np.linalg.norm(corner - center)
        if c <= dcenter + abs(dc):
            return True
    return False
