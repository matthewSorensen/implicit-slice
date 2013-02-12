from region import *
import util

class TopologicalImpossibility(Exception):
    pass

def sign_changes(points):
    changes = []
    prev = util.sgn(points[0])
    l = len(points)
    for i, point in enumerate(points + [points[0]]):
        if not util.sgn(point) == prev:
            changes = changes + [((i -1) % l, i % l)]
        prev = util.sgn(point)
    return changes

def join_segments(eps,this_points, this_dir, other_points, other_dir):
    # we must be at the start of the other list
    if not other_dir:
        other_points.reverse()
    # and the end of this list
    if this_dir:
        this_points.reverse()
    if eps >= np.linalg.norm(this_points[-1] - other_points[0]):
        other_points = other_points[1:]
    this_points.extend(other_points)

def replace_references(op, old, new):
    for i in range(len(op)):
        curve, direction = op[i]
        if curve == old:
            op[i] = (new, direction)

class OccupiedRegion:
    def __init__(self,region,dcorners):
        self.region = region
        self.closed = []
        self.open = [[],[],[],[]]
        
        (starts, starte), (ends, ende) = sign_changes(dcorners)[:2]
        # Then linearly interpolate to find the endpoints for this segment
        x,y = comps(region.span)
        
        trans = [0.5 * y, y + 0.5 * x, x + 0.5 * y, 0.5 * x]

        start = region.ll + trans[starts]
        end = region.ll + trans[ends]

        self.segs = [[start,end]]
        self.open = [[],[],[],[]]
        self.open[self.region.index(start)] = [(0,True)]
        self.open[self.region.index(end)] = [(0,False)]

        # this needs to be parameterized
        self.epsilon = 0.001
 
    def merge_empty(self,other):
        """ Merges this occupied region with an empty region """
        if len(self.open[self.region.index(other.center())]) > 0:
            print self.segs, self.open
            print self.region.ll, self.region.span
            print other.ll, other.span
            raise TopologicalImpossibility()
        self.region.merge(other)

    def merge(self,other):
        """ Merges this occupied region with a second occupied region """

        side = self.region.index(other.region.center())
        to_join = other.open[invert(side)]
        to_join.reverse()
        if not len(to_join) == len(self.open[side]):
            raise TopologicalImpossibility()

        # Now we walk along self.open[side] and to_join in lockstep
        for i in range(len(self.open[side])):
            seg = self.open[side][i]
            indext, dirt = seg
            indexo, diro = to_join[i]
            # Fuse the two segments. if dirto is None, we may have a closure
            if not diro is None:
                join_segments(self.epsilon,self.segs[indext],dirt,other.segs[indexo],diro)
                other.segs[indexo] = None
            elif not indext == indexo:
                # we're joining segments in the same object, but not closing a curve
                join_segments(self.epsilon,self.segs[indexo],False,self.segs[indext],dirt)
                # all references to indexo now must be references to indext
                replace_references(self.open[side],indexo, indext)
            else:
                self.closed.append(self.segs[indext])
                self.segs[indext] = None
            # If the next index in to_join refers to the same curve,           
            if i+1 < len(to_join) and not diro is None and indexo == to_join[i+1][0]:
                # place this entire curve into to_join
                to_join[i+1] = (indext, None)

        self.region.merge(other.region)
        self.closed.extend(other.closed)
        self.segs = [seg for seg in self.segs + other.segs if not seg is None]
        # This isn't the best way to rebuild open, but it works
        self.open = [[],[],[],[]]
        for i, curve in enumerate(self.segs):
            self.open[self.region.index(curve[ 0])].append((i,True))
            self.open[self.region.index(curve[-1])].append((i,False))

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
