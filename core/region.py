import numpy as np
import math

ihat = np.array([1,0])
jhat = np.array([0,1])
ones = np.array([1,1])
zeros = np.array([0,0])

def invert(index):
    """ Given r.index(p.center()), compute p.index(r.center()) """
    return [2,3,0,1][index]

def comps(vect):
    """ Project a vector onto i and j """
    x,y = vect
    return ihat * x, jhat * y

class InvalidRegionMerge(Exception):
    pass

class SpatialRegion:
    """ Rectangular regions of space """
    def __init__(self,lower_left,span):
        self.ll = lower_left
        self.span = span

    def index(self,point):
        half_span = self.span * 0.5
        displace = point - half_span - self.ll
        dx, dy = half_span * np.absolute(displace)        

        if dy < dx:
            return 0 if displace[0] < 0 else 2
        else:
            return 3 if displace[1] < 0 else 1

    def corners(self):
        """ List corners of this region, in clockwise order from lower-left """
        dx, dy = comps(self.span)
        return [self.ll, self.ll + dy, self.ll + self.span, self.ll + dx]

    def center(self):
        return self.ll + 0.5 * self.span

    def merge(self,other):
        """ Destructively merge a region into this region """
        disp = other.ll - self.ll
        if not any(zeros == disp):
            raise InvalidRegionMerge()
        # We have exactly one non-zero component of the displacement
        upright = np.maximum(self.span + self.ll,other.span + other.ll)
        if any(zeros > disp):
            self.ll = other.ll
        self.span = upright - self.ll

    def quarterable(self):
        """ Is this region 4-divisible? """
        return not any(ones == self.span)

    def quarter(self):
        """ Enumerate four subregions of a 4-divisible region """
        low = self.span / 2

        ux, uy = comps(self.span - low)
        lx, ly = comps(low)

        yield SpatialRegion(self.ll, low)
        yield SpatialRegion(self.ll + ly, uy + lx)
        yield SpatialRegion(self.ll + low, ux + uy)
        yield SpatialRegion(self.ll + lx, ly + ux)
        
    def pixels(self):
        """ Enumerate all unit regions in a given region """
        x,y = self.span
        for dx in range(x):
            for dy in range(y):
                yield SpatialRegion(self.ll + ihat * dx + jhat * dy ,ones)

    def quadtree_recursion_scheme(self,pred,transfer,merge):
        if self.quarterable() and pred(self):
            v = self.quarter()
            q1 = v.next().quadtree_recursion_scheme(pred,transfer,merge)
            q2 = v.next().quadtree_recursion_scheme(pred,transfer,merge)
            q3 = v.next().quadtree_recursion_scheme(pred,transfer,merge)
            q4 = v.next().quadtree_recursion_scheme(pred,transfer,merge)
            return merge(merge(q1,q2),merge(q3,q4))

        ret = None
        for x in self.pixels():
            if ret is None:
                ret = transfer(x)
            else:
                ret = merge(ret,transfer(x))
        return ret

unit_box = SpatialRegion(zeros,ones)
