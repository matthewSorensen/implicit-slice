import numpy as np
import util
import math

class Polyline:

    def __init__(self,a,b):
        self.points = [a,b]
        self.closed = False

    def close(self):
        self.closed = True
        return self

    def reverse(self):
        self.points.reverse()
        return self

    def head(self):
        return self.points[0]

    def tail(self):
        return self.points[-1]

    def join(self,direction,other,other_direction,eps = 0):
        """ Joins a second polyline to this one, maybe modifying the second 
        returns whether the first element of this line is unchanged """

        unchanged = True

        if direction:
            self.reverse()
            unchanged = False
        if not other_direction:
            other.reverse()
        if eps < np.linalg.norm(self.tail() - other.head()):
            self.points.extend(other.points)
        else:
            self.points.extend(other.points[1:])

        return unchanged

def sign_changes(points):
    changes = []
    prev = util.sgn(points[0])
    l = len(points)
    for i, point in enumerate(points + [points[0]]):
        if not util.sgn(point) == prev:
            changes = changes + [((i -1) % l, i % l)]
        prev = util.sgn(point)
    return changes

def intersect_unitbox(point,normal):
    """ Always returns two intersections """

    tnear, tfar = -1 * util.infinity, util.infinity
    x,y = point
    dx,dy = normal

    if dx == 0:
        return -1 * y / dy, (1-y) / dy

    t1 = -1 * x / dx
    t2 = (1-x) /dx
    if t1 > t2:
        t1, t2 = t2, t1
    if t1 > tnear:
        tnear = t1
    if t2 < tfar:
        tfar = t2
    if dy == 0:
        return tnear, tfar
    t1 = -1 * y / dy
    t2 = (1-y) /dy
    if t1 > t2:
        t1, t2 = t2, t1
    if t1 > tnear:
        tnear = t1
    if t2 < tfar:
        tfar = t2
    return tnear, tfar

def from_sdf_corners(region,sdf):        
    grad = 0.5 * util.jhat * (sdf[1] - sdf[0] + sdf[2] - sdf[3])
    grad += 0.5 * util.ihat * (sdf[2] - sdf[1] - sdf[3] - sdf[0])
    grad = np.array([-1 * grad[1], grad[0]])    

    corners = list(region.corners())
    center = region.center()

    (starts, starte), (ends, ende) = sign_changes(sdf)[:2]

    theta = math.atan2(grad[1], grad[0])

    half = [0.5 * util.jhat, util.jhat + 0.5 * util.ihat, util.ihat + 0.5 * util.jhat, 0.5 * util.ihat]

    if starte == ends or starts == ende:
        start = None
        if theta < math.pi / 4 or (theta > math.pi and theta < math.pi * (7/8)):
            start = half[ends]
        else:
            start = half[ende]
        
        t = sum(intersect_unitbox(start, grad))
        end = t * grad + start + region.ll
        return Polyline(start + region.ll, end)
    else:
        v = 0.5 * util.ones
        tmin, tmax = intersect_unitbox(v, grad)
        start = center  + tmin * grad
        end = tmax * grad + center 
        return Polyline(start,end)
