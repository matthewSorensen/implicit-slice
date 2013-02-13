import numpy as np
import util

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

def from_sdf_corners(corners,sdf):        
    (starts, starte), (ends, ende) = sign_changes(sdf)[:2]
    # Then linearly interpolate to find the endpoints for this segment   
    start = util.lerp(corners[starts], corners[starte], abs(sdf[starts]))
    end = util.lerp(corners[ends], corners[ende], abs(sdf[ends]))

    return Polyline(start,end)
