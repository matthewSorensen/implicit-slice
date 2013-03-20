import numpy as np
import util
import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def area(x,y):
    a,b = x
    c,d = y

    return 0.5 * abs(a * d - b * c)

class Polyline:

    def __init__(self,a,*b):
        if b == ():
            self.points = [x for x in a]
        else:
            self.points = [a]
            self.points.extend(b)

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

    def plot(self):
        ax = plt.gca()
        patch = patches.Polygon(self.points)
        ax.add_patch(patch)

#        plt.plot(self.points)

    def simplify(self,threshold):
        """ Implementation of Visvalingam's algorithm for line simplification """
        l = len(self.points)

        neighbors = [(i-1,i+1) for i in range(l)]
        neighbors[0] = None
        neighbors[-1] = None
        triangles = [0] * (l - 2)
        for i in range(1,l-1):
            here = self.points[i]
            size = area(self.points[i-1] - here, self.points[i+1] - here)
            triangles[i-1] = (size, i)

        heapq.heapify(triangles)
    
        while [] != triangles:
            size, i = heapq.heappop(triangles)
            if size > threshold:
                break

            entry = neighbors[i]
            if entry is None:
                continue

            lower, upper = entry
            
            if neighbors[lower] is None or neighbors[upper] is None:
                continue
            lowerp = self.points[lower]
            upperp = self.points[upper]
            diff = lowerp - upperp

            ll, lu = neighbors[lower]
            
            heapq.heappush(triangles, (area(self.points[ll] - lowerp, diff), lower))

            hl, hu = neighbors[upper]
        
            heapq.heappush(triangles, (area(self.points[hu] - upperp, diff), upper))
            
            neighbors[i] = None
            neighbors[lower] = (ll,upper)
            neighbors[upper] = (lower,hu)

        neighbors[0] = 0
        neighbors[-1] = 0

        self.points = [self.points[i] for i in range(l) if not neighbors[i] is None]


def fast_simplify(points):    
    """ Performs preliminary simplification """

    start = points.next()
    yield start
    end = points.next()
    dirx, diry = end - start
    for p in points:
        dx, dy = p - start
        
        if diry * dx != dy  * dirx:
            yield end
            start = end
            end = p
            dirx, diry = end - start
        end = p
    yield end

def polylines_via_matplotlib(data,levels):
    """ This is a completely horrific hack and needs to be expunged asap. """

    plt.clf()
    contour = plt.contour(data,levels = levels)
    for x in contour.collections:
        for y in x._paths:
            yield Polyline(fast_simplify(iter(y.vertices)))

# Look into homothetic centers to perform initial segment fitting. With some assumptions, these give correct results.
