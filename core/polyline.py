import numpy as np

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

    def join(self,eps,other):
        if eps >= np.norm(self.points[-1] - other.points[0]):
            self.points.extend(other.points[1:])
        else:
            self.points.extend(other.points)
        return self
