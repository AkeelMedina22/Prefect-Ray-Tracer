# raytracer/tuples/tuple.py

class Tuple:
    def __init__(self, x, y, z, w):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.w: int = int(w)

    def isPoint(self):
        return self.w == 1

    def isVector(self):
        return self.w == 0

    def normalize(self):
        # Implementation for normalization goes here
        pass

    def __repr__(self):
        return f"Tuple({self.x}, {self.y}, {self.z}, {self.w})"
