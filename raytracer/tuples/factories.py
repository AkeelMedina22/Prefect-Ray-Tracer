from .tuple import Tuple

class Vector(Tuple):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, 0)

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x)

class Point(Tuple):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, 1)