# raytracer/tuples/tuple.py

import math
from raytracer.utils import float_equal


class Tuple:
    def __init__(self, x: float, y: float, z: float, w: int):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.w: int = w

    def isPoint(self):
        return self.w == 1

    def isVector(self):
        return self.w == 0

    def __eq__(self, other):
        if not isinstance(other, Tuple):
            return NotImplemented
        return (
            float_equal(self.x, other.x)
            and float_equal(self.y, other.y)
            and float_equal(self.z, other.z)
            and float_equal(self.w, other.w)
        )

    def __add__(self, other):
        if not isinstance(other, Tuple):
            return NotImplemented
        return Tuple(
            self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w
        )

    def __sub__(self, other):
        if not isinstance(other, Tuple):
            return NotImplemented
        return Tuple(
            self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w
        )

    def __neg__(self):
        return Tuple(-self.x, -self.y, -self.z, -self.w)

    def __mul__(self, scalar):
        return Tuple(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)

    def __truediv__(self, scalar):
        return Tuple(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def normalize(self):
        mag = self.magnitude()
        return Tuple(self.x / mag, self.y / mag, self.z / mag, self.w / mag)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def __repr__(self):
        return f"Tuple({self.x}, {self.y}, {self.z}, {self.w})"
