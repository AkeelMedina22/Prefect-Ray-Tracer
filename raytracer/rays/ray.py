# raytracer/rays/ray.py

from raytracer.tuples import Point, Vector


class Ray:
    def __init__(self, origin: Point, direction: Vector):
        self.origin: Point = origin
        self.direction: Vector = direction

    def position(self, t: float) -> Point:
        return self.origin + self.direction * t

    def intersect(self, shape):
        return shape.intersect(self)