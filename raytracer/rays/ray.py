# raytracer/rays/ray.py

from raytracer.tuples import Point, Vector


class Ray:
    def __init__(self, origin: Point, direction: Vector):
        self.origin: Point = origin
        self.direction: Vector = direction