# raytracer/colors/color.py

from raytracer.tuples import Tuple

from typing import Union


class Color(Tuple):
    def __init__(self, r: float, g: float, b: float):
        super().__init__(r, g, b, 0)

    def __mul__(self, other: Union[int, float, "Color"]):
        if isinstance(other, (int, float)):
            return Color(self.x * other, self.y * other, self.z * other)

        if isinstance(other, Color):
            return Color(self.x * other.x, self.y * other.y, self.z * other.z)
