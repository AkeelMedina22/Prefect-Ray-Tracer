from .tuple import Tuple

def Point(x, y, z):
    return Tuple(x, y, z, 1)

def Vector(x, y, z):
    return Tuple(x, y, z, 0)
