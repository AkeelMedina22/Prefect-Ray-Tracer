# raytracer/matrices/transformations/rotation.py

from math import cos, sin
from raytracer.matrices import Matrix

def rotation_x(r):
    """Return a matrix for rotating points around the x-axis."""
    return Matrix(4, 4, [
        [1, 0, 0, 0],
        [0, cos(r), -sin(r), 0],
        [0, sin(r), cos(r), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(r):
    """Return a matrix for rotating points around the y-axis."""
    return Matrix(4, 4, [
        [cos(r), 0, sin(r), 0],
        [0, 1, 0, 0],
        [-sin(r), 0, cos(r), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(r):
    """Return a matrix for rotating points around the z-axis."""
    return Matrix(4, 4, [
        [cos(r), -sin(r), 0, 0],
        [sin(r), cos(r), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])