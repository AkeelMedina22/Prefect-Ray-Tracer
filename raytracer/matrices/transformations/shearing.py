# raytracer/matrices/transformations/shearing.py

from raytracer.matrices import Matrix

def shearing(xy, xz, yx, yz, zx, zy):
    """Return a matrix for shearing."""
    return Matrix(4, 4, [
        [1, xy, xz, 0],
        [yx, 1, yz, 0],
        [zx, zy, 1, 0],
        [0, 0, 0, 1]
    ])