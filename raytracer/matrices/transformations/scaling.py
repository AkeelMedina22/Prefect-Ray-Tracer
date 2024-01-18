# raytracer/matrices/transformations/scaling.py

from raytracer.matrices import Matrix

def scaling(sx, sy, sz):
    """Return a matrix for scaling by (sx, sy, sz)."""
    return Matrix(4, 4, [
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])