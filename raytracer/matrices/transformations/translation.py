# raytracer/matrices/transformations/translation.py

from raytracer.matrices import Matrix

def translation(dx, dy, dz):
    """Return a matrix for translating points by (dx, dy, dz)."""
    return Matrix(4, 4, [
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])
