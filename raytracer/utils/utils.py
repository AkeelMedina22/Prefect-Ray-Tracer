from typing import List

def float_equal(a: float, b: float, tolerance=1e-5) -> bool:
    return abs(a - b) <= tolerance

def identity_matrix(size: int) -> List[List[float]]:
    return [[1 if row == column else 0 for column in range(size)] for row in range(size)]