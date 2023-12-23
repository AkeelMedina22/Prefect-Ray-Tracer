# raytracer/matrices/matrix.py

import math
from typing import List
from raytracer.utils import float_equal
from raytracer.tuples import Tuple


class Matrix:
    def __init__(self, rows: int, columns: int, elements: List[List[float]]):
        self.rows: int = rows
        self.columns: int = columns
        self.elements: List[List[float]] = elements

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        return (
            self.rows == other.rows
            and self.columns == other.columns
            and all(
                float_equal(self.elements[row][column], other.elements[row][column])
                for row in range(self.rows)
                for column in range(self.columns)
            )
        )

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.multiplyMatrix(other)
        elif isinstance(other, Tuple):
            return self.multiplyTuple(other)
        else:
            return NotImplemented

    def multiplyMatrix(self, other: 'Matrix'):
        if self.columns != other.rows:
            raise ValueError(
                f"Cannot multiply matrices with {self.columns} columns and {other.rows} rows"
            )
        result = Matrix(
            self.rows, other.columns, [[0] * other.columns for _ in range(self.rows)]
        )
        for row in range(self.rows):
            for column in range(other.columns):
                result.elements[row][column] = sum(
                    self.elements[row][i] * other.elements[i][column]
                    for i in range(self.columns)
                )
        return result

    def multiplyTuple(self, other: Tuple):
        if self.columns != 4:
            raise ValueError(
                f"Cannot multiply matrix with {self.columns} columns by tuple"
            )

        result = self.multiplyMatrix(Matrix(4, 1, [[other.x], [other.y], [other.z], [other.w]]))

        return Tuple(result.elements[0][0], result.elements[1][0], result.elements[2][0], result.elements[3][0])

    def transpose(self):
        return Matrix(
            self.columns,
            self.rows,
            [[self.elements[row][column] for row in range(self.rows)] for column in range(self.columns)],
        )
