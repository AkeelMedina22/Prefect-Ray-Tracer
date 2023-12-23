from prefect import task, flow
from prefect_ray.task_runners import RayTaskRunner
from raytracer.matrices import Matrix
from raytracer.tuples import Tuple, Point, Vector
from raytracer.utils import identity_matrix

import math
from multiprocessing import cpu_count

NUM_CPUS = cpu_count()


@task
def test_matrix_elements_4x4():
    m = Matrix(
        4,
        4,
        [
            [1, 2, 3, 4],
            [5.5, 6.5, 7.5, 8.5],
            [9, 10, 11, 12],
            [13.5, 14.5, 15.5, 16.5],
        ],
    )
    assert m.elements[0][0] == 1
    assert m.elements[0][3] == 4
    assert m.elements[1][0] == 5.5
    assert m.elements[1][2] == 7.5
    assert m.elements[2][2] == 11
    assert m.elements[3][0] == 13.5
    assert m.elements[3][2] == 15.5


@task
def test_matrix_elements_2x2():
    m = Matrix(2, 2, [[-3, 5], [1, -2]])
    assert m.elements[0][0] == -3
    assert m.elements[0][1] == 5
    assert m.elements[1][0] == 1
    assert m.elements[1][1] == -2


@task
def test_matrix_elements_3x3():
    m = Matrix(3, 3, [[-3, 5, 0], [1, -2, -7], [0, 1, 1]])
    assert m.elements[0][0] == -3
    assert m.elements[1][1] == -2
    assert m.elements[2][2] == 1


@task
def test_matrix_equality():
    m1 = Matrix(4, 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    m2 = Matrix(4, 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    assert m1 == m2


@task
def test_matrix_inequality():
    m1 = Matrix(4, 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    m2 = Matrix(4, 4, [[2, 3, 4, 5], [6, 7, 8, 9], [8, 7, 6, 5], [4, 3, 2, 1]])
    assert m1 != m2


@task
def test_matrix_multiplication():
    m1 = Matrix(4, 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    m2 = Matrix(4, 4, [[-2, 1, 2, 3], [3, 2, 1, -1], [4, 3, 6, 5], [1, 2, 7, 8]])
    assert m1 * m2 == Matrix(
        4,
        4,
        [
            [20, 22, 50, 48],
            [44, 54, 114, 108],
            [40, 58, 110, 102],
            [16, 26, 46, 42],
        ],
    )


@task
def test_matrix_multiplication_by_tuple():
    m = Matrix(
        4,
        4,
        [
            [1, 2, 3, 4],
            [2, 4, 4, 2],
            [8, 6, 4, 1],
            [0, 0, 0, 1],
        ],
    )
    t = Tuple(1, 2, 3, 1)
    assert m * t == Tuple(18, 24, 33, 1)


@task
def test_matrix_multiplication_by_identity_matrix():
    m = Matrix(
        4,
        4,
        [
            [0, 1, 2, 4],
            [1, 2, 4, 8],
            [2, 4, 8, 16],
            [4, 8, 16, 32],
        ],
    )
    assert m * Matrix(4, 4, identity_matrix(4)) == m


@task
def test_matrix_multiplication_by_identity_matrix_opposite():
    m = Matrix(
        4,
        4,
        [
            [0, 1, 2, 4],
            [1, 2, 4, 8],
            [2, 4, 8, 16],
            [4, 8, 16, 32],
        ],
    )
    assert Matrix(4, 4, identity_matrix(4)) * m == m


@task
def test_matrix_transposition():
    m = Matrix(
        4,
        4,
        [
            [0, 9, 3, 0],
            [9, 8, 0, 8],
            [1, 8, 5, 3],
            [0, 0, 5, 8],
        ],
    )
    assert m.transpose() == Matrix(
        4,
        4,
        [
            [0, 9, 1, 0],
            [9, 8, 8, 0],
            [3, 0, 5, 5],
            [0, 8, 3, 8],
        ],
    )


@task
def test_matrix_transposition_identity():
    assert Matrix(4, 4, identity_matrix(4)).transpose() == Matrix(4, 4, identity_matrix(4))


@task
def test_matrix_determinant_2x2():
    m = Matrix(2, 2, [[1, 5], [-3, 2]])
    assert m.determinant() == 17


@task
def test_submatrix_3x3():
    m = Matrix(3, 3, [[1, 5, 0], [-3, 2, 7], [0, 6, -3]])
    assert m.submatrix(0, 2) == Matrix(2, 2, [[-3, 2], [0, 6]])


@task
def test_submatrix_4x4():
    m = Matrix(
        4,
        4,
        [
            [-6, 1, 1, 6],
            [-8, 5, 8, 6],
            [-1, 0, 8, 2],
            [-7, 1, -1, 1],
        ],
    )
    assert m.submatrix(2, 1) == Matrix(3, 3, [[-6, 1, 6], [-8, 8, 6], [-7, -1, 1]])


@task
def test_matrix_minor():
    m = Matrix(
        3,
        3,
        [
            [3, 5, 0],
            [2, -1, -7],
            [6, -1, 5],
        ],
    )
    submatrix = m.submatrix(1, 0)
    assert submatrix.determinant() == 25
    assert m.minor(1, 0) == 25


@task
def test_matrix_cofactor():
    m = Matrix(
        3,
        3,
        [
            [3, 5, 0],
            [2, -1, -7],
            [6, -1, 5],
        ],
    )
    assert m.minor(0, 0) == -12
    assert m.cofactor(0, 0) == -12
    assert m.minor(1, 0) == 25
    assert m.cofactor(1, 0) == -25


@flow(task_runner=RayTaskRunner(init_kwargs={"num_cpus": NUM_CPUS}))
def test_matrix() -> None:
    test_matrix_elements_4x4()
    test_matrix_elements_2x2()
    test_matrix_elements_3x3()
    test_matrix_equality()
    test_matrix_inequality()
    test_matrix_multiplication()
    test_matrix_multiplication_by_tuple()
    test_matrix_multiplication_by_identity_matrix()
    test_matrix_multiplication_by_identity_matrix_opposite()
    test_matrix_transposition()
    test_matrix_transposition_identity()
    test_matrix_determinant_2x2()
    test_submatrix_3x3()
    test_submatrix_4x4()
    test_matrix_minor()
    test_matrix_cofactor()
    