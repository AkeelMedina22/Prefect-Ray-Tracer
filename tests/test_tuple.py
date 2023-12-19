from prefect import task, flow
from prefect_ray.task_runners import RayTaskRunner
from raytracer.tuples import Tuple, Point, Vector

import math
from multiprocessing import cpu_count

NUM_CPUS = cpu_count()


@task
def test_tuple_as_point():
    a = Tuple(4.3, -4.2, 3.1, 1.0)
    assert a.x == 4.3
    assert a.y == -4.2
    assert a.z == 3.1
    assert a.w == 1.0
    assert a.isPoint() is True
    assert a.isVector() is False


@task
def test_tuple_as_vector():
    a = Tuple(4.3, -4.2, 3.1, 0.0)
    assert a.x == 4.3
    assert a.y == -4.2
    assert a.z == 3.1
    assert a.w == 0.0
    assert a.isPoint() is False
    assert a.isVector() is True


@task
def test_point_factory_function():
    p = Point(4, -4, 3)
    assert str(p) == str(Tuple(4, -4, 3, 1))


@task
def test_vector_factory_function():
    v = Vector(4, -4, 3)
    assert str(v) == str(Tuple(4, -4, 3, 0))


@task
def test_tuple_addition():
    a1 = Tuple(3, -2, 5, 1)
    a2 = Tuple(-2, 3, 1, 0)
    assert a1 + a2 == Tuple(1, 1, 6, 1)


@task
def test_point_subtraction():
    p1 = Point(3, 2, 1)
    p2 = Point(5, 6, 7)
    assert p1 - p2 == Vector(-2, -4, -6)


@task
def test_vector_subtraction():
    p = Point(3, 2, 1)
    v = Vector(5, 6, 7)
    assert p - v == Point(-2, -4, -6)


@task
def test_vector_vector_subtraction():
    v1 = Vector(3, 2, 1)
    v2 = Vector(5, 6, 7)
    assert v1 - v2 == Vector(-2, -4, -6)


@task
def test_vector_subtraction_from_zero_vector():
    zero = Vector(0, 0, 0)
    v = Vector(1, -2, 3)
    assert zero - v == Vector(-1, 2, -3)


@task
def test_tuple_negation():
    a = Tuple(1, -2, 3, -4)
    assert -a == Tuple(-1, 2, -3, 4)


@task
def test_tuple_scalar_multiplication():
    a = Tuple(1, -2, 3, -4)
    assert a * 3.5 == Tuple(3.5, -7, 10.5, -14)


@task
def test_tuple_fractional_multiplication():
    a = Tuple(1, -2, 3, -4)
    assert a * 0.5 == Tuple(0.5, -1, 1.5, -2)


@task
def test_tuple_scalar_division():
    a = Tuple(1, -2, 3, -4)
    assert a / 2 == Tuple(0.5, -1, 1.5, -2)


@task
def test_tuple_magnitude():
    assert Vector(1, 0, 0).magnitude() == 1
    assert Vector(0, 1, 0).magnitude() == 1
    assert Vector(0, 0, 1).magnitude() == 1
    assert Vector(1, 2, 3).magnitude() == math.sqrt(14)
    assert Vector(-1, -2, -3).magnitude() == math.sqrt(14)


@task
def test_tuple_normalize():
    assert Vector(4, 0, 0).normalize() == Vector(1, 0, 0)
    assert Vector(1, 2, 3).normalize() == Vector(0.26726, 0.53452, 0.80178)


@task
def test_tuple_dot_product():
    a = Vector(1, 2, 3)
    b = Vector(2, 3, 4)
    assert a.dot(b) == 20


@task
def test_tuple_cross_product():
    a = Vector(1, 2, 3)
    b = Vector(2, 3, 4)
    assert a.cross(b) == Vector(-1, 2, -1)
    assert b.cross(a) == Vector(1, -2, 1)


@flow(task_runner=RayTaskRunner(init_kwargs={"num_cpus": NUM_CPUS}))
def test_tuple() -> None:
    test_tuple_as_point()
    test_tuple_as_vector()
    test_point_factory_function()
    test_vector_factory_function()
    test_tuple_addition()
    test_point_subtraction()
    test_vector_subtraction()
    test_vector_vector_subtraction()
    test_vector_subtraction_from_zero_vector()
    test_tuple_negation()
    test_tuple_scalar_multiplication()
    test_tuple_fractional_multiplication()
    test_tuple_scalar_division()
    test_tuple_magnitude()
    test_tuple_normalize()
    test_tuple_dot_product()
    test_tuple_cross_product()
