from prefect import task, flow
from prefect_ray.task_runners import RayTaskRunner
from raytracer.matrices import Matrix
from raytracer.matrices.transformations import translation, scaling, rotation_x, rotation_y, rotation_z, shearing
from raytracer.tuples import Point, Vector
from raytracer.utils import identity_matrix

import math
from multiprocessing import cpu_count

NUM_CPUS = cpu_count()


@task
def test_translation():
    transform = translation(5, -3, 2)
    p = Point(-3, 4, 5)
    assert transform * p == Point(2, 1, 7)


@task
def test_inverse_translation():
    transform = translation(5, -3, 2)
    inv = transform.inverse()
    p = Point(-3, 4, 5)
    assert inv * p == Point(-8, 7, 3)


@task
def test_translation_does_not_affect_vectors():
    transform = translation(5, -3, 2)
    v = Vector(-3, 4, 5)
    assert transform * v == v


@task
def test_scaling_matrix_applied_to_point():
    transform = scaling(2, 3, 4)
    p = Point(-4, 6, 8)
    assert transform * p == Point(-8, 18, 32)


@task
def test_scaling_matrix_applied_to_vector():
    transform = scaling(2, 3, 4)
    v = Vector(-4, 6, 8)
    assert transform * v == Vector(-8, 18, 32)


@task
def test_inverse_scaling_matrix_applied_to_vector():
    transform = scaling(2, 3, 4)
    inv = transform.inverse()
    v = Vector(-4, 6, 8)
    assert inv * v == Vector(-2, 2, 2)


@task
def test_reflection_is_scaling_by_a_negative_value():
    transform = scaling(-1, 1, 1)
    p = Point(2, 3, 4)
    assert transform * p == Point(-2, 3, 4)


@task
def test_rotating_a_point_around_the_x_axis():
    p = Point(0, 1, 0)
    half_quarter = rotation_x(math.pi / 4)
    full_quarter = rotation_x(math.pi / 2)
    assert half_quarter * p == Point(0, math.sqrt(2) / 2, math.sqrt(2) / 2)
    assert full_quarter * p == Point(0, 0, 1)


@task
def test_inverse_rotating_a_point_around_the_x_axis():
    p = Point(0, 1, 0)
    half_quarter = rotation_x(math.pi / 4)
    inv = half_quarter.inverse()
    assert inv * p == Point(0, math.sqrt(2) / 2, -math.sqrt(2) / 2)


@task
def test_rotating_a_point_around_the_y_axis():
    p = Point(0, 0, 1)
    half_quarter = rotation_y(math.pi / 4)
    full_quarter = rotation_y(math.pi / 2)
    assert half_quarter * p == Point(math.sqrt(2) / 2, 0, math.sqrt(2) / 2)
    assert full_quarter * p == Point(1, 0, 0)


@task
def test_rotating_a_point_around_the_z_axis():
    p = Point(0, 1, 0)
    half_quarter = rotation_z(math.pi / 4)
    full_quarter = rotation_z(math.pi / 2)
    assert half_quarter * p == Point(-math.sqrt(2) / 2, math.sqrt(2) / 2, 0)
    assert full_quarter * p == Point(-1, 0, 0)


@task
def test_shearing_transformation_moves_x_in_proportion_to_y():
    transform = shearing(1, 0, 0, 0, 0, 0)
    p = Point(2, 3, 4)
    assert transform * p == Point(5, 3, 4)


@task
def test_shearing_transformation_moves_x_in_proportion_to_z():
    transform = shearing(0, 1, 0, 0, 0, 0)
    p = Point(2, 3, 4)
    assert transform * p == Point(6, 3, 4)


@task
def test_shearing_transformation_moves_y_in_proportion_to_x():
    transform = shearing(0, 0, 1, 0, 0, 0)
    p = Point(2, 3, 4)
    assert transform * p == Point(2, 5, 4)


@task
def test_shearing_transformation_moves_y_in_proportion_to_z():
    transform = shearing(0, 0, 0, 1, 0, 0)
    p = Point(2, 3, 4)
    assert transform * p == Point(2, 7, 4)


@task
def test_shearing_transformation_moves_z_in_proportion_to_x():
    transform = shearing(0, 0, 0, 0, 1, 0)
    p = Point(2, 3, 4)
    assert transform * p == Point(2, 3, 6)


@task
def test_shearing_transformation_moves_z_in_proportion_to_y():
    transform = shearing(0, 0, 0, 0, 0, 1)
    p = Point(2, 3, 4)
    assert transform * p == Point(2, 3, 7)


@task
def test_individual_transformations_applied_in_sequence():
    p = Point(1, 0, 1)
    a = rotation_x(math.pi / 2)
    b = scaling(5, 5, 5)
    c = translation(10, 5, 7)
    p2 = a * p
    assert p2 == Point(1, -1, 0)
    p3 = b * p2
    assert p3 == Point(5, -5, 0)
    p4 = c * p3
    assert p4 == Point(15, 0, 7)


@task
def test_chained_transformations_applied_in_reverse_order():
    p = Point(1, 0, 1)
    a = rotation_x(math.pi / 2)
    b = scaling(5, 5, 5)
    c = translation(10, 5, 7)
    t = c * b * a
    assert t * p == Point(15, 0, 7)


@flow(task_runner=RayTaskRunner(init_kwargs={"num_cpus": NUM_CPUS}))
def test_matrix_transformations() -> None:
    test_translation()
    test_inverse_translation()
    test_translation_does_not_affect_vectors()
    test_scaling_matrix_applied_to_point()
    test_scaling_matrix_applied_to_vector()
    test_inverse_scaling_matrix_applied_to_vector()
    test_reflection_is_scaling_by_a_negative_value()
    test_rotating_a_point_around_the_x_axis()
    test_inverse_rotating_a_point_around_the_x_axis()
    test_rotating_a_point_around_the_y_axis()
    test_rotating_a_point_around_the_z_axis()
    test_shearing_transformation_moves_x_in_proportion_to_y()
    test_shearing_transformation_moves_x_in_proportion_to_z()
    test_shearing_transformation_moves_y_in_proportion_to_x()
    test_shearing_transformation_moves_y_in_proportion_to_z()
    test_shearing_transformation_moves_z_in_proportion_to_x()
    test_shearing_transformation_moves_z_in_proportion_to_y()
    test_individual_transformations_applied_in_sequence()
    test_chained_transformations_applied_in_reverse_order()
