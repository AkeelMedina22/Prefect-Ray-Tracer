from prefect import task, flow
from prefect_ray.task_runners import RayTaskRunner
from raytracer.colors import Color

import math
from multiprocessing import cpu_count

NUM_CPUS = cpu_count()


@task
def test_add_color():
    c1 = Color(0.9, 0.6, 0.75)
    c2 = Color(0.7, 0.1, 0.25)
    assert c1 + c2 == Color(1.6, 0.7, 1.0)


@task
def test_subtract_color():
    c1 = Color(0.9, 0.6, 0.75)
    c2 = Color(0.7, 0.1, 0.25)
    assert c1 - c2 == Color(0.2, 0.5, 0.5)


@task
def test_multiply_color_by_scalar():
    c = Color(0.2, 0.3, 0.4)
    assert c * 2 == Color(0.4, 0.6, 0.8)


@task
def test_multiply_colors():
    c1 = Color(1, 0.2, 0.4)
    c2 = Color(0.9, 1, 0.1)
    assert c1 * c2 == Color(0.9, 0.2, 0.04)


@flow(task_runner=RayTaskRunner(init_kwargs={"num_cpus": NUM_CPUS}))
def test_color() -> None:
    test_add_color()
    test_subtract_color()
    test_multiply_color_by_scalar()
    test_multiply_colors()
