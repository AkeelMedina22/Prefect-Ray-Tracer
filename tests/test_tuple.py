from prefect import task, flow
from prefect_ray.task_runners import RayTaskRunner
from raytracer.tuples import Tuple, Point, Vector


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


@flow(task_runner=RayTaskRunner(init_kwargs={"num_cpus": 4}))
def test_tuple() -> None:
    test_tuple_as_point()
    test_tuple_as_vector()
    test_point_factory_function()
    test_vector_factory_function()
