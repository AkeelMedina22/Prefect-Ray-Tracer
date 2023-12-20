from prefect import task, flow
from prefect_ray.task_runners import RayTaskRunner
from raytracer.canvas import Canvas
from raytracer.colors import Color

import math
from multiprocessing import cpu_count

NUM_CPUS = cpu_count()


@task
def test_create_canvas():
    c = Canvas(10, 20)
    assert c.width == 10
    assert c.height == 20
    for row in c.pixels:
        for pixel in row:
            assert pixel == Color(0, 0, 0)


@task
def test_write_pixel():
    c = Canvas(10, 20)
    red = Color(1, 0, 0)
    c.write_pixel(2, 3, red)
    assert c.pixel_at(2, 3) == red


@task
def test_construct_ppm_header():
    c = Canvas(5, 3)
    ppm = c.to_ppm()
    lines = ppm.splitlines()
    assert lines[0] == "P3"
    assert lines[1] == "5 3"
    assert lines[2] == "255"


@task
def test_construct_ppm_pixel_data():
    c = Canvas(5, 3)
    c1 = Color(1.5, 0, 0)
    c2 = Color(0, 0.5, 0)
    c3 = Color(-0.5, 0, 1)
    c.write_pixel(0, 0, c1)
    c.write_pixel(2, 1, c2)
    c.write_pixel(4, 2, c3)
    ppm = c.to_ppm()
    lines = ppm.splitlines()
    assert lines[3] == "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    assert lines[4] == "0 0 0 0 0 0 0 128 0 0 0 0 0 0 0"
    assert lines[5] == "0 0 0 0 0 0 0 0 0 0 0 0 0 0 255"


@flow(task_runner=RayTaskRunner(init_kwargs={"num_cpus": NUM_CPUS}))
def test_canvas() -> None:
    test_create_canvas()
    test_write_pixel()
    test_construct_ppm_header()
    test_construct_ppm_pixel_data()
