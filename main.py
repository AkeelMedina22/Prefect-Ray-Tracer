from prefect import task, flow, get_run_logger
from multiprocessing import cpu_count

from tests.test_tuple import test_tuple

NUM_CPUS = cpu_count()


@flow(name="Run Tests")
def run_test() -> None:
    test_tuple()


@flow(name="Ray Tracing Flow")
def run_ray_tracer() -> None:
    logger = get_run_logger()
    
    logger.info("Running automated tests.")
    run_test()
    logger.info("Automated tests complete.")


if __name__ == "__main__":
    run_ray_tracer()