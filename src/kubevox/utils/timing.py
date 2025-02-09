import time
from contextlib import contextmanager

from loguru import logger


@contextmanager
def timing(operation: str):
    """Context manager for timing operations and logging their duration."""
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logger.info(f"{operation}: {elapsed_time:.3f}s")
