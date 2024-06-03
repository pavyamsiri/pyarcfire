"""This module contains useful utilities for debugging or profiling."""

# Internal libraries
from functools import wraps
import logging
import time
from typing import Callable


log: logging.Logger = logging.getLogger(__name__)


def benchmark(func: Callable) -> Callable:
    """Decorator used to time functions.

    Parameters
    ----------
    func : Callable
        The function to time.

    Returns
    -------
    Callable
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        time_start = time.perf_counter()
        result = func(*args, **kwargs)
        time_end = time.perf_counter()
        time_duration = time_end - time_start
        log.info(
            f"[magenta]PROFILER[/magenta]: [blue underline]{func.__qualname__}[/blue underline] took {time_duration:.3f} seconds"
        )
        return result

    return wrapper
