"""Useful utilities for debugging or profiling."""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from typing_extensions import ParamSpec

log: logging.Logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def benchmark(func: Callable[P, R]) -> Callable[P, R]:
    """Time functions.

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
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        time_start = time.perf_counter()
        result = func(*args, **kwargs)
        time_end = time.perf_counter()
        time_duration = time_end - time_start
        log.info(
            "[magenta]PROFILER[/magenta]: [blue underline]%s[/blue underline] took %.3f seconds",
            func.__qualname__,
            time_duration,
        )
        return result

    return wrapper
