"""This module contains useful utilities for debugging or profiling."""

# Internal libraries
from functools import wraps
import logging
import time
from typing import Callable, Sequence, Union

# External libraries
from matplotlib import pyplot as plt

# Internal libraries
from .definitions import Array2D


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
        log.debug(
            f"[blue underline]{func.__qualname__}[/blue underline] took {time_duration:.3f} seconds"
        )
        return result

    return wrapper


def debug_plot_image(image: Union[Array2D, Sequence[Array2D]]) -> None:
    is_sequence = isinstance(image, list) or isinstance(image, tuple)
    fig = plt.figure()
    axis = fig.add_subplot(111)
    if is_sequence:
        axis.imshow(image[0])
        for current_image in image[:1]:
            axis.imshow(current_image, alpha=0.5)
    else:
        axis.imshow(image)
    plt.show()
    plt.close()
