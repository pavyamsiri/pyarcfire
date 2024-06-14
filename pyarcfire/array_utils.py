"""Utilities related to dealing with NDArrays."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

ScalarType_co = TypeVar("ScalarType_co", bound=np.generic, covariant=True)


def get_origin_points(image: NDArray[ScalarType_co]) -> Sequence[tuple[int, int]]:
    """Return points that are either the centre of the image or touching the centre.

    Parameters
    ----------
    image : NDArray[ScalarType_co]
        The image.

    Returns
    -------
    Sequence[tuple[int, int]]
        A sequence of points written in index form that are in the centre.

    Notes
    -----
    Dimensions with an odd size have a clear central cell so we can return a single index,
    however even dimensions have their centre between two cells. In this case we return
    the indices of the two adjacent cells.

    """
    # Assume that dimensions are even
    is_height_even = image.shape[0] % 2 == 0
    is_width_even = image.shape[1] % 2 == 0

    central_indices: Sequence[tuple[int, int]]
    # Even, Even
    if is_height_even and is_width_even:
        bottom = image.shape[0] // 2
        right = image.shape[1] // 2
        central_indices = (
            # Top left
            (bottom - 1, right - 1),
            # Top right
            (bottom - 1, right),
            # Bottom left
            (bottom, right - 1),
            # Bottom right
            (bottom, right),
        )
    # Even, Odd
    elif is_height_even and not is_width_even:
        bottom = image.shape[0] // 2
        width_centre = image.shape[1] // 2
        central_indices = (
            (bottom - 1, width_centre),
            (bottom, width_centre),
        )
    # Odd, Even
    elif not is_height_even and is_width_even:
        height_centre = image.shape[0] // 2
        right = image.shape[1] // 2
        central_indices = (
            (height_centre, right - 1),
            (height_centre, right),
        )
    # Odd, Odd
    else:
        central_indices = ((image.shape[0] // 2, image.shape[1] // 2),)

    return central_indices


def get_origin_points_unnested(
    image: NDArray[ScalarType_co],
) -> tuple[Sequence[int], Sequence[int]]:
    """Return points that are either the centre of the image or touching the centre.

    This is the same get_origin_points but returns the points as two separate sequences
    of indices instead of a single sequence of tuples of indices.

    Parameters
    ----------
    image : NDArray[ScalarType_co]
        The image.

    Returns
    -------
    row_indices : Sequence[int]
        The row indices of the central points.
    column_indices : Sequence[int]
        The column indices of the central points.

    Notes
    -----
    Dimensions with an odd size have a clear central cell so we can return a single index,
    however even dimensions have their centre between two cells. In this case we return
    the indices of the two adjacent cells.

    """
    points = get_origin_points(image)
    row_indices = [row_idx for row_idx, _ in points]
    column_indices = [column_idx for _, column_idx in points]
    return (row_indices, column_indices)
