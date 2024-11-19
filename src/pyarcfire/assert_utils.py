"""Functions to verify inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    import optype as op


_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any])
_Shape = TypeVar("_Shape", bound=tuple[int, ...])
_Array2D = np.ndarray[tuple[int, int], np.dtype[_SCT]]
_ArrayND = np.ndarray[_Shape, np.dtype[_SCT]]


def verify_data_is_normalized(data: _ArrayND[_Shape, _SCT_f]) -> None:
    """Verify that the given data is normalized to the range [0, 1].

    Parameters
    ----------
    data : NDArray[Any]
        The data to verify.

    """
    not_normalized = np.min(data) < 0 or np.max(data) > 1
    if not_normalized:
        msg = "The data is not normalized to the range [0, 1]! Perhaps use `preprocess_image`?"
        raise ValueError(msg)


def verify_data_is_2d(data: _ArrayND[_Shape, _SCT]) -> None:
    """Verify that the given data is 2D.

    Parameters
    ----------
    data : NDArray[Any]
        The data to verify.

    """
    is_not_2d = len(data.shape) != 2
    if is_not_2d:
        msg = "The data is not 2D! This function requires a 2D array."
        raise ValueError(msg)


def verify_data_can_be_shrunk_orientation(
    data: _Array2D[_SCT],
    *,
    num_orientation_field_levels: op.CanInt,
) -> None:
    """Verify that the given data is 2D.

    Parameters
    ----------
    data : NDArray[Any]
        The data to verify.
    num_orientation_field_levels : int
        The number of orientation field levels.

    """
    # The dimensions of the image must be divisible by the largest shrink factor
    maximum_shrink_factor: int = 2 ** int(num_orientation_field_levels)
    if data.shape[0] % maximum_shrink_factor != 0 or data.shape[1] % maximum_shrink_factor != 0:
        msg = f"""Image dimensions must be divisible by 2^{num_orientation_field_levels} = {maximum_shrink_factor}.
        Perhaps use the `preprocess_image`?
        """
        raise ValueError(msg)
