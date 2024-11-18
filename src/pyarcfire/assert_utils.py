"""Functions to verify inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


def verify_data_is_normalized(data: NDArray[Any]) -> None:
    """Verify that the given data is normalized to the range [0, 1].

    Parameters
    ----------
    data : NDArray[Any]
        The data to verify.

    """
    not_normalized = np.min(data) < 0 or np.max(data) > 1
    if not_normalized:
        msg = "The data is not normalized to the range [0, 1]! Perhaps use the `preprocess_image`?"
        raise ValueError(msg)


def verify_data_is_2d(data: NDArray[Any]) -> None:
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
    data: NDArray[Any],
    *,
    num_orientation_field_levels: int,
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
    maximum_shrink_factor: int = 2**num_orientation_field_levels
    if data.shape[0] % maximum_shrink_factor != 0 or data.shape[1] % maximum_shrink_factor != 0:
        msg = f"""Image dimensions must be divisible by 2^{num_orientation_field_levels} = {maximum_shrink_factor}.
        Perhaps use the `preprocess_image`?
        """
        raise ValueError(msg)
