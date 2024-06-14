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
        msg = "The data is not normalized to the range [0, 1]! Please normalize your data before using this function."
        raise ValueError(msg)
