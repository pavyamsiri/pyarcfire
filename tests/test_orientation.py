"""Tests for the orientation module."""

from typing import TypeAlias, TypeVar

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pyarcfire import orientation

_SCT = TypeVar("_SCT", bound=np.generic)
_Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_SCT]]
_Array2D_f32: TypeAlias = _Array2D[np.float32]


def valid_images(
    levels: int,
    min_multiple: int,
    max_multiple: int,
) -> st.SearchStrategy[_Array2D_f32]:
    """Generate a valid test image.

    Parameters
    ----------
    levels : int
        The number of orientation field levels.
    min_multiple : int
        The minimum multiple of 2^levels that an image size can be.
    max_multiple: int
        The maximum multiple of 2^levels that an image size can be.

    Returns
    -------
    valid_image : st.SearchStrategy[Array2D[f32]]
        The generated image.

    """
    factor: int = 2**levels
    elements = st.floats(
        width=32,
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
    return st.integers(min_multiple, max_multiple).flatmap(
        lambda n: arrays(
            dtype=np.float32,
            shape=(n * factor, n * factor),
            elements=elements,
        ),
    )


@given(valid_images(levels=3, min_multiple=2, max_multiple=16))
def test_generation(arr: _Array2D_f32) -> None:
    """Test the generate orientation fields function.

    Parameters
    ----------
    arr : Array2D[f32]
        A valid test image.

    """
    field = orientation.generate_orientation_fields(
        arr,
        num_orientation_field_levels=3,
        neighbour_distance=5,
        kernel_radius=5,
    )
    assert field.shape[0] == arr.shape[0]
    assert field.shape[1] == arr.shape[1]
