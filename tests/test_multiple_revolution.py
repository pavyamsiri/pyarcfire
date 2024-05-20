# Standard libraries

# External libraries
import numpy as np
from numpy import testing as nptesting

# Testing libraries
from hypothesis import given
from hypothesis import strategies as st

# Testing
from pyarcfire.arc.multiple_revolution import _find_single_revolution_regions_polar
from pyarcfire.definitions import BoolArray1D, ImageBoolArray


@st.composite
def valid_polar_images(
    draw,
    min_size: int,
    max_size: int,
    shrink_amount: int,
) -> tuple[ImageBoolArray, BoolArray1D, int]:
    size: int = draw(st.integers(min_size, max_size))
    double_region_size: int = draw(st.integers(0, size))
    single_region_size: int = draw(st.integers(shrink_amount, size))
    shift_amount: int = draw(st.integers(0, size - 1))

    arr = np.zeros((4, size), dtype=np.bool_)
    arr[0, :double_region_size] = True
    arr[2, :single_region_size] = True
    arr = np.roll(arr, shift_amount, axis=1)

    expected = np.logical_xor(arr[2, :], arr[0, :])
    expected = np.logical_and(
        expected,
        np.logical_and(
            np.roll(expected, shrink_amount), np.roll(expected, -shrink_amount)
        ),
    )

    return arr, expected, shrink_amount


@given(valid_polar_images(min_size=5, max_size=360, shrink_amount=5))
def test_find_single_revolution_regions_polar(
    data: tuple[ImageBoolArray, BoolArray1D, int],
) -> None:
    arr, expected, shrink_amount = data
    single_revolution_array = _find_single_revolution_regions_polar(arr, shrink_amount)
    nptesting.assert_array_equal(expected, single_revolution_array)
