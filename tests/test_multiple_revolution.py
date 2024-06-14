"""Tests for the arc module, specifically regarding how multiple revolution spirals are fitted."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies._internal.core import DrawFn
from numpy import testing as nptesting
from numpy.typing import NDArray

from pyarcfire.arc.fit import _find_single_revolution_regions_polar


@st.composite
def valid_polar_images(
    draw: DrawFn,
    min_size: int,
    max_size: int,
    shrink_amount: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], int]:
    """Generate a valid test polar image.

    A polar image is an image transformed to be in the polar coordinate frame.

    Parameters
    ----------
    draw : DrawFn
        The draw function.
    min_size : int
        The minimum size of the image.
    max_size : int
        The maximum size of the image.
    shrink_amount: int
        The amount to shrink the cluster by in polar angle bin units.

    Returns
    -------
    arr : NDArray[np.bool_]
        The polar image as a binary array where `True` means it is part of the cluster.
    expected : NDArray[np.bool_]
        The polar angle bins where the cluster does only a single revolution.
    shrink_amount: int
        The amount to shrink the cluster by in polar angle bin units.

    """
    size = draw(st.integers(min_size, max_size))
    double_region_size: int = draw(st.integers(0, size))
    single_region_size: int = draw(st.integers(shrink_amount, size))
    shift_amount: int = draw(st.integers(0, size - 1))

    arr: NDArray[np.bool_] = np.zeros((4, size), dtype=np.bool_)
    arr[0, :double_region_size] = True
    arr[2, :single_region_size] = True
    arr = np.roll(arr, shift_amount, axis=1)

    expected = np.logical_xor(arr[2, :], arr[0, :])
    expected = np.logical_and(
        expected,
        np.logical_and(
            np.roll(expected, shrink_amount),
            np.roll(expected, -shrink_amount),
        ),
    )

    return arr, expected, shrink_amount


@given(valid_polar_images(min_size=5, max_size=360, shrink_amount=5))
def test_find_single_revolution_regions_polar(
    data: tuple[NDArray[np.bool_], NDArray[np.bool_], int],
) -> None:
    """Test the function which identifies single revolution polar angle bins.

    Parameters
    ----------
    data : tuple[NDArray[np.bool_], NDArray[np.bool_], int]
        This tuple is comprised of three components.
    arr : NDArray[np.bool_]
        The polar image as a binary array where `True` means it is part of the cluster.
    expected : NDArray[np.bool_]
        The polar angle bins where the cluster does only a single revolution.
    shrink_amount: int
        The amount to shrink the cluster by in polar angle bin units.

    """
    arr, expected, shrink_amount = data
    single_revolution_array = _find_single_revolution_regions_polar(arr, shrink_amount)
    nptesting.assert_array_equal(expected, single_revolution_array)
