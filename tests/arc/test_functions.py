"""Test functions in the `arc.functions` module."""

from typing import TypeAlias

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pyarcfire.arc.functions import log_spiral

_Array1D_f64: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]


def test_log_spiral_basic_no_modulo() -> None:
    """Simple test of log spiral function with some basic inputs for non-modulo mode."""
    theta = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, 5 * np.pi / 2, 3 * np.pi, 7 * np.pi / 2], dtype=np.float64)
    offset = 0.0
    growth_factor = 0.1
    initial_radius = 10.0
    result = log_spiral(theta, offset, growth_factor, initial_radius, use_modulo=False)
    expected = np.array(
        [10.0, 8.54635999, 7.30402691, 6.24228434, 5.33488091, 4.55938128, 3.89661137, 3.33018435], dtype=np.float64
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_log_spiral_basic_use_modulo() -> None:
    """Simple test of log spiral function with some basic inputs for modulo mode."""
    theta = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, 5 * np.pi / 2, 3 * np.pi, 7 * np.pi / 2], dtype=np.float64)
    offset = 0.0
    growth_factor = 0.1
    initial_radius = 10.0
    result = log_spiral(theta, offset, growth_factor, initial_radius, use_modulo=True)
    expected = np.array([10.0, 8.54635999, 7.30402691, 6.24228434, 10.0, 8.54635999, 7.30402691, 6.24228434], dtype=np.float64)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@given(
    theta=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),  # Array with at least 1 element.
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    ),
    offset=st.floats(min_value=-2 * np.pi, max_value=2 * np.pi),
    growth_factor=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
    initial_radius=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    use_modulo=st.booleans(),
)
def test_log_spiral_sign(
    theta: _Array1D_f64,
    offset: float,
    growth_factor: float,
    initial_radius: float,
    *,
    use_modulo: bool,
) -> None:
    """Test sign property of log spiral function.

    Parameters
    ----------
    theta : Array1D[f64]
        The angles to evaluate the radius at.
    offset : float
        The offset.
    growth_factor : float
        The growth factor.
    initial_radius : float
        The initial radius.
    use_modulo : bool
        Set this flag to keep the angles within 2 pi.

    Notes
    -----
    The sign property is that the sign of the result is the same as the sign of initial radius given that
    the initial radius is sufficiently large.

    """
    # Tolerance for small values
    atol = 1e-9

    # Generate the output
    result = log_spiral(theta, offset, growth_factor, initial_radius, use_modulo=use_modulo)

    # Case 1: The initial radius is very close to zero -> result is approximately zero as well
    if abs(initial_radius) < atol:
        assert np.all(np.abs(result) < 1e2 * atol)
    # Case 2: The initial radius is not close to zero -> result has same sign as initial radius
    else:
        expected_sign = np.sign(initial_radius)
        assert np.all(np.sign(result[result != 0]) == expected_sign)


@given(
    theta=arrays(
        dtype=np.float64,
        shape=st.integers(1, 100),
        elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    ),
    offset=st.floats(-2 * np.pi, 2 * np.pi),
    growth_factor=st.floats(0, 10, allow_nan=False, allow_infinity=False),
    initial_radius=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    use_modulo=st.booleans(),
)
def test_log_spiral_dimension_consistency(
    theta: _Array1D_f64, offset: float, growth_factor: float, initial_radius: float, *, use_modulo: bool
) -> None:
    """Test that the resulting shape of log spiral is the same as the input shape.

    Also that the dtypes are the same.

    Parameters
    ----------
    theta : Array1D[f64]
        The angles to evaluate the radius at.
    offset : float
        The offset.
    growth_factor : float
        The growth factor.
    initial_radius : float
        The initial radius.
    use_modulo : bool
        Set this flag to keep the angles within 2 pi.

    """
    result = log_spiral(theta, offset, growth_factor, initial_radius, use_modulo=use_modulo)
    assert result.shape == theta.shape
    assert result.dtype == theta.dtype
