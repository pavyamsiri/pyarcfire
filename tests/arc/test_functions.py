"""Test functions in the `arc.functions` module."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pyarcfire.arc.functions import log_spiral


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
    theta: np.ndarray[tuple[int], np.dtype[np.float64]],
    offset: float,
    growth_factor: float,
    initial_radius: float,
    *,
    use_modulo: bool,
) -> None:
    # Tolerance for small values
    atol = 1e-9

    # Generate the output
    result = log_spiral(theta, offset, growth_factor, initial_radius, use_modulo=use_modulo)

    # Check results
    # Case 1: The initial radius is very close to zero -> result is approximately zero as well
    if abs(initial_radius) < atol:
        assert np.all(np.abs(result) < atol)
    # Case 2: The initial radius is not close to zero -> result has same sign as initial radius
    else:
        expected_sign = np.sign(initial_radius)
        assert np.all(np.sign(result[result != 0]) == expected_sign)
