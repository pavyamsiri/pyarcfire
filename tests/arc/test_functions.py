"""Test functions in the `arc.functions` module."""

import numpy as np

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
