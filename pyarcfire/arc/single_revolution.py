# Standard libraries
import logging

# External libraries
import numpy as np
from scipy import optimize

# Internal libraries
from pyarcfire.definitions import ImageFloatArray
from .common import LogSpiralFitResult
from .functions import (
    calculate_best_initial_radius,
    calculate_log_spiral_error,
    calculate_log_spiral_error_from_pitch_angle,
)
from .utils import (
    _adjust_theta_to_zero,
    _calculate_bounds,
    _get_polar_coordinates,
    _get_arc_bounds,
)

log = logging.getLogger(__name__)


def fit_spiral_to_image_single_revolution(
    image: ImageFloatArray,
    initial_pitch_angle: float = 0,
) -> LogSpiralFitResult:
    # Convert to polar coordinates
    radii, theta, weights = _get_polar_coordinates(image)

    # Find suitable bounds for the offset parameter
    bad_bounds, (lower_bound, upper_bound), rotation_amount, max_gap_size = (
        _calculate_bounds(theta)
    )
    theta = (theta - rotation_amount) % (2 * np.pi)

    # Perform a fit to get the pitch angle
    if bad_bounds:
        log.warn(f"Bad bounds! Gap size = {max_gap_size}")
        offset = 0
        pitch_angle = 0
    else:
        offset = (lower_bound + upper_bound) / 2
        res = optimize.least_squares(
            calculate_log_spiral_error_from_pitch_angle,
            x0=initial_pitch_angle,
            args=(radii, theta, weights, offset),
        )
        assert res.success, "Failed to fit pitch angle"
        pitch_angle = res.x[0]

    # Calculate the error from the fit
    initial_radius = calculate_best_initial_radius(
        radii, theta, weights, offset, pitch_angle
    )
    error, _ = calculate_log_spiral_error(
        radii, theta, weights, offset, pitch_angle, initial_radius
    )

    # Rotate back
    theta = (theta + rotation_amount) % (2 * np.pi)
    offset += rotation_amount

    # Get arc bounds
    arc_bounds = _get_arc_bounds(offset, rotation_amount, lower_bound, upper_bound)

    # Adjust so that arc bounds is relative to theta
    (theta, arc_bounds, offset) = _adjust_theta_to_zero(theta, arc_bounds, offset)

    # Recalculate initial radius and error after adjustment
    initial_radius = calculate_best_initial_radius(
        radii, theta, weights, offset, pitch_angle
    )
    new_error, residuals = calculate_log_spiral_error(
        radii, theta, weights, offset, pitch_angle, initial_radius
    )

    # Ensure consistency
    square_err_difference_per_pixel = abs(new_error - error) / len(theta)
    assert np.isclose(
        square_err_difference_per_pixel, 0
    ), f"Inconsistent fit when eliminating theta offset; difference = {square_err_difference_per_pixel}"

    result = LogSpiralFitResult(
        offset=offset,
        pitch_angle=pitch_angle,
        initial_radius=initial_radius,
        arc_bounds=arc_bounds,
        total_error=new_error,
        errors=np.square(residuals),
    )

    return result
