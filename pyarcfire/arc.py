# Standard libraries
from dataclasses import dataclass
import logging

# External libraries
import numpy as np
from scipy import optimize

# Internal libraries
from .definitions import FloatArray1D, ImageArray

log = logging.getLogger(__name__)


@dataclass
class LogSpiralFitResult:
    offset: float
    pitch_angle: float
    initial_radius: float
    arc_bounds: tuple[float, float]
    error: float


def fit_spiral_to_image(
    image: ImageArray,
) -> LogSpiralFitResult:
    row_indices, column_indices = image.nonzero()
    # Subpixel centering
    row_offset = image.shape[0] / 2 + 0.5
    column_offset = image.shape[1] / 2 + 0.5
    x = column_indices - column_offset
    y = -(row_indices - row_offset)
    radii = np.sqrt(np.square(x) + np.square(y))
    theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
    weights = image[row_indices, column_indices]

    (lower_bound, upper_bound), rotation_amount, max_gap_size = _calculate_bounds(theta)
    theta = (theta - rotation_amount) % (2 * np.pi)
    if max_gap_size <= 0.1:
        log.warn(f"Bad bounds! Gap size = {max_gap_size}")
        initial_offset = 0
        pitch_angle = 0
    else:
        initial_offset = (lower_bound + upper_bound) / 2
        res = optimize.least_squares(
            calculate_log_spiral_error_from_pitch_angle,
            jac=calculate_log_spiral_jacobian_from_pitch_angle,  # type:ignore
            x0=0,
            args=(radii, theta, weights, initial_offset),
        )
        assert res.success, "Failed to fit pitch angle"
        pitch_angle = res.x[0]
    initial_radius = calculate_best_initial_radius(
        radii, theta, weights, initial_offset, pitch_angle
    )
    residuals = calculate_log_spiral_residual_vector(
        radii, theta, weights, initial_offset, pitch_angle, initial_radius
    )
    square_residuals = np.square(residuals)
    error = square_residuals.sum()
    # Rotate back
    theta = (theta + rotation_amount) % (2 * np.pi)
    initial_offset += rotation_amount
    offset = initial_offset
    arc_size = 2 * np.pi - (upper_bound - lower_bound)
    arc_start = np.min(
        (
            _calculate_angle_distance(
                np.array([initial_offset, initial_offset]),
                np.array([lower_bound, upper_bound]),
            )
            + rotation_amount
        )
        % (2 * np.pi)
    )
    arc_bounds = (float(arc_start), float(arc_start) + arc_size)
    theta = (theta - initial_offset) % (2 * np.pi) + initial_offset
    offset_shift = arc_bounds[0]
    new_offset = initial_offset + offset_shift
    theta = theta + new_offset - np.min(theta)
    initial_radius = calculate_best_initial_radius(
        radii, theta, weights, new_offset, pitch_angle
    )
    offset = new_offset
    arc_bounds = (
        arc_bounds[0] - offset_shift + offset,
        arc_bounds[1] - offset_shift + offset,
    )
    residuals = calculate_log_spiral_residual_vector(
        radii, theta, weights, offset, pitch_angle, initial_radius
    )
    error = np.sum(np.square(residuals))

    error_diagnostic = np.abs(np.sum(residuals) - error) / len(residuals)
    # assert error_diagnostic < 1e-4, "Total error is too high!"

    result = LogSpiralFitResult(
        offset=offset,
        pitch_angle=pitch_angle,
        initial_radius=initial_radius,
        arc_bounds=arc_bounds,
        error=error,
    )

    return result


def calculate_log_spiral_error_from_pitch_angle(
    pitch_angle: float,
    radii: FloatArray1D,
    theta: FloatArray1D,
    weights: FloatArray1D,
    offset: float,
) -> FloatArray1D:
    initial_radius = calculate_best_initial_radius(
        radii, theta, weights, offset, pitch_angle
    )
    residuals = calculate_log_spiral_residual_vector(
        radii, theta, weights, offset, pitch_angle, initial_radius
    )
    return residuals


def calculate_log_spiral_jacobian_from_pitch_angle(
    pitch_angle: float,
    radii: FloatArray1D,
    theta: FloatArray1D,
    weights: FloatArray1D,
    offset: float,
) -> FloatArray1D:
    initial_radius = calculate_best_initial_radius(
        radii, theta, weights, offset, pitch_angle
    )
    angles = (theta - offset) % (2 * np.pi)
    # NOTE: Not sure where this comes from. The partial derivative would not have sqrt(weights)
    # it would be -angles * log_spiral(...)
    jac = -angles * log_spiral(theta, offset, pitch_angle, initial_radius)
    from_matlab = True
    if from_matlab:
        jac *= -np.sqrt(weights)
    jac = jac.reshape((len(jac), 1))
    return jac


def log_spiral[T: (float, FloatArray1D)](
    theta: T, offset: float, pitch_angle: float, initial_radius: float
) -> T:
    angles = (theta - offset) % (2 * np.pi)
    result: T = initial_radius * np.exp(-pitch_angle * angles)  # type:ignore
    return result


def calculate_log_spiral_residual_vector(
    radii: FloatArray1D,
    theta: FloatArray1D,
    weights: FloatArray1D,
    offset: float,
    pitch_angle: float,
    initial_radius: float,
) -> FloatArray1D:
    result = (
        np.sqrt(weights)
        * (radii - log_spiral(theta, offset, pitch_angle, initial_radius))
        / weights.sum()
    )
    return result


def calculate_best_initial_radius(
    radii: FloatArray1D,
    theta: FloatArray1D,
    weights: FloatArray1D,
    offset: float,
    pitch_angle: float,
) -> float:
    log_spiral_term = log_spiral(theta, offset, pitch_angle, 1)
    result = float(
        np.sum(radii * weights * log_spiral_term)
        / np.sum(weights * np.square(log_spiral_term))
    )
    return result


def _calculate_bounds(
    theta: FloatArray1D,
) -> tuple[tuple[float, float], float, float]:
    sorted_theta = np.sort(theta)
    gaps = np.diff(sorted_theta)

    end_gap = sorted_theta[0] + 2 * np.pi - sorted_theta[-1]
    max_gap_size = np.max(gaps)
    # If the cluster crosses the polar axis then this is false
    gap_crosses_axis = end_gap > max_gap_size
    # The optimization function lets us restrict theta-offset values by
    # specifying lower and upper bounds.  If the range of allowable
    # values goes through zero, then this range gets split into two
    # parts, which we can't express with a single pair of bounds.  In
    # this case, we temporarily rotate the points to allievate this
    # problem, fit the log-spiral model to the set of rotated points,
    # and then reverse the rotation on the fitted model.
    if not gap_crosses_axis:
        rotation_amount = 0
        max_gap_size_idx = np.argmax(gaps)
        lower_bound = sorted_theta[max_gap_size_idx]
        upper_bound = sorted_theta[max_gap_size_idx + 1]
    else:
        rotation_amount = sorted_theta[0]
        lower_bound = sorted_theta[-1] - rotation_amount % (2 * np.pi)
        upper_bound = 2 * np.pi
        max_gap_size = upper_bound - lower_bound
    bounds = (lower_bound, upper_bound)
    return (bounds, rotation_amount, max_gap_size)


def _calculate_angle_distance(
    from_angle: FloatArray1D, to_angle: FloatArray1D
) -> FloatArray1D:
    is_wrapping = from_angle > to_angle
    distance = to_angle - from_angle
    distance[is_wrapping] += 2 * np.pi
    return distance
