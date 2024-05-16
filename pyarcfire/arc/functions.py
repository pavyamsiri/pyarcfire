# Standard libraries

# External libraries
import numpy as np

# Internal libraries
from pyarcfire.definitions import FloatArray1D


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
    result = np.sqrt(weights) * (
        radii - log_spiral(theta, offset, pitch_angle, initial_radius)
    )
    return result


def calculate_log_spiral_error(
    radii: FloatArray1D,
    theta: FloatArray1D,
    weights: FloatArray1D,
    offset: float,
    pitch_angle: float,
    initial_radius: float,
) -> tuple[float, FloatArray1D]:
    residuals = calculate_log_spiral_residual_vector(
        radii, theta, weights, offset, pitch_angle, initial_radius
    )
    sum_square_error = np.sum(np.square(residuals))
    return (sum_square_error, residuals)


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
