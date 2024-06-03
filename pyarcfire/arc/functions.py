from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

FloatType = TypeVar("FloatType", np.float32, np.float64)


def log_spiral(
    theta: NDArray[FloatType],
    offset: float,
    pitch_angle: float,
    initial_radius: float,
    use_modulo: bool,
) -> NDArray[FloatType]:
    angles = theta - offset
    if use_modulo:
        angles %= 2 * np.pi
    result: NDArray[FloatType] = np.multiply(
        initial_radius, np.exp(np.multiply(-pitch_angle, angles))
    )
    return result


def calculate_log_spiral_residual_vector(
    radii: NDArray[FloatType],
    theta: NDArray[FloatType],
    weights: NDArray[FloatType],
    offset: float,
    pitch_angle: float,
    initial_radius: float,
    use_modulo: bool,
) -> NDArray[FloatType]:
    result = np.multiply(
        np.sqrt(weights),
        (radii - log_spiral(theta, offset, pitch_angle, initial_radius, use_modulo)),
    )
    return result


def calculate_log_spiral_error(
    radii: NDArray[FloatType],
    theta: NDArray[FloatType],
    weights: NDArray[FloatType],
    offset: float,
    pitch_angle: float,
    initial_radius: float,
    use_modulo: bool,
) -> tuple[float, NDArray[FloatType]]:
    residuals = calculate_log_spiral_residual_vector(
        radii, theta, weights, offset, pitch_angle, initial_radius, use_modulo
    )
    sum_square_error = np.sum(np.square(residuals))
    return (sum_square_error, residuals)


def calculate_log_spiral_error_from_pitch_angle(
    pitch_angle: float,
    radii: NDArray[FloatType],
    theta: NDArray[FloatType],
    weights: NDArray[FloatType],
    offset: float,
    use_modulo: bool,
) -> NDArray[FloatType]:
    initial_radius = calculate_best_initial_radius(
        radii, theta, weights, offset, pitch_angle, use_modulo
    )
    residuals = calculate_log_spiral_residual_vector(
        radii, theta, weights, offset, pitch_angle, initial_radius, use_modulo
    )
    return residuals


def calculate_best_initial_radius(
    radii: NDArray[FloatType],
    theta: NDArray[FloatType],
    weights: NDArray[FloatType],
    offset: float,
    pitch_angle: float,
    use_modulo: bool,
) -> float:
    log_spiral_term = log_spiral(theta, offset, pitch_angle, 1, use_modulo)
    result = float(
        np.sum(radii * weights * log_spiral_term)
        / np.sum(weights * np.square(log_spiral_term))
    )
    return result
