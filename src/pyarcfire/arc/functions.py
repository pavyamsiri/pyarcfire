"""Functions that calculate log spirals and their residuals."""

from typing import Any, TypeAlias, TypeVar

import numpy as np
import optype as op

from pyarcfire._typing import AnyReal

_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any])
_Shape = TypeVar("_Shape", bound=tuple[int, ...])
_ArrayND: TypeAlias = np.ndarray[_Shape, np.dtype[_SCT]]


def log_spiral(
    theta: _ArrayND[_Shape, _SCT_f], offset: AnyReal, growth_factor: AnyReal, initial_radius: AnyReal, *, use_modulo: op.CanBool
) -> _ArrayND[_Shape, _SCT_f]:
    """Calculate the radius of a log spiral given parameters and theta.

    Parameters
    ----------
    theta : ArrayND[S, F]
        The polar angle of the log spiral in radians.
    offset : float
        The offset angle in radians.
    growth_factor : float
        The growth factor.
    initial_radius : float
        The initial radius in pixels.
    use_modulo : bool
        Set this flag to apply the modulo operator to the angles before computing the radii.

    Returns
    -------
    log_spiral : ArrayND[S, F]
        The radius value of the log spiral.

    """
    angles = theta - offset
    if use_modulo:
        angles %= 2 * np.pi
    return (initial_radius * np.exp(-np.asarray(growth_factor).astype(angles.dtype) * angles)).astype(theta.dtype)


def calculate_log_spiral_residual_vector(
    radii: _ArrayND[_Shape, _SCT_f],
    theta: _ArrayND[_Shape, _SCT_f],
    weights: _ArrayND[_Shape, _SCT_f],
    offset: AnyReal,
    growth_factor: AnyReal,
    initial_radius: AnyReal,
    *,
    use_modulo: op.CanBool,
) -> _ArrayND[_Shape, _SCT_f]:
    """Calculate the residuals of a log spiral with respect to a cluster.

    Parameters
    ----------
    radii : ArrayND[S, F]
        The polar radii of the cluster's pixels in pixels.
    theta : ArrayND[S, F]
        The polar angle of the cluster's pixels in radians.
    weights : ArrayND[S, F]
        The weights of the cluster's pixels in pixels.
    offset : float
        The offset angle in radians.
    growth_factor : float
        The growth factor.
    initial_radius : float
        The initial radius in pixels.
    use_modulo : bool
        Set this flag to apply the modulo operator to the angles before computing the radii.

    Returns
    -------
    residuals : ArrayND[S, F]
        The residual associated with each pixel in the cluster.

    """
    return (np.sqrt(weights) * (radii - log_spiral(theta, offset, growth_factor, initial_radius, use_modulo=use_modulo))).astype(
        radii.dtype
    )


def calculate_log_spiral_error(
    radii: _ArrayND[_Shape, _SCT_f],
    theta: _ArrayND[_Shape, _SCT_f],
    weights: _ArrayND[_Shape, _SCT_f],
    offset: AnyReal,
    growth_factor: AnyReal,
    initial_radius: AnyReal,
    *,
    use_modulo: op.CanBool,
) -> tuple[float, _ArrayND[_Shape, _SCT_f]]:
    """Calculate the sum of square residuals of a log spiral with respect to a cluster.

    Parameters
    ----------
    radii : ArrayND[S, F]
        The polar radii of the cluster's pixels in pixels.
    theta : ArrayND[S, F]
        The polar angle of the cluster's pixels in radians.
    weights : ArrayND[S, F]
        The weights of the cluster's pixels in pixels.
    offset : float
        The offset angle in radians.
    growth_factor : float
        The growth factor.
    initial_radius : float
        The initial radius in pixels.
    use_modulo : bool
        Set this flag to apply the modulo operator to the angles before computing the radii.

    Returns
    -------
    sse : float
        The sum of square errors.
    residuals : ArrayND[S, F]
        The residual associated with each pixel in the cluster.

    """
    residuals = calculate_log_spiral_residual_vector(
        radii,
        theta,
        weights,
        offset,
        growth_factor,
        initial_radius,
        use_modulo=use_modulo,
    )
    sum_square_error = np.sum(np.square(residuals))
    return (sum_square_error, residuals)


def calculate_log_spiral_error_from_growth_factor(
    growth_factor: AnyReal,
    radii: _ArrayND[_Shape, _SCT_f],
    theta: _ArrayND[_Shape, _SCT_f],
    weights: _ArrayND[_Shape, _SCT_f],
    offset: AnyReal,
    *,
    use_modulo: op.CanBool,
) -> _ArrayND[_Shape, _SCT_f]:
    """Return the residuals of a log spiral fit to the given cluster.

    This function automatically determines the optimal initial radius given an offset and the growth factor.

    Parameters
    ----------
    growth_factor : float
        The growth factor.
    radii : ArrayND[S, F]
        The polar radii of the cluster's pixels in pixels.
    theta : ArrayND[S, F]
        The polar angle of the cluster's pixels in radians.
    weights : ArrayND[S, F]
        The weights of the cluster's pixels in pixels.
    offset : float
        The offset angle in radians.
    initial_radius : float
        The initial radius in pixels.
    use_modulo : bool
        Set this flag to apply the modulo operator to the angles before computing the radii.

    Returns
    -------
    residuals : ArrayND[S, F]
        The residual associated with each pixel in the cluster.

    """
    initial_radius = calculate_best_initial_radius(radii, theta, weights, offset, growth_factor, use_modulo=use_modulo)
    return calculate_log_spiral_residual_vector(
        radii,
        theta,
        weights,
        offset,
        growth_factor,
        initial_radius,
        use_modulo=use_modulo,
    )


def calculate_best_initial_radius(
    radii: _ArrayND[_Shape, _SCT_f],
    theta: _ArrayND[_Shape, _SCT_f],
    weights: _ArrayND[_Shape, _SCT_f],
    offset: AnyReal,
    growth_factor: AnyReal,
    *,
    use_modulo: op.CanBool,
) -> float:
    """Determine the most optimal initial radius given a growth factor and offset.

    This function automatically determines the optimal initial radius given an offset and growth factor.

    Parameters
    ----------
    radii : ArrayND[S, F]
        The polar radii of the cluster's pixels in pixels.
    theta : ArrayND[S, F]
        The polar angle of the cluster's pixels in radians.
    weights : ArrayND[S, F]
        The weights of the cluster's pixels in pixels.
    offset : float
        The offset angle in radians.
    growth_factor : float
        The growth factor.
    use_modulo : bool
        Set this flag to apply the modulo operator to the angles before computing the radii.

    Returns
    -------
    float
        The optimal initial radius.

    """
    log_spiral_term = log_spiral(theta, offset, growth_factor, 1, use_modulo=use_modulo)
    return float(np.sum(radii * weights * log_spiral_term) / np.sum(weights * np.square(log_spiral_term)))
