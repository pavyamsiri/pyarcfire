"""Functions to compute the best fitting log spiral to a cluster of pixels."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np
import skimage
import skimage.measure
from scipy import ndimage, optimize

from pyarcfire.array_utils import get_origin_points

from .common import LogSpiralFitResult
from .functions import (
    calculate_best_initial_radius,
    calculate_log_spiral_error,
    calculate_log_spiral_error_from_growth_factor,
)
from .utils import (
    adjust_theta_to_zero,
    calculate_bounds,
    get_arc_bounds,
    get_polar_coordinates,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

IndexType = TypeVar("IndexType", np.int32, np.int64)
FloatType = TypeVar("FloatType", np.float32, np.float64)
BoolType = np.bool_

SINGLE_REVOLUTION_TOLERANCE: float = 1e-8
MULTIPLE_REVOLUTION_TOLERANCE: float = 1e-12

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class WrapData:
    """Data encoding how a cluster wraps in polar coordinates.

    A wrap is defined as a cluster of pixels which goes past 2 * pi
    somewhere in the middle of itself.

    Attributes
    ----------
    start_length : int
        The length of the cluster in polar angle bin units before it wraps.
    end_length : int
        The length of the cluster in polar angle bin units after it wraps.
    length : int
        The total length of the cluster in polar angle bin units.
    start : int
        The index of the polar angle bin where the cluster starts.
    end : int
        The index of the polar angle bin where the cluster ends.

    Notes
    -----
    The length of the cluster is not exactly correct as there is a degree of shrinking
    to remove cluster pixels on the extremes of the cluster.

    """

    start_length: int
    end_length: int
    length: int
    start: int
    end: int


@dataclass
class InternalLogSpiralFitResult(Generic[FloatType]):
    """The result of a log spiral fit to a cluster.

    This is an internal type only designed to be used in this module.
    For the public interface please see `LogSpiralFitResult`.

    Attributes
    ----------
    theta : NDArray[FloatType]
        The polar angle of the cluster's pixels in radians.
    offset : float
        The offset in radians.
    growth_factor : float
        The growth factor.
    arc_bounds : tuple[float, float]
        The azimuthal bounds of the arc.
    error : float
        The sum of the square residuals.
    bad_bounds : bool
        This is `True` if no suitable polar angle bounds could be found.

    """

    theta: NDArray[FloatType]
    offset: float
    growth_factor: float
    arc_bounds: tuple[float, float]
    error: float
    bad_bounds: bool


def fit_spiral_to_image(
    image: NDArray[FloatType],
    initial_growth_factor: float = 0,
    *,
    force_single_revolution: bool = False,
) -> LogSpiralFitResult[FloatType]:
    """Fits a single log spiral to the given cluster encoded as an image of non-zero pixels.

    Parameters
    ----------
    image : NDArray[FloatType]
        The cluster encoded as an array of non-zero pixels.
    initial_growth_factor : float
        The initial guess of the growth factor k.
    force_single_revolution : bool
        Set this flag if you only want single revolution solutions.

    Returns
    -------
    LogSpiralFitResult[FloatType]
        The result of the fit.

    """
    # Convert to polar coordinates
    radii, theta, weights = get_polar_coordinates(image)

    # Check if the cluster revolves more than once
    bad_bounds, _, _, _ = calculate_bounds(theta)

    inner_region: NDArray[BoolType] | None = None

    # Gap in theta is not large enough to not need multiple revolutions
    # and the cluster does not contain the origin or is not closed around centre
    if not force_single_revolution and bad_bounds or not __cluster_has_no_endpoints_or_contains_origin(image):
        inner_region = identify_inner_and_outer_spiral(image, shrink_amount=5)
        if inner_region is None or inner_region.sum() == 0 or inner_region.sum() == len(inner_region):
            inner_region = None
        else:
            theta = _remove_theta_discontinuities(theta, image, inner_region)
    need_multiple_revolutions: bool = inner_region is not None
    if inner_region is not None:
        fit_result = _fit_spiral_to_image_multiple_revolution_core(
            radii,
            theta,
            weights,
            initial_growth_factor,
            inner_region,
        )
    else:
        fit_result = _fit_spiral_to_image_single_revolution_core(
            radii,
            theta,
            weights,
            initial_growth_factor,
        )

    arc_bounds = fit_result.arc_bounds
    offset = fit_result.offset
    growth_factor = fit_result.growth_factor
    error = fit_result.error
    theta = fit_result.theta
    bad_bounds = fit_result.bad_bounds

    # Adjust so that arc bounds is relative to theta
    (theta, arc_extent, offset) = adjust_theta_to_zero(
        theta,
        arc_bounds,
        offset,
        use_modulo=not need_multiple_revolutions,
    )

    growth_factor = growth_factor if not bad_bounds else 0

    # Recalculate initial radius and error after adjustment
    initial_radius = calculate_best_initial_radius(
        radii,
        theta,
        weights,
        offset,
        growth_factor,
        use_modulo=not need_multiple_revolutions,
    )
    new_error, residuals = calculate_log_spiral_error(
        radii,
        theta,
        weights,
        offset,
        growth_factor,
        initial_radius,
        use_modulo=not need_multiple_revolutions,
    )

    # Ensure consistency
    square_err_difference_per_pixel = abs(new_error - error) / len(theta)
    inconsistent_fit_after_adjustment = not np.isclose(
        square_err_difference_per_pixel,
        0,
    )
    if inconsistent_fit_after_adjustment:
        log.debug(
            "[red]SUBOPTIM[/red]: Inconsistent fit when eliminating theta offset. Difference = %f",
            square_err_difference_per_pixel,
        )

    return LogSpiralFitResult(
        offset=offset,
        growth_factor=growth_factor,
        initial_radius=initial_radius,
        arc_extent=arc_extent,
        total_error=new_error,
        errors=np.square(residuals),
        has_multiple_revolutions=need_multiple_revolutions,
    )


def _fit_spiral_to_image_single_revolution_core(
    radii: NDArray[FloatType],
    theta: NDArray[FloatType],
    weights: NDArray[FloatType],
    initial_growth_factor: float,
) -> InternalLogSpiralFitResult[FloatType]:
    # Find suitable bounds for the offset parameter
    bad_bounds, (lower_bound, upper_bound), rotation_amount, _ = calculate_bounds(theta)
    rotated_theta = np.mod(theta - rotation_amount, 2 * np.pi)

    # Perform a fit to get the growth factor
    offset: float
    growth_factor: float
    if bad_bounds:
        offset = 0
        growth_factor = 0
    else:
        offset = (lower_bound + upper_bound) / 2
        res = optimize.least_squares(
            calculate_log_spiral_error_from_growth_factor,
            x0=initial_growth_factor,
            args=(radii, rotated_theta, weights, offset),
            kwargs={"use_modulo": True},
        )
        assert res.success, "Failed to fit growth factor"
        growth_factor = res.x[0]

    # Calculate the error from the fit
    initial_radius = calculate_best_initial_radius(
        radii,
        rotated_theta,
        weights,
        offset,
        growth_factor,
        use_modulo=True,
    )
    error, _ = calculate_log_spiral_error(
        radii,
        rotated_theta,
        weights,
        offset,
        growth_factor,
        initial_radius,
        use_modulo=True,
    )

    # Rotate back
    offset += rotation_amount

    theta_adjust = np.add(np.mod(theta - offset, 2 * np.pi), offset)
    # Get arc bounds
    arc_bounds = get_arc_bounds(offset, rotation_amount, lower_bound, upper_bound)
    return InternalLogSpiralFitResult(
        theta=theta_adjust,
        offset=offset,
        growth_factor=growth_factor,
        arc_bounds=arc_bounds,
        error=error,
        bad_bounds=bad_bounds,
    )


def _fit_spiral_to_image_multiple_revolution_core(
    radii: NDArray[FloatType],
    theta: NDArray[FloatType],
    weights: NDArray[FloatType],
    initial_growth_factor: float,
    inner_region: NDArray[BoolType],
) -> InternalLogSpiralFitResult[FloatType]:
    # fitting depends on the arc bounds, but we don't know what the arc
    # bounds are (i.e., whether to calculate the bounds from the inner or
    # outer points) until we know the chirality.  Since we only know the
    # chirality after the fit, we do two fits, one assuming CW and the
    # other assuming CCW, and take the one with the better error.

    # For the >2*pi case, we don't do mod 2*pi in the rotations because
    # theta-values can be outside the range [0, 2*pi], and discontinuities
    # in theta-values can severely impact the fitting.

    min_theta: float = float(np.min(theta))
    max_theta: float = float(np.max(theta))

    # TODO(pavyamsiri): Growth factor optimisation is totally wrong
    # Assume chirality is clockwise and fit a spiral
    cw_rotated_theta, cw_offset, cw_growth_factor, cw_error = __fit_multiple_revolution_spiral(
        radii,
        theta,
        weights,
        ~inner_region,
        initial_growth_factor,
        clockwise=True,
    )
    # Assume chirality is counter clockwise and fit a spiral
    ccw_rotated_theta, ccw_offset, ccw_growth_factor, ccw_error = __fit_multiple_revolution_spiral(
        radii,
        theta,
        weights,
        inner_region,
        initial_growth_factor,
        clockwise=False,
    )

    if cw_error < ccw_error:
        adjusted_theta = cw_rotated_theta
        offset = cw_offset
        growth_factor = cw_growth_factor
        error = cw_error
    else:
        adjusted_theta = ccw_rotated_theta
        offset = ccw_offset
        growth_factor = ccw_growth_factor
        error = ccw_error

    # Construct arc bounds
    if offset > min_theta:
        arc_bounds = (min_theta - offset, max_theta - offset)
    else:
        arc_bounds = (min_theta - offset + 2 * np.pi, max_theta - offset + 2 * np.pi)

    return InternalLogSpiralFitResult(
        theta=adjusted_theta,
        offset=offset,
        growth_factor=growth_factor,
        arc_bounds=arc_bounds,
        error=error,
        bad_bounds=False,
    )


def __fit_multiple_revolution_spiral(
    radii: NDArray[FloatType],
    theta: NDArray[FloatType],
    weights: NDArray[FloatType],
    region: NDArray[BoolType],
    initial_growth_factor: float,
    *,
    clockwise: bool,
) -> tuple[NDArray[FloatType], float, float, float]:
    bad_bounds, (lower_bound, upper_bound), rotation_amount, _ = calculate_bounds(
        theta[region],
    )
    assert not bad_bounds
    rotated_theta = np.subtract(theta, rotation_amount)
    offset = (lower_bound + upper_bound) / 2
    growth_factor_bounds = (0, np.inf) if clockwise else (-np.inf, 0)
    # NOTE: Have to invert the guess if counter clockwise to fit in the bounds
    if not clockwise and initial_growth_factor > 0:
        initial_growth_factor = -initial_growth_factor
    if clockwise and initial_growth_factor < 0:
        initial_growth_factor = -initial_growth_factor
    res = optimize.least_squares(
        calculate_log_spiral_error_from_growth_factor,
        x0=initial_growth_factor,
        bounds=growth_factor_bounds,
        args=(radii, rotated_theta, weights, offset),
        kwargs={"use_modulo": False},
        ftol=MULTIPLE_REVOLUTION_TOLERANCE,
        gtol=MULTIPLE_REVOLUTION_TOLERANCE,
        xtol=MULTIPLE_REVOLUTION_TOLERANCE,
    )
    assert res.success, "Failed to fit growth factor"
    growth_factor: float = cast(float, res.x[0])

    # Calculate the error from the fit
    initial_radius = calculate_best_initial_radius(
        radii,
        rotated_theta,
        weights,
        offset,
        growth_factor,
        use_modulo=False,
    )
    error, _ = calculate_log_spiral_error(
        radii,
        rotated_theta,
        weights,
        offset,
        growth_factor,
        initial_radius,
        use_modulo=False,
    )

    # Rotate back
    offset = (offset + rotation_amount) % (2 * np.pi)

    return (rotated_theta, offset, growth_factor, error)


def __cluster_has_no_endpoints_or_contains_origin(
    image: NDArray[FloatType],
    max_half_gap_fill_for_undefined_bounds: int = 3,
) -> bool:
    # See if the cluster has actual spiral endpoints by seeing if it is
    # possible to "escape" from the center point to the image boundary,
    # considering non-cluster pixels as empty pixels.
    centre_indices = get_origin_points(image)
    centre_in_cluster = any(image[row_idx, column_idx] for row_idx, column_idx in centre_indices)
    if centre_in_cluster:
        return True
    in_cluster = image > 0
    structure_element_size = 2 * max_half_gap_fill_for_undefined_bounds + 1
    structure_element = np.ones((structure_element_size, structure_element_size))
    is_hole_or_cluster = ndimage.binary_fill_holes(
        ndimage.binary_closing(in_cluster, structure=structure_element),
    )
    assert is_hole_or_cluster is not None
    # NOTE: Check if is_hole_or_cluster is not already a binary array
    return any(is_hole_or_cluster[row_idx, column_idx] > 0 for row_idx, column_idx in centre_indices)


def identify_inner_and_outer_spiral(
    image: NDArray[FloatType],
    shrink_amount: int,
    max_diagonal_distance: float = 1.5,
) -> NDArray[BoolType] | None:
    """Identify the inner and outer portion of a mutliple revolution spiral.

    Parameters
    ----------
    image : NDArray[FloatType]
        The cluster encoded as an image of non-zero pixels.
    shrink_amount : int
        The amount of polar angle bin units to each cluster by on each side.
    max_diagonal_distance : float
        The maximum diagonal distance allowed between neighbours in pixels.

    Returns
    -------
    NDArray[BoolType] | None
        An array of booleans which determine whether a cluster pixel is part of the inner or outer
        spiral. If the pixel is part of the inner spiral it will have a value of `True`.
        This value is `None` if it is not possible to split the spiral into two parts.

    """
    num_radii = int(np.ceil(max((image.shape[0], image.shape[1])) / 2))
    num_theta: int = 360

    min_acceptable_length = 5 * np.ceil(num_theta / 360)
    # Find theta bins which contain only a single revolution
    can_be_single_revolution = _find_single_revolution_regions(
        image,
        num_radii,
        num_theta,
        min_acceptable_length,
        shrink_amount,
    )

    # Find the start and end of each region
    single_revolution_differences = np.diff(can_be_single_revolution.astype(np.float32))
    start_indices: NDArray[np.int32] = np.flatnonzero(single_revolution_differences == 1) + 1
    end_indices: NDArray[np.int32] = np.flatnonzero(single_revolution_differences == -1)

    # No start and end
    if len(start_indices) == 0 and len(end_indices) == 0:
        # Single revolution for the entire theta-range, so no endpoints picked up
        # (this could be a ring, but more likely it"s a cluster in the
        # center)
        if np.all(can_be_single_revolution):
            start_indices = np.array([1], dtype=np.int32)
            end_indices = np.array([len(can_be_single_revolution) - 1], dtype=np.int32)
        # Suboptimal: No single revolution regions in entire theta-range
        else:
            assert not np.any(can_be_single_revolution)
            return None

    start_indices, end_indices, wrap_data = __calculate_wrap(
        can_be_single_revolution,
        start_indices,
        end_indices,
    )
    theta_bin_values: NDArray[FloatType] = np.linspace(
        2 * np.pi,
        0,
        num_theta,
        endpoint=False,
    ).astype(image.dtype)[::-1]
    row_indices, column_indices = image.nonzero()
    cluster_mask = np.zeros_like(image, dtype=np.bool_)
    cluster_mask[row_indices, column_indices] = True
    image_index_to_point_index = np.full(
        (image.shape[0], image.shape[1]),
        -1,
        dtype=np.int32,
    )
    image_index_to_point_index[image > 0] = np.arange(len(row_indices))
    radii, theta, _ = get_polar_coordinates(image)

    first_region, second_region, max_length = __split_regions(
        start_indices,
        end_indices,
        theta,
        theta_bin_values,
        wrap_data,
    )

    # Suboptimal: Longest single revolution region length is below the minimum length
    if max_length < min_acceptable_length:
        return None

    first_region_mask = np.zeros_like(image, dtype=np.bool_)
    first_region_mask[row_indices[first_region], column_indices[first_region]] = True
    second_region_mask = np.zeros_like(image, dtype=np.bool_)
    second_region_mask[row_indices[second_region], column_indices[second_region]] = True
    non_region_mask = np.zeros_like(image, dtype=np.int32)
    non_region_mask[
        np.logical_and(
            cluster_mask,
            np.logical_and(~first_region_mask, ~second_region_mask),
        )
    ] = 1

    first_region_distance = ndimage.distance_transform_edt(
        ~first_region_mask,
        return_distances=True,
    )
    assert isinstance(first_region_distance, np.ndarray)
    second_region_distance = ndimage.distance_transform_edt(
        ~second_region_mask,
        return_distances=True,
    )
    assert isinstance(second_region_distance, np.ndarray)
    connected_components, num_components = skimage.measure.label(
        non_region_mask,
        return_num=True,
    )
    assert isinstance(connected_components, np.ndarray)
    for label in range(1, num_components + 1):
        current_row_indices, current_column_indices = (connected_components == label).nonzero()
        point_indices = image_index_to_point_index[
            current_row_indices,
            current_column_indices,
        ]
        first_distance = first_region_distance[
            current_row_indices,
            current_column_indices,
        ].min()
        second_distance = second_region_distance[
            current_row_indices,
            current_column_indices,
        ].min()
        if first_distance < max_diagonal_distance:
            first_region[point_indices] = True
        elif second_distance < max_diagonal_distance:
            second_region[point_indices] = True

    # Use updated regions
    first_region_mask = np.zeros_like(image, dtype=np.bool_)
    first_region_mask[row_indices[first_region], column_indices[first_region]] = True
    second_region_mask = np.zeros_like(image, dtype=np.bool_)
    second_region_mask[row_indices[second_region], column_indices[second_region]] = True
    non_region_mask = np.zeros_like(image, dtype=np.int32)
    non_region_mask[
        np.logical_and(
            cluster_mask,
            np.logical_and(~first_region_mask, ~second_region_mask),
        )
    ] = 1

    # Combine

    # Assign the remaining pixels according to closest distance to one of the
    # two regions

    first_region_distance = ndimage.distance_transform_edt(
        ~first_region_mask,
        return_distances=True,
    )
    assert isinstance(first_region_distance, np.ndarray)
    second_region_distance = ndimage.distance_transform_edt(
        ~second_region_mask,
        return_distances=True,
    )
    assert isinstance(second_region_distance, np.ndarray)
    connected_components, num_components = skimage.measure.label(
        non_region_mask,
        return_num=True,
    )
    assert isinstance(connected_components, np.ndarray)
    for label in range(1, num_components + 1):
        current_row_indices, current_column_indices = (connected_components == label).nonzero()
        point_indices = image_index_to_point_index[
            current_row_indices,
            current_column_indices,
        ]
        first_distances = first_region_distance[
            current_row_indices,
            current_column_indices,
        ]
        second_distances = second_region_distance[
            current_row_indices,
            current_column_indices,
        ]
        if first_distances.min() < second_distances.min():
            first_region[point_indices] = True
            first_region_mask = np.zeros_like(image, dtype=np.bool_)
            first_region_mask[
                row_indices[first_region],
                column_indices[first_region],
            ] = True
            first_region_distance = ndimage.distance_transform_edt(
                ~first_region_mask,
                return_distances=True,
            )
            assert isinstance(first_region_distance, np.ndarray)
        else:
            second_region[point_indices] = True
            second_region_mask = np.zeros_like(image, dtype=np.bool_)
            second_region_mask[
                row_indices[second_region],
                column_indices[second_region],
            ] = True
            second_region_distance = ndimage.distance_transform_edt(
                ~second_region_mask,
                return_distances=True,
            )
            assert isinstance(second_region_distance, np.ndarray)

    assert np.all(
        np.logical_xor(first_region, second_region),
    ), "First and second regions are inconsistent!"

    # Find innermost region
    first_radii = radii[first_region]
    second_radii = radii[second_region]
    first_mean_radius = np.mean(first_radii) if len(first_radii) > 0 else np.inf
    second_mean_radius = np.mean(second_radii) if len(second_radii) > 0 else np.inf
    assert not (np.isinf(first_mean_radius) and np.isinf(second_mean_radius))
    if first_mean_radius < second_mean_radius:
        return first_region
    return second_region


def _find_single_revolution_regions(
    image: NDArray[FloatType],
    num_radii: int,
    num_theta: int,
    min_acceptable_length: int,
    shrink_amount: int,
) -> NDArray[BoolType]:
    assert shrink_amount <= min_acceptable_length
    polar_image = np.flip(
        __image_transform_from_cartesian_to_polar(image, num_radii, num_theta),
        axis=1,
    )
    polar_image = np.nan_to_num(polar_image, nan=0)

    dilated_polar_image: NDArray[BoolType] = ndimage.binary_dilation(
        polar_image,
        structure=np.ones((3, 3)),
    )

    return _find_single_revolution_regions_polar(dilated_polar_image, shrink_amount)


def _find_single_revolution_regions_polar(
    polar_image: NDArray[BoolType],
    shrink_amount: int,
) -> NDArray[BoolType]:
    num_radii: int = polar_image.shape[0]
    num_theta: int = polar_image.shape[1]
    # Pad columns with zeros
    theta_diff = np.diff(polar_image, prepend=0, append=0, axis=0)
    # Theta bins which only have a single start and end point
    # i.e. ray from centre only hits the cluster once and not multiple times
    can_be_single_revolution = np.logical_and(
        np.sum(theta_diff == -1, axis=0) == 1,
        np.sum(theta_diff == 1, axis=0) == 1,
    )
    # Radial index for every point in the polar image
    radial_locations = np.tile(np.arange(1, num_radii + 1), (num_theta, 1)).T

    polar_image_for_min = polar_image.astype(np.float32)
    polar_image_for_min[polar_image_for_min == 0] = np.inf
    min_locations = np.min(polar_image_for_min * radial_locations, axis=0)
    min_locations[np.isinf(min_locations)] = 0
    max_locations = np.max(polar_image.astype(np.float32) * radial_locations, axis=0)
    neighbour_max_location_left = np.roll(max_locations, 1)
    neighbour_max_location_right = np.roll(max_locations, -1)
    neighbour_min_location_left = np.roll(min_locations, 1)
    neighbour_min_location_right = np.roll(min_locations, -1)
    conditions: Sequence[NDArray[BoolType]] = (
        can_be_single_revolution,
        np.logical_or(
            min_locations <= neighbour_max_location_left,
            neighbour_max_location_left == 0,
        ),
        np.logical_or(
            min_locations <= neighbour_max_location_right,
            neighbour_max_location_right == 0,
        ),
        np.logical_or(
            max_locations >= neighbour_min_location_left,
            neighbour_min_location_left == 0,
        ),
        np.logical_or(
            max_locations >= neighbour_min_location_right,
            neighbour_min_location_right == 0,
        ),
    )
    can_be_single_revolution = functools.reduce(
        lambda x, y: np.logical_and(x, y),
        conditions,
    )

    conditions = (
        can_be_single_revolution,
        np.roll(can_be_single_revolution, shrink_amount),
        np.roll(can_be_single_revolution, -shrink_amount),
    )

    return functools.reduce(lambda x, y: np.logical_and(x, y), conditions)


def __image_transform_from_cartesian_to_polar(
    image: NDArray[FloatType],
    num_radii: int,
    num_theta: int,
) -> NDArray[FloatType]:
    centre_x = image.shape[1] / 2 - 0.5
    centre_y = image.shape[0] / 2 - 0.5

    # Calculate maximum radius value
    row_indices, column_indices = image.nonzero()
    dx = max(centre_x, (column_indices - centre_x).max())
    dy = max(centre_y, -(row_indices - centre_y).max())
    max_radius = np.sqrt(dx**2 + dy**2)

    return (
        skimage.transform.warp_polar(
            image,
            center=(centre_x, centre_y),
            radius=max_radius,
            output_shape=(num_theta, num_radii),
        )
        .astype(image.dtype)
        .T
    )


def __split_regions(
    start_indices: NDArray[IndexType],
    end_indices: NDArray[IndexType],
    theta: NDArray[FloatType],
    theta_bin_centres: NDArray[FloatType],
    wrap_data: WrapData | None,
) -> tuple[NDArray[BoolType], NDArray[BoolType], int]:
    region_lengths = end_indices - start_indices + 1
    max_region_length: int | None = region_lengths.max() if len(region_lengths) > 0 else None
    only_wrap_exists: bool = max_region_length is None
    wrap_larger_than_all_regions: bool = False
    if wrap_data is not None and max_region_length is not None:
        wrap_length = wrap_data.length
        wrap_larger_than_all_regions = wrap_length > max_region_length
    if only_wrap_exists or wrap_larger_than_all_regions:
        # Only wrap exists
        assert wrap_data is not None
        max_length = wrap_data.length
        wrap_mid_length = np.round(wrap_data.length / 2)
        theta_start = theta_bin_centres[wrap_data.start]
        theta_end = theta_bin_centres[wrap_data.end]
        inner_region = np.logical_or(theta >= theta_start, theta < theta_end)
        if wrap_data.start_length >= wrap_mid_length:
            split_theta_idx = int(wrap_data.start + wrap_mid_length - 1)
        else:
            split_theta_idx = int(wrap_mid_length - wrap_data.start_length - 1)
        # NOTE: Sometimes we need to wrap the angles
        split_theta_idx %= len(theta_bin_centres)
        split_theta = theta_bin_centres[split_theta_idx]
        first_region = np.logical_and(
            inner_region,
            np.logical_and(theta >= split_theta, theta < theta_end),
        )
        second_region = np.logical_and(inner_region, ~first_region)
    else:
        assert max_region_length is not None
        max_length = max_region_length
        max_index = region_lengths.argmax()
        theta_start = theta_bin_centres[start_indices[max_index]]
        theta_end = theta_bin_centres[start_indices[max_index]]
        split_theta = theta_bin_centres[round((start_indices[max_index] + end_indices[max_index]) / 2)]
        first_region = np.logical_and(theta >= theta_start, theta < split_theta)
        second_region = np.logical_and(theta >= split_theta, theta < theta_end)
    return first_region, second_region, max_length


def __calculate_wrap(
    can_be_single_revolution: NDArray[BoolType],
    start_indices: NDArray[IndexType],
    end_indices: NDArray[IndexType],
) -> tuple[NDArray[IndexType], NDArray[IndexType], WrapData | None]:
    has_wrapped: bool = False
    wrap_start: int | None = None
    wrap_end: int | None = None
    # Region ends but either doesn't start or it wraps
    if len(end_indices) > 0 and (len(start_indices) == 0 or start_indices[0] > end_indices[0]):
        has_wrapped = True
        wrap_end = end_indices[0] + 1
        end_indices = end_indices[1:]
        assert len(start_indices) == 0 or len(end_indices) == 0 or start_indices[0] <= end_indices[0]
    # Region starts but either doesn't end or it wraps
    if len(start_indices) > 0 and (len(end_indices) == 0 or start_indices[-1] > end_indices[-1]):
        has_wrapped = True
        wrap_start = start_indices[-1]
        start_indices = start_indices[:-1]
        assert len(start_indices) == 0 or len(end_indices) == 0 or start_indices[-1] <= end_indices[-1]
    assert len(start_indices) == len(end_indices)

    wrap_data: WrapData | None = None
    if has_wrapped:
        # Last continuous single revolution region is at the beginning, but doesn"t
        # actually wrap around
        if wrap_start is None:
            assert wrap_end is not None, "Other branch must be executed as this has wrapped"
            start_indices = np.insert(start_indices, 0, 0)
            end_indices = np.insert(end_indices, 0, wrap_end)
        # Last continuous single revolution region is at the end, but doesn"t
        # actually wrap around
        elif wrap_end is None:
            assert wrap_start is not None, "Other branch must be executed as this has wrapped"
            start_indices = np.insert(start_indices, 0, wrap_start)
            end_indices = np.insert(end_indices, 0, len(can_be_single_revolution) - 1)
        # Wrap does happen
        else:
            # Length of region from start to edge
            wrap_start_length = len(can_be_single_revolution) - wrap_start + 1
            # Length of region from end to end
            wrap_end_length = wrap_end
            wrap_data = WrapData(
                start_length=wrap_start_length,
                end_length=wrap_end_length,
                length=wrap_start_length + wrap_end_length,
                start=wrap_start,
                end=wrap_end,
            )
    return (start_indices, end_indices, wrap_data)


def _remove_theta_discontinuities(
    theta: NDArray[FloatType],
    image: NDArray[FloatType],
    inner_region: NDArray[BoolType],
) -> NDArray[FloatType]:
    assert np.count_nonzero(inner_region) > 0
    adjusted_theta = np.copy(theta)
    inner_adjusted_theta = _adjust_theta_for_gap(theta, image, inner_region)
    if inner_adjusted_theta is not None:
        adjusted_theta = inner_adjusted_theta

    outer_adjusted_theta = _adjust_theta_for_gap(theta, image, ~inner_region)
    if outer_adjusted_theta is not None:
        adjusted_theta = outer_adjusted_theta

    modded_adjusted_theta = adjusted_theta % (2 * np.pi)

    min_theta_index = np.argmin(modded_adjusted_theta)
    min_theta = modded_adjusted_theta[min_theta_index]

    max_theta_index = np.argmax(modded_adjusted_theta)
    max_theta = modded_adjusted_theta[max_theta_index]

    min_theta_multiple = int(np.floor(min_theta / (2 * np.pi)))
    max_theta_multiple = int(np.floor(max_theta / (2 * np.pi)))

    if min_theta_multiple != max_theta_multiple:
        assert inner_adjusted_theta is None
        assert outer_adjusted_theta is None
        if inner_region[min_theta_index]:
            assert not inner_region[max_theta_index]
            adjusted_theta[inner_region] = modded_adjusted_theta[inner_region] + (2 * np.pi) * (max_theta_multiple + 1)
        else:
            assert inner_region[max_theta_index]
            adjusted_theta[~inner_region] = modded_adjusted_theta[~inner_region] + (2 * np.pi) * (max_theta_multiple + 1)

    return adjusted_theta


def _adjust_theta_for_gap(
    theta: NDArray[FloatType],
    image: NDArray[FloatType],
    region: NDArray[BoolType],
) -> NDArray[FloatType] | None:
    row_indices, column_indices = image.nonzero()
    assert len(row_indices) == len(column_indices)
    assert len(region) == len(row_indices)

    sorted_theta = np.sort(theta[region])
    gaps = np.diff(sorted_theta)
    max_gap_idx = np.argmax(gaps)
    max_gap_size = gaps[max_gap_idx]
    end_gap = sorted_theta[0] + 2 * np.pi - sorted_theta[-1]
    # Gap in points doesn't go through zero so some points will need to
    # be adjusted for continuity
    if end_gap < max_gap_size:
        adjusted_theta = np.copy(theta)
        # Split points into points above and below x-axis
        # Theta value before gap
        max_theta = sorted_theta[max_gap_idx]
        top_half = np.logical_and(region, theta <= max_theta)
        # Theta value after gap
        min_theta = sorted_theta[max_gap_idx + 1]
        bottom_half = np.logical_and(region, theta >= min_theta)

        negative_image = np.zeros_like(image, dtype=np.bool_)
        negative_image[row_indices[region], column_indices[region]] = True

        negative_distances = ndimage.distance_transform_edt(
            negative_image,
            return_distances=True,
        )
        assert isinstance(negative_distances, np.ndarray)

        min_top_distance = np.min(
            negative_distances[row_indices[top_half], column_indices[top_half]],
        )
        min_bottom_distance = np.min(
            negative_distances[row_indices[bottom_half], column_indices[bottom_half]],
        )
        if min_top_distance > min_bottom_distance:
            adjusted_theta[top_half] += 2 * np.pi
        else:
            adjusted_theta[bottom_half] -= 2 * np.pi
        return adjusted_theta
    return None
