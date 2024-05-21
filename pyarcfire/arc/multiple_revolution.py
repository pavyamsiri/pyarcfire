# Standard libraries
from dataclasses import dataclass
import functools
import logging
from typing import Sequence

# External libraries
import numpy as np
from scipy import ndimage
from scipy import optimize
import skimage
import skimage.measure

# Internal libraries
from pyarcfire.definitions import (
    FloatArray1D,
    ImageBoolArray,
    ImageFloatArray,
    IntegerArray1D,
    BoolArray1D,
)
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

MULTIPLE_REVOLUTION_TOLERANCE: float = 1e-12

log = logging.getLogger(__name__)


@dataclass
class WrapData:
    start_length: int
    end_length: int
    length: int
    start: int
    end: int


@dataclass
class InternalLogSpiralFitResult:
    theta: FloatArray1D
    offset: float
    pitch_angle: float
    arc_bounds: tuple[float, float]
    error: float


def fit_spiral_to_image_multiple_revolution(
    image: ImageFloatArray,
    initial_pitch_angle: float = 0,
) -> LogSpiralFitResult:
    # Convert to polar coordinates
    radii, theta, weights = _get_polar_coordinates(image)

    # Check if the cluster revolves more than once
    bad_bounds, _, _, _ = _calculate_bounds(theta)

    inner_region: BoolArray1D | None = None

    # Gap in theta is not large enough to not need multiple revolutions
    # and the cluster does not contain the origin or is not closed around centre
    if bad_bounds or not __cluster_has_no_endpoints_or_contains_origin(image):
        inner_region = identify_inner_and_outer_spiral(image, shrink_amount=5)
        if (
            inner_region is None
            or inner_region.sum() == 0
            or inner_region.sum() == len(inner_region)
        ):
            log.debug("Don't need multiple revolutions")
            inner_region = None
        else:
            theta = _remove_theta_discontinuities(theta, image, inner_region)
            log.debug(f"Adjusted theta = [{theta.min()}, {theta.max()}]")
    log.debug(f"Need multiple revolutions = {inner_region is not None}")
    need_multiple_revolutions: bool = inner_region is not None
    if need_multiple_revolutions:
        fit_result = _fit_spiral_to_image_multiple_revolution_core(
            radii, theta, weights, initial_pitch_angle, inner_region
        )
    else:
        fit_result = _fit_spiral_to_image_single_revolution_core(
            radii, theta, weights, initial_pitch_angle
        )

    arc_bounds = fit_result.arc_bounds
    offset = fit_result.offset
    pitch_angle = fit_result.pitch_angle
    error = fit_result.error
    theta = fit_result.theta

    # Adjust so that arc bounds is relative to theta
    (theta, arc_bounds, offset) = _adjust_theta_to_zero(
        theta, arc_bounds, offset, not need_multiple_revolutions
    )

    # Recalculate initial radius and error after adjustment
    initial_radius = calculate_best_initial_radius(
        radii,
        theta,
        weights,
        offset,
        pitch_angle,
        use_modulo=not need_multiple_revolutions,
    )
    new_error, residuals = calculate_log_spiral_error(
        radii,
        theta,
        weights,
        offset,
        pitch_angle,
        initial_radius,
        use_modulo=not need_multiple_revolutions,
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
        has_multiple_revolutions=need_multiple_revolutions,
    )

    return result


def _fit_spiral_to_image_single_revolution_core(
    radii: FloatArray1D,
    theta: FloatArray1D,
    weights: FloatArray1D,
    initial_pitch_angle: float,
) -> InternalLogSpiralFitResult:
    # Find suitable bounds for the offset parameter
    bad_bounds, (lower_bound, upper_bound), rotation_amount, max_gap_size = (
        _calculate_bounds(theta)
    )
    rotated_theta = (theta - rotation_amount) % (2 * np.pi)

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
            args=(radii, rotated_theta, weights, offset, True),
        )
        assert res.success, "Failed to fit pitch angle"
        pitch_angle = res.x[0]

    # Calculate the error from the fit
    initial_radius = calculate_best_initial_radius(
        radii, rotated_theta, weights, offset, pitch_angle, use_modulo=True
    )
    error, _ = calculate_log_spiral_error(
        radii,
        rotated_theta,
        weights,
        offset,
        pitch_angle,
        initial_radius,
        use_modulo=True,
    )

    # Rotate back
    offset += rotation_amount

    theta_adjust = (theta - offset) % (2 * np.pi) + offset
    # Get arc bounds
    arc_bounds = _get_arc_bounds(offset, rotation_amount, lower_bound, upper_bound)
    fit_result = InternalLogSpiralFitResult(
        theta=theta_adjust,
        offset=offset,
        pitch_angle=pitch_angle,
        arc_bounds=arc_bounds,
        error=error,
    )
    return fit_result


def _fit_spiral_to_image_multiple_revolution_core(
    radii: FloatArray1D,
    theta: FloatArray1D,
    weights: FloatArray1D,
    initial_pitch_angle: float,
    inner_region: BoolArray1D,
) -> InternalLogSpiralFitResult:
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

    # TODO: Pitch angle optimisation is totally wrong
    # Assume chirality is clockwise and fit a spiral
    cw_rotated_theta, cw_offset, cw_pitch_angle, cw_error = (
        __fit_multiple_revolution_spiral(
            radii, theta, weights, ~inner_region, initial_pitch_angle, clockwise=True
        )
    )
    # Assume chirality is counter clockwise and fit a spiral
    ccw_rotated_theta, ccw_offset, ccw_pitch_angle, ccw_error = (
        __fit_multiple_revolution_spiral(
            radii, theta, weights, inner_region, initial_pitch_angle, clockwise=False
        )
    )

    if cw_error < ccw_error:
        adjusted_theta = cw_rotated_theta
        offset = cw_offset
        pitch_angle = cw_pitch_angle
        error = cw_error
    else:
        adjusted_theta = ccw_rotated_theta
        offset = ccw_offset
        pitch_angle = ccw_pitch_angle
        error = ccw_error

    # Construct arc bounds
    if offset > min_theta:
        arc_bounds = (min_theta - offset, max_theta - offset)
    else:
        arc_bounds = (min_theta - offset + 2 * np.pi, max_theta - offset + 2 * np.pi)

    result = InternalLogSpiralFitResult(
        theta=adjusted_theta,
        offset=offset,
        pitch_angle=pitch_angle,
        arc_bounds=arc_bounds,
        error=error,
    )
    return result


def __fit_multiple_revolution_spiral(
    radii: FloatArray1D,
    theta: FloatArray1D,
    weights: FloatArray1D,
    region: BoolArray1D,
    initial_pitch_angle: float,
    clockwise: bool,
) -> tuple[FloatArray1D, float, float, float]:
    bad_bounds, (lower_bound, upper_bound), rotation_amount, _ = _calculate_bounds(
        theta[region]
    )
    assert not bad_bounds
    rotated_theta = theta - rotation_amount
    offset = (lower_bound + upper_bound) / 2
    pitch_angle_bounds = (0, np.inf) if clockwise else (-np.inf, 0)
    res = optimize.least_squares(
        calculate_log_spiral_error_from_pitch_angle,
        x0=initial_pitch_angle,
        bounds=pitch_angle_bounds,
        args=(radii, rotated_theta, weights, offset, False),
        ftol=MULTIPLE_REVOLUTION_TOLERANCE,
        gtol=MULTIPLE_REVOLUTION_TOLERANCE,
        xtol=MULTIPLE_REVOLUTION_TOLERANCE,
    )
    assert res.success, "Failed to fit pitch angle"
    pitch_angle = res.x[0]

    # Calculate the error from the fit
    initial_radius = calculate_best_initial_radius(
        radii, rotated_theta, weights, offset, pitch_angle, use_modulo=False
    )
    error, _ = calculate_log_spiral_error(
        radii,
        rotated_theta,
        weights,
        offset,
        pitch_angle,
        initial_radius,
        use_modulo=False,
    )

    # Rotate back
    offset = (offset + rotation_amount) % (2 * np.pi)

    return (rotated_theta, offset, pitch_angle, error)


def __cluster_has_no_endpoints_or_contains_origin(
    image: ImageFloatArray, max_half_gap_fill_for_undefined_bounds: int = 3
) -> bool:
    # See if the cluster has actual spiral endpoints by seeing if it is
    # possible to "escape" from the center point to the image boundary,
    # considering non-cluster pixels as empty pixels.
    central_row: int = image.shape[0] // 2
    central_column: int = image.shape[1] // 2
    centre_in_cluster = image[central_row, central_column] != 0
    if centre_in_cluster:
        return True
    in_cluster = image > 0
    structure_element_size = 2 * max_half_gap_fill_for_undefined_bounds + 1
    structure_element = np.ones((structure_element_size, structure_element_size))
    is_hole_or_cluster = ndimage.binary_fill_holes(
        ndimage.binary_closing(in_cluster, structure=structure_element)
    )
    assert is_hole_or_cluster is not None
    # NOTE: Check if is_hole_or_cluster is not already a binary array
    return is_hole_or_cluster[central_row, central_column] > 0


def identify_inner_and_outer_spiral(
    image: ImageFloatArray, shrink_amount: int, max_diagonal_distance: float = 1.5
) -> BoolArray1D | None:
    num_radii = int(np.ceil(max((image.shape[0], image.shape[1])) / 2))
    num_theta: int = 360

    min_acceptable_length = 5 * np.ceil(num_theta / 360)
    # Find theta bins which contain only a single revolution
    can_be_single_revolution = _find_single_revolution_regions(
        image, num_radii, num_theta, min_acceptable_length, shrink_amount
    )

    # Find the start and end of each region
    single_revolution_differences = np.diff(can_be_single_revolution.astype(np.float32))
    start_indices: IntegerArray1D = (single_revolution_differences == 1).nonzero()[
        0
    ].astype(np.int32) + 1
    end_indices: IntegerArray1D = (
        (single_revolution_differences == -1).nonzero()[0].astype(np.int32)
    )

    # No start and end
    if len(start_indices) == 0 and len(end_indices) == 0:
        # Single revolution for the entire theta-range, so no endpoints picked up
        # (this could be a ring, but more likely it"s a cluster in the
        # center)
        if np.all(can_be_single_revolution):
            start_indices = np.array([1], dtype=np.int32)
            end_indices = np.array([len(can_be_single_revolution) - 1], dtype=np.int32)
        else:
            assert not np.any(can_be_single_revolution)
            log.warn("No single revolution regions in entire theta-range")
            return None

    start_indices, end_indices, wrap_data = __calculate_wrap(
        can_be_single_revolution, start_indices, end_indices
    )
    theta_bin_values = np.arange(1, num_theta) * 2 * np.pi / num_theta
    row_indices, column_indices = image.nonzero()
    cluster_mask = np.zeros_like(image, dtype=np.bool_)
    cluster_mask[row_indices, column_indices] = True
    image_index_to_point_index = np.full(
        (image.shape[0], image.shape[1]), -1, dtype=np.int32
    )
    image_index_to_point_index[image > 0] = np.arange(len(row_indices))
    radii, theta, _ = _get_polar_coordinates(image)

    first_region, second_region, max_length = __split_regions(
        start_indices, end_indices, theta, theta_bin_values, wrap_data
    )

    if max_length < min_acceptable_length:
        log.warn(
            f"Warning:idInnerOuterSpiral:longest sgl-rev region length ({max_length}) is below the minimum length {min_acceptable_length}"
        )
        return None

    first_region_mask = np.zeros_like(image, dtype=np.bool_)
    first_region_mask[row_indices[first_region], column_indices[first_region]] = True
    second_region_mask = np.zeros_like(image, dtype=np.bool_)
    second_region_mask[row_indices[second_region], column_indices[second_region]] = True
    non_region_mask = np.zeros_like(image, dtype=np.int32)
    non_region_mask[
        np.logical_and(
            cluster_mask, np.logical_and(~first_region_mask, ~second_region_mask)
        )
    ] = 1

    first_region_distance = ndimage.distance_transform_edt(
        ~first_region_mask, return_distances=True
    )
    assert isinstance(first_region_distance, np.ndarray)
    second_region_distance = ndimage.distance_transform_edt(
        ~second_region_mask, return_distances=True
    )
    assert isinstance(second_region_distance, np.ndarray)
    connected_components, num_components = skimage.measure.label(  # type:ignore
        non_region_mask, return_num=True
    )
    assert isinstance(connected_components, np.ndarray)
    for label in range(1, num_components + 1):
        current_row_indices, current_column_indices = (
            connected_components == label
        ).nonzero()
        point_indices = image_index_to_point_index[
            current_row_indices, current_column_indices
        ]
        first_distance = first_region_distance[
            current_row_indices, current_column_indices
        ].min()
        second_distance = second_region_distance[
            current_row_indices, current_column_indices
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
            cluster_mask, np.logical_and(~first_region_mask, ~second_region_mask)
        )
    ] = 1

    # Combine

    # Assign the remaining pixels according to closest distance to one of the
    # two regions

    first_region_distance = ndimage.distance_transform_edt(
        ~first_region_mask, return_distances=True
    )
    assert isinstance(first_region_distance, np.ndarray)
    second_region_distance = ndimage.distance_transform_edt(
        ~second_region_mask, return_distances=True
    )
    assert isinstance(second_region_distance, np.ndarray)
    connected_components, num_components = skimage.measure.label(  # type:ignore
        non_region_mask, return_num=True
    )
    assert isinstance(connected_components, np.ndarray)
    for label in range(1, num_components + 1):
        current_row_indices, current_column_indices = (
            connected_components == label
        ).nonzero()
        point_indices = image_index_to_point_index[
            current_row_indices, current_column_indices
        ]
        first_distances = first_region_distance[
            current_row_indices, current_column_indices
        ]
        second_distances = second_region_distance[
            current_row_indices, current_column_indices
        ]
        if first_distances.min() < second_distances.min():
            first_region[point_indices] = True
            first_region_mask = np.zeros_like(image, dtype=np.bool_)
            first_region_mask[
                row_indices[first_region], column_indices[first_region]
            ] = True
            first_region_distance = ndimage.distance_transform_edt(
                ~first_region_mask, return_distances=True
            )
            assert isinstance(first_region_distance, np.ndarray)
        else:
            second_region[point_indices] = True
            second_region_mask = np.zeros_like(image, dtype=np.bool_)
            second_region_mask[
                row_indices[second_region], column_indices[second_region]
            ] = True
            second_region_distance = ndimage.distance_transform_edt(
                ~second_region_mask, return_distances=True
            )
            assert isinstance(second_region_distance, np.ndarray)

    assert np.all(
        np.logical_xor(first_region, second_region)
    ), f"After closeness, XOR = {(~np.logical_xor(first_region, second_region)).nonzero()}"

    # Find innermost region
    first_radii = radii[first_region]
    second_radii = radii[second_region]
    first_mean_radius = np.mean(first_radii) if len(first_radii) > 0 else np.inf
    second_mean_radius = np.mean(second_radii) if len(second_radii) > 0 else np.inf
    assert not (np.isinf(first_mean_radius) and np.isinf(second_mean_radius))
    if first_mean_radius < second_mean_radius:
        return first_region
    else:
        return second_region


def _find_single_revolution_regions(
    image: ImageFloatArray,
    num_radii: int,
    num_theta: int,
    min_acceptable_length: int,
    shrink_amount: int,
) -> BoolArray1D:
    assert shrink_amount <= min_acceptable_length
    polar_image = np.flip(
        __image_transform_from_cartesian_to_polar(image, num_radii, num_theta), axis=1
    )
    polar_image = np.nan_to_num(polar_image, nan=0)

    dilated_polar_image: ImageBoolArray = ndimage.binary_dilation(  # type:ignore
        polar_image, structure=np.ones((3, 3))
    )

    return _find_single_revolution_regions_polar(dilated_polar_image, shrink_amount)


def _find_single_revolution_regions_polar(
    polar_image: ImageBoolArray, shrink_amount: int
) -> BoolArray1D:
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
    conditions: Sequence[BoolArray1D] = (
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
        lambda x, y: np.logical_and(x, y), conditions
    )

    conditions = (
        can_be_single_revolution,
        np.roll(can_be_single_revolution, shrink_amount),
        np.roll(can_be_single_revolution, -shrink_amount),
    )

    can_be_single_revolution = functools.reduce(
        lambda x, y: np.logical_and(x, y), conditions
    )
    return can_be_single_revolution


def __image_transform_from_cartesian_to_polar(
    image: ImageFloatArray, num_radii: int, num_theta: int
) -> ImageFloatArray:
    centre_x = image.shape[1] / 2 + 0.5
    centre_y = image.shape[0] / 2 + 0.5 - 1

    # Calculate maximum radius value
    row_indices, column_indices = image.nonzero()
    dx = max(centre_x, (column_indices - centre_x).max())
    dy = max(centre_y, (row_indices - centre_y).max())
    max_radius = np.sqrt(dx**2 + dy**2)

    return skimage.transform.warp_polar(
        image,
        center=(centre_x, centre_y),
        radius=max_radius,
        output_shape=(num_theta, num_radii),
    ).T


def __split_regions(
    start_indices: IntegerArray1D,
    end_indices: IntegerArray1D,
    theta: FloatArray1D,
    theta_bin_centres: FloatArray1D,
    wrap_data: WrapData | None,
) -> tuple[BoolArray1D, BoolArray1D, int]:
    region_lengths = end_indices - start_indices + 1
    max_region_length: int | None = (
        region_lengths.max() if len(region_lengths) > 0 else None
    )
    only_wrap_exists: bool = max_region_length is None
    wrap_larger_than_all_regions: bool = False
    if wrap_data is not None and max_region_length is not None:
        wrap_length = wrap_data.length
        wrap_larger_than_all_regions = wrap_length > max_region_length
    if only_wrap_exists or wrap_larger_than_all_regions:
        log.debug("Wrap happens")
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
        split_theta = theta_bin_centres[split_theta_idx]
        first_region = np.logical_and(
            inner_region, np.logical_and(theta >= split_theta, theta < theta_end)
        )
        second_region = np.logical_and(inner_region, ~first_region)
    else:
        log.debug("No wrap")
        assert max_region_length is not None
        max_length = max_region_length
        max_index = region_lengths.argmax()
        theta_start = theta_bin_centres[start_indices[max_index]]
        theta_end = theta_bin_centres[start_indices[max_index]]
        split_theta = theta_bin_centres[
            round((start_indices[max_index] + end_indices[max_index]) / 2)
        ]
        first_region = np.logical_and(theta >= theta_start, theta < split_theta)
        second_region = np.logical_and(theta >= split_theta, theta < theta_end)
    log.debug(f"Max length = {max_length}")
    return first_region, second_region, max_length


def __calculate_wrap(
    can_be_single_revolution: BoolArray1D,
    start_indices: IntegerArray1D,
    end_indices: IntegerArray1D,
) -> tuple[IntegerArray1D, IntegerArray1D, WrapData | None]:
    has_wrapped: bool = False
    wrap_start: int | None = None
    wrap_end: int | None = None
    # Region ends but either doesn't start or it wraps
    if len(end_indices) > 0 and (
        len(start_indices) == 0 or start_indices[0] > end_indices[0]
    ):
        has_wrapped = True
        wrap_end = end_indices[0]
        end_indices = end_indices[1:]
        assert (
            len(start_indices) == 0
            or len(end_indices) == 0
            or start_indices[0] <= end_indices[0]
        )
    # Region starts but either doesn't end or it wraps
    if len(start_indices) > 0 and (
        len(end_indices) == 0 or start_indices[-1] > end_indices[-1]
    ):
        has_wrapped = True
        wrap_start = start_indices[-1]
        start_indices = start_indices[:-1]
        assert (
            len(start_indices) == 0
            or len(end_indices) == 0
            or start_indices[-1] <= end_indices[-1]
        )
    assert len(start_indices) == len(end_indices)

    wrap_data: WrapData | None = None
    if has_wrapped:
        # Last continuous single revolution region is at the beginning, but doesn"t
        # actually wrap around
        if wrap_start is None:
            assert (
                wrap_end is not None
            ), "Other branch must be executed as this has wrapped"
            start_indices = np.insert(start_indices, 0, 0)
            end_indices = np.insert(end_indices, 0, wrap_end)
        # Last continuous single revolution region is at the end, but doesn"t
        # actually wrap around
        elif wrap_end is None:
            assert (
                wrap_start is not None
            ), "Other branch must be executed as this has wrapped"
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
    theta: FloatArray1D, image: ImageFloatArray, inner_region: BoolArray1D
) -> FloatArray1D:
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
        log.warn("Theta-discontinuity through x-axis!")
        assert inner_adjusted_theta is None and outer_adjusted_theta is None
        if inner_region[min_theta_index]:
            assert not inner_region[max_theta_index]
            adjusted_theta[inner_region] = modded_adjusted_theta[inner_region] + (
                2 * np.pi
            ) * (max_theta_multiple + 1)
        else:
            assert inner_region[max_theta_index]
            adjusted_theta[~inner_region] = modded_adjusted_theta[~inner_region] + (
                2 * np.pi
            ) * (max_theta_multiple + 1)

    return adjusted_theta


def _adjust_theta_for_gap(
    theta: FloatArray1D, image: ImageFloatArray, region: BoolArray1D
) -> FloatArray1D | None:
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
            negative_image, return_distances=True
        )
        assert isinstance(negative_distances, np.ndarray)

        min_top_distance = np.min(
            negative_distances[row_indices[top_half], column_indices[top_half]]
        )
        min_bottom_distance = np.min(
            negative_distances[row_indices[bottom_half], column_indices[bottom_half]]
        )
        if min_top_distance > min_bottom_distance:
            adjusted_theta[top_half] += 2 * np.pi
        else:
            adjusted_theta[bottom_half] -= 2 * np.pi
        return adjusted_theta
    return None
