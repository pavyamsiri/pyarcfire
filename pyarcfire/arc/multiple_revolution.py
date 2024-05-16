# Standard libraries
import functools
import logging
from typing import Sequence

# External libraries
import numpy as np
from numpy import typing as npt
from scipy import ndimage
from scipy import optimize
import skimage
import skimage.measure

# Internal libraries
from pyarcfire.definitions import ImageArray
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


def fit_spiral_to_image_multiple_revolution(
    image: ImageArray,
    initial_pitch_angle: float = 0,
) -> LogSpiralFitResult:
    # Convert to polar coordinates
    radii, theta, weights = _get_polar_coordinates(image)

    # Check if the cluster revolves more than once
    bad_bounds, _, _, _ = _calculate_bounds(theta)

    # Gap in theta is large enough to not need multiple revolutions
    if not bad_bounds:
        need_multiple_revolutions = False
    # The cluster contains the origin or is closed around centre
    elif __cluster_has_no_endpoints_or_contains_origin(image):
        need_multiple_revolutions = False
    else:
        need_multiple_revolutions = True
        log.debug("IMPLEMENT idInnerOuterSpiral")
        identify_inner_and_outer_spiral(image, shrink_amount=5)
        # [isInner, gapFail] = idInnerOuterSpiral(img, ctrR, ctrC, plotFlag);
        # nInner = sum(isInner);
        # failed2rev = gapFail;
        # if gapFail || (nInner == 0) || (nInner == numel(isInner))
        #     allowArcBeyond2pi = false;
        # else
        #     thAdj = removeThetaDiscontFor2rev(theta, img, isInner);
        #     # TODO: make theta and thAdj one variable after making sure that
        #     # bounds calculation doesn"t need a different (unaltered) value
        #     theta = thAdj;
    log.debug(f"Need multiple revolutions = {need_multiple_revolutions}")

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


def __cluster_has_no_endpoints_or_contains_origin(
    image: ImageArray, max_half_gap_fill_for_undefined_bounds: int = 3
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
    image: ImageArray, shrink_amount: int, max_diagonal_distance: float = 1.5
) -> None:
    num_radii = int(np.ceil(max((image.shape[0], image.shape[1])) / 2))
    num_theta: int = 360

    min_acceptable_length = 5 * np.ceil(num_theta / 360)
    # Find theta bins which contain only a single revolution
    can_be_single_revolution = __find_single_revolution_regions(
        image, num_radii, num_theta, min_acceptable_length, shrink_amount
    )

    # Find the start and end of each region
    single_revolution_differences = np.diff(can_be_single_revolution.astype(np.float32))
    start_indices: npt.NDArray[np.int32] = (
        single_revolution_differences == 1
    ).nonzero()[0].astype(np.int32) + 1
    end_indices: npt.NDArray[np.int32] = (
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
            # isInner = true(size(find(img)))
            # gapFail = true
            assert False

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

    wrap_lengths: tuple[int, int, int] | None = None
    if has_wrapped:
        # Last continuous single revolution region is at the beginning, but doesn"t
        # actually wrap around
        if wrap_start is None:
            assert (
                wrap_end is not None
            ), "Other branch must be executed as this has wrapped"
            start_indices = np.insert(start_indices, 0, 0)
            end_indices = np.insert(end_indices, 0, wrap_end)
            has_wrapped = False
        # Last continuous single revolution region is at the end, but doesn"t
        # actually wrap around
        elif wrap_end is None:
            assert (
                wrap_start is not None
            ), "Other branch must be executed as this has wrapped"
            start_indices = np.insert(start_indices, 0, wrap_start)
            end_indices = np.insert(end_indices, 0, len(can_be_single_revolution) - 1)
            has_wrapped = False
        # Wrap does happen
        else:
            # Length of region from start to edge
            wrap_start_length = len(can_be_single_revolution) - wrap_start + 1
            # Length of region from end to end
            wrap_end_length = wrap_end
            wrap_lengths = (
                wrap_start_length,
                wrap_end_length,
                wrap_start_length + wrap_end_length,
            )

    theta_bin_values = np.arange(1, num_theta) * 2 * np.pi / num_theta
    row_indices, column_indices = image.nonzero()
    point_index_to_image_index = np.ravel_multi_index(
        (row_indices, column_indices), image.shape
    )
    image_index_to_point_index = np.zeros_like(image, dtype=np.int32)
    image_index_to_point_index[image > 0] = np.arange(len(row_indices))
    log.debug(f"Point idx to image idx = {point_index_to_image_index.shape}")
    _, theta, _ = _get_polar_coordinates(image)

    start_indices = np.array([])
    end_indices = np.array([])
    region_lengths = end_indices - start_indices + 1
    max_region_length: int | None = (
        region_lengths.max() if len(region_lengths) > 0 else None
    )
    only_wrap_exists: bool = max_region_length is None
    wrap_larger_than_all_regions: bool = False
    if wrap_lengths is not None and max_region_length is not None:
        wrap_larger_than_all_regions = wrap_lengths[2] > max_region_length
    if only_wrap_exists or wrap_larger_than_all_regions:
        # Only wrap exists
        assert wrap_lengths is not None
        wrap_start_length, wrap_end_length, wrap_length = wrap_lengths
        max_length = wrap_length
        wrap_mid_length = np.round(wrap_length / 2)
        theta_start = theta_bin_values[wrap_start]
        theta_end = theta_bin_values[wrap_end]
        inner_region = np.logical_or(theta >= theta_start, theta < theta_end)
        if wrap_start_length >= wrap_mid_length:
            split_theta_idx = int(wrap_start + wrap_mid_length - 1)
        else:
            split_theta_idx = int(wrap_mid_length - wrap_start_length - 1)
        split_theta = theta_bin_values[split_theta_idx]
        first_inner_region = np.logical_and(
            inner_region, np.logical_and(theta >= theta_start, theta < split_theta)
        )
        second_inner_region = np.logical_and(inner_region, ~first_inner_region)
    else:
        assert max_region_length is not None
        max_length = max_region_length
        max_index = region_lengths.argmax()
        theta_start = theta_bin_values[start_indices[max_index]]
        theta_end = theta_bin_values[start_indices[max_index]]
        split_theta = theta_bin_values[
            round((start_indices[max_index] + end_indices[max_index]) / 2)
        ]
        first_inner_region = np.logical_and(theta >= theta_start, theta < split_theta)
        second_inner_region = np.logical_and(theta >= split_theta, theta < theta_end)

    if max_length < min_acceptable_length:
        log.warn(
            f"Warning:idInnerOuterSpiral:longest sgl-rev region length ({max_length}) is below the minimum length {min_acceptable_length}"
        )
        assert False
        # isInner = true(size(find(img)));
        # gapFail = true;

    first_region_mask = np.zeros_like(image, dtype=np.bool_)
    first_region_mask[
        row_indices[first_inner_region], column_indices[first_inner_region]
    ] = True
    second_region_mask = np.zeros_like(image, dtype=np.bool_)
    second_region_mask[
        row_indices[second_inner_region], column_indices[second_inner_region]
    ] = True
    non_region_mask = np.zeros_like(image, dtype=np.bool_)
    non_region_mask[np.logical_and(~first_region_mask, ~second_region_mask)] = True

    first_region_distance = ndimage.distance_transform_edt(
        ~first_region_mask, return_distances=True
    )
    assert isinstance(first_region_distance, np.ndarray)
    second_region_distance = ndimage.distance_transform_edt(
        ~second_region_mask, return_distances=True
    )
    assert isinstance(second_region_distance, np.ndarray)
    connected_components = skimage.measure.label(non_region_mask)
    assert isinstance(connected_components, np.ndarray)
    labels = set(list(connected_components.flatten()))
    for label in labels:
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
            first_inner_region[point_indices] = True
        elif second_distance < max_diagonal_distance:
            second_inner_region[point_indices] = True

    # Use updated regions
    first_region_mask = np.zeros_like(image, dtype=np.bool_)
    first_region_mask[
        row_indices[first_inner_region], column_indices[first_inner_region]
    ] = True
    second_region_mask = np.zeros_like(image, dtype=np.bool_)
    second_region_mask[
        row_indices[second_inner_region], column_indices[second_inner_region]
    ] = True
    non_region_mask = np.zeros_like(image, dtype=np.bool_)
    non_region_mask[np.logical_and(~first_region_mask, ~second_region_mask)] = True


def __find_single_revolution_regions(
    image: ImageArray,
    num_radii: int,
    num_theta: int,
    min_acceptable_length: int,
    shrink_amount: int,
) -> npt.NDArray[np.bool_]:
    assert shrink_amount <= min_acceptable_length
    polar_image = np.flip(
        __image_transform_from_cartesian_to_polar(image, num_radii, num_theta), axis=1
    )
    polar_image = np.nan_to_num(polar_image, nan=0)

    dilated_polar_image = ndimage.binary_dilation(
        polar_image, structure=np.ones((3, 3))
    )

    # Pad columns with zeros
    theta_diff = np.diff(dilated_polar_image, prepend=0, append=0, axis=0)
    # Theta bins which only have a single start and end point
    # i.e. ray from centre only hits the cluster once and not multiple times
    can_be_single_revolution = np.logical_and(
        np.sum(theta_diff == -1, axis=0) == 1,
        np.sum(theta_diff == 1, axis=0) == 1,
    )
    # Radial index for every point in the polar image
    radial_locations = np.tile(np.arange(1, num_radii + 1), (num_theta, 1)).T

    polar_image_for_min = dilated_polar_image.astype(np.float32)
    polar_image_for_min[polar_image_for_min == 0] = np.inf
    min_locations = np.min(polar_image_for_min * radial_locations, axis=0)
    min_locations[np.isinf(min_locations)] = 0
    max_locations = np.max(
        dilated_polar_image.astype(np.float32) * radial_locations, axis=0
    )
    neighbour_max_location_left = np.roll(max_locations, 1)
    neighbour_max_location_right = np.roll(max_locations, -1)
    neighbour_min_location_left = np.roll(min_locations, 1)
    neighbour_min_location_right = np.roll(min_locations, -1)
    conditions: Sequence[npt.NDArray[np.bool_]] = (
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
    image: ImageArray, num_radii: int, num_theta: int
) -> ImageArray:
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
