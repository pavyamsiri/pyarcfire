# Standard libraries
from dataclasses import dataclass
import logging

# External libraries
import numpy as np
from scipy import optimize
from scipy import ndimage

# Internal libraries
from .definitions import FloatArray1D, ImageArray

log = logging.getLogger(__name__)


@dataclass
class LogSpiralFitResult:
    offset: float
    pitch_angle: float
    initial_radius: float
    arc_bounds: tuple[float, float]
    total_error: float
    errors: FloatArray1D

    def calculate_cartesian_coordinates(
        self, num_points: int
    ) -> tuple[FloatArray1D, FloatArray1D]:
        start_angle = self.offset
        end_angle = start_angle + self.arc_bounds[1]

        theta = np.linspace(start_angle, end_angle, num_points)
        radii = log_spiral(theta, self.offset, self.pitch_angle, self.initial_radius)
        x = radii * np.cos(theta)
        y = radii * np.sin(theta)
        return (x, y)


def fit_spiral_to_image(
    image: ImageArray,
    initial_pitch_angle: float = 0,
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

    # Check if the cluster revolves more than once
    bad_bounds, _, _, _ = _calculate_bounds(theta)

    # Gap in theta is large enough to not need multiple revolutions
    if not bad_bounds:
        need_multiple_revolutions = False
    # The cluster contains the origin or is closed around centre
    elif cluster_has_no_endpoints_or_contains_origin(image):
        need_multiple_revolutions = False
    else:
        need_multiple_revolutions = True
        log.debug("IMPLEMENT idInnerOuterSpiral")
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

    bad_bounds, (lower_bound, upper_bound), rotation_amount, max_gap_size = (
        _calculate_bounds(theta)
    )
    theta = (theta - rotation_amount) % (2 * np.pi)
    if bad_bounds:
        log.warn(f"Bad bounds! Gap size = {max_gap_size}")
        initial_offset = 0
        pitch_angle = 0
    else:
        initial_offset = (lower_bound + upper_bound) / 2
        res = optimize.least_squares(
            calculate_log_spiral_error_from_pitch_angle,
            x0=initial_pitch_angle,
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

    # ZERO THETA START
    theta = (theta - initial_offset) % (2 * np.pi) + initial_offset
    offset_shift = arc_start
    new_offset = float(initial_offset + offset_shift)
    theta += new_offset - np.min(theta)
    initial_radius = calculate_best_initial_radius(
        radii, theta, weights, new_offset, pitch_angle
    )
    offset = new_offset
    arc_bounds = (
        float(arc_bounds[0] - arc_start),
        float(arc_bounds[1] - arc_start),
    )
    residuals = calculate_log_spiral_residual_vector(
        radii, theta, weights, offset, pitch_angle, initial_radius
    )
    new_error = np.sum(np.square(residuals))
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
) -> tuple[bool, tuple[float, float], float, float]:
    """Calculates optimisation bounds for the theta offset.
    If the bounds would cross the polar axis, then the bounds must be
    split into two. To avoid this, the theta values can be rotated so that
    the bounds can be expressed as a single bound.

    Parameters
    ----------
    theta : FloatArray1D
        The theta values of the cluster.

    Returns
    -------
    bad_bounds : bool
        This flag is true if the cluster covers a substantial portion
        of the unit circle.
    bounds : tuple[float, float]
        The lower bound and the upper bound of theta offset.
    rotation_amount : float
        The amount to rotate the theta values so that the bounds can be singular.
    max_gap_size : float
        The largest gap between nearby theta values.
    """
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
    BAD_BOUNDS_THRESHOLD: float = 0.1
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
    bad_bounds = max_gap_size <= BAD_BOUNDS_THRESHOLD
    return (bad_bounds, bounds, rotation_amount, max_gap_size)


def _calculate_angle_distance(
    from_angle: FloatArray1D, to_angle: FloatArray1D
) -> FloatArray1D:
    is_wrapping = from_angle > to_angle
    distance = to_angle - from_angle
    distance[is_wrapping] += 2 * np.pi
    return distance


def cluster_has_no_endpoints_or_contains_origin(
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
    return is_hole_or_cluster[central_row, central_column] > 0
