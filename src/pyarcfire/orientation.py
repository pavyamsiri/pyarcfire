"""Generates an orientation field from an image.

The algorithms used here are adapted from:
    1. Inferring Galaxy Morphology Through Texture Analysis (K. Au 2006).
    2. Automated Quantification of Arbitrary Arm-Segment Structure in Spiral Galaxies (D. Davis 2014).
and from the SpArcFiRe code [https://github.com/waynebhayes/SpArcFiRe]
"""

from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

import numpy as np
from scipy import signal
from skimage import transform

from pyarcfire.assert_utils import (
    verify_data_can_be_shrunk_orientation,
    verify_data_is_2d,
    verify_data_is_normalized,
)

from .debug_utils import benchmark

if TYPE_CHECKING:
    import optype as op

    from pyarcfire._typing import AnyReal

_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any])
_Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_SCT]]
_Array3D: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[_SCT]]
_Array2D_f64: TypeAlias = _Array2D[np.float64]
_Array3D_f64: TypeAlias = _Array3D[np.float64]


log: logging.Logger = logging.getLogger(__name__)


class OrientationField:
    """The orientation field of an image.

    Each pixel in the image is given a corresponding orientation field strength and direction,
    dependent on how aligned it is with nearby pixels.
    """

    DENOISE_SUBTRACT_AMOUNT: float = float(np.cos(np.pi / 4))

    def __init__(self, field: _Array3D[_SCT_f]) -> None:
        """Initiliases an orientation field from an array.

        Parameters
        ----------
        field : Array3D[F]
            The orientation field array. This is a 3D array of size MxNx2.

        """
        if len(field.shape) != 3 or field.shape[2] != 2:
            msg = "The array shape must be MxNx2."
            raise ValueError(msg)
        if field.shape[0] % 2 != 0 or field.shape[1] % 2 != 0:
            msg = "The dimensions of an OrientationField must both be even."
            raise ValueError(msg)
        self._field: _Array3D_f64 = field.astype(np.float64)

    @staticmethod
    def from_cartesian(
        x: _Array2D[_SCT_f],
        y: _Array2D[_SCT_f],
    ) -> OrientationField:
        """Create an orientation field given the x and y components of the orientation field.

        Parameters
        ----------
        x : Array2D[F]
            The x-component of the orientation field.
        y : Array2D[F]
            The y-component of the orientation field.

        Returns
        -------
        field : OrientationField
            The orientation field.

        """
        field = np.zeros((x.shape[0], x.shape[1], 2), dtype=x.dtype)
        field[:, :, 0] = x
        field[:, :, 1] = y
        return OrientationField(field)

    @staticmethod
    def from_polar(
        strengths: _Array2D[_SCT_f],
        directions: _Array2D[_SCT_f],
    ) -> OrientationField:
        """Create an orientation field given orientation strengths and directions.

        Parameters
        ----------
        strengths : Array2D[F]
            A scalar array of orientation field strengths.
        directions: Array2D[F]
            A 2D vector array of orientation field directions.

        Returns
        -------
        field : OrientationField
            The orientation field.

        """
        x = strengths * np.cos(directions)
        y = strengths * np.sin(directions)
        return OrientationField.from_cartesian(x, y)

    def __str__(self) -> str:
        """Return a string representation of OrientationField."""
        return f"OrientationField(num_rows={self.num_rows}, num_columns={self.num_columns})"

    @property
    def num_cells(self) -> int:
        """int: The total number of grid cells."""
        return self.num_rows * self.num_columns

    @property
    def num_rows(self) -> int:
        """int: The number of rows."""
        return self._field.shape[0]

    @property
    def num_columns(self) -> int:
        """int: The number of columns."""
        return self._field.shape[1]

    @property
    def shape(self) -> tuple[int, int, int]:
        """tuple[int, int, int]: The shape of the underlying array."""
        return (self.num_rows, self.num_columns, 2)

    @property
    def field(self) -> _Array3D_f64:
        """Array2D[f64]: The underlying field array."""
        return self._field

    @property
    def x(self) -> _Array2D_f64:
        """Array2D[f64]: The x-component of the orientation."""
        return self._field[:, :, 0]

    @property
    def y(self) -> _Array2D_f64:
        """Array2D[f64]: The y-component of the orientation."""
        return self._field[:, :, 1]

    def get_strengths(self) -> _Array2D_f64:
        """Get the orientation strength of each cell.

        Returns
        -------
        strengths : Array2D[f64]
            The orientation strength as an array.

        """
        return np.sqrt(np.square(self.x) + np.square(self.y)).astype(np.float64)

    def get_directions(self) -> _Array2D_f64:
        """Get the orientation direction of each cell given as angles in the range [0, pi).

        Returns
        -------
        directions : Array2D[f64]
            The orientation directions in angles in the range [0, pi).

        """
        return np.arctan2(self.y, self.x) % np.pi

    def resize(self, new_width: op.CanIndex, new_height: op.CanIndex) -> OrientationField:
        """Return the orientation field resized via interpolation.

        Parameters
        ----------
        new_width : int
            The new width of the field.
        new_height : int
            The new height of the field.

        Returns
        -------
        OrientationField
            The resized field.

        """
        return OrientationField(cast(_Array3D_f64, transform.resize(self.field, (new_height, new_width)).astype(np.float64)))

    @staticmethod
    def merge(
        coarser_field: OrientationField,
        finer_field: OrientationField,
    ) -> OrientationField:
        """Merge two orientation fields together. The first field must be of a lower resolution than the second field.

        The resultant field is the same resolution as that of the finer field.

        Parameters
        ----------
        coarser_field : OrientationField
            The coarse field to merge.
        finer_field : OrientationField
            The fine field to merge.

        Returns
        -------
        merged_field : OrientationField
            The merged field.

        """
        is_finer_higher_resolution: bool = (
            coarser_field.num_rows < finer_field.num_rows and coarser_field.num_columns < finer_field.num_columns
        )
        if not is_finer_higher_resolution:
            msg = "The finer field must be a higher resolution than the coarser field."
            raise ValueError(msg)

        # Upscale coarse field to have same resolution as finer field
        resized_coarse_field = coarser_field.resize(
            finer_field.num_columns,
            finer_field.num_rows,
        )

        resized_coarse_strengths = resized_coarse_field.get_strengths()
        fine_field_strengths = finer_field.get_strengths()

        # G -> Sf / (Sf + Sc)
        gains = fine_field_strengths
        denominator = fine_field_strengths + resized_coarse_strengths
        gains[denominator != 0] /= denominator[denominator != 0]

        # Vf' -> Vc + Sf / (Sf + Sc) * (Vf - Vc)
        return resized_coarse_field.add(
            finer_field.subtract(resized_coarse_field).scalar_field_multiply(gains),
        )

    def scalar_field_multiply(
        self,
        scalar_field: _Array2D[_SCT_f],
    ) -> OrientationField:
        """Return the orientation field mulitplied by a scalar field.

        Parameters
        ----------
        scalar_field : Array2D[F]
            The scalar field to multiply with.

        Returns
        -------
        OrientationField
            The multiplied orientation field.

        """
        if len(scalar_field.shape) != 2:
            msg = "The scalar field must be a 2D array."
            raise ValueError(msg)
        if scalar_field.shape[0] != self.num_rows or scalar_field.shape[1] != self.num_columns:
            msg = "The scalar field must have the same dimensions as the OrientationField."
            raise ValueError(msg)

        # Multiply each component
        result = self.field
        result[:, :, 0] *= scalar_field
        result[:, :, 1] *= scalar_field
        return OrientationField(result)

    def add(self, other: OrientationField) -> OrientationField:
        """Return the orientation field added with another field.

        Parameters
        ----------
        other : OrientationField
            The field to add.

        Returns
        -------
        OrientationField
            The sum of the two fields.

        """
        vector_sum, vector_difference, sum_greater = OrientationField._prepare_sum(
            self,
            other,
        )
        result = np.zeros_like(vector_sum)
        result[sum_greater] = vector_sum[sum_greater]
        result[~sum_greater] = vector_difference[~sum_greater]
        return OrientationField(result)

    def subtract(self, other: OrientationField) -> OrientationField:
        """Return the orientation field subtracted by another field.

        Parameters
        ----------
        other : OrientationField
            The field to subtract.

        Returns
        -------
        OrientationField
            The difference of the two fields.

        """
        vector_sum, vector_difference, sum_greater = OrientationField._prepare_sum(
            self,
            other,
        )
        result = np.zeros_like(vector_sum)
        result[~sum_greater] = vector_sum[~sum_greater]
        result[sum_greater] = vector_difference[sum_greater]
        return OrientationField(result)

    @staticmethod
    def _prepare_sum(
        left: OrientationField,
        right: OrientationField,
    ) -> tuple[_Array3D_f64, _Array3D_f64, _Array2D[np.bool_]]:
        """Calculate the necessary components to perform an orientation field sum or difference.

        Parameters
        ----------
        left : OrientationField
            The left hand side of the sum/difference.
        right : OrientationField
            The right hand side of the sum/difference.

        Returns
        -------
        vector_sum : Array3D[F]
            The vector sum of the two fields.
        vector_difference : Array3D[F]
            The vector difference of the two fields.
        sum_greater : Array2D[bool]
            The mask where true indicates that the norm of vector sum is greater than the
            norm of vector difference.

        """
        negative_vertical = right.y < 0
        b = right.field
        b[negative_vertical, 0] = -b[negative_vertical, 0]
        b[negative_vertical, 1] = -b[negative_vertical, 1]
        # Vector sum
        vector_sum = left.field + b
        vector_sum_lengths = np.sqrt(np.sum(np.square(vector_sum), axis=2))

        # Vector difference
        vector_difference = left.field - b
        vector_difference_lengths = np.sqrt(
            np.sum(np.square(vector_difference), axis=2),
        )

        sum_greater = vector_sum_lengths > vector_difference_lengths
        sum_greater = np.repeat(sum_greater[:, :, np.newaxis], 2, axis=2)
        return (vector_sum, vector_difference, sum_greater)

    def denoise(self, neighbour_distance: op.CanInt) -> OrientationField:
        """Denoise orientation field.

        Parameters
        ----------
        neighbour_distance : int, optional
            The distance between a pixel and its four cardinal neighbours.

        Returns
        -------
        OrientationField
            The denoised field.

        """
        neighbour_distance = int(neighbour_distance)

        # Allocate new field
        denoised = np.zeros_like(self.field)

        neighbour_arrays: list[_Array3D_f64] = [
            np.roll(self.field, -neighbour_distance, axis=1),
            np.roll(self.field, neighbour_distance, axis=1),
            np.roll(self.field, -neighbour_distance, axis=0),
            np.roll(self.field, neighbour_distance, axis=0),
        ]
        neighbour_masks = np.ones((self.num_rows, self.num_columns, 4), dtype=np.bool_)
        neighbour_masks[:, -neighbour_distance:, 0] = False
        neighbour_masks[:, :neighbour_distance, 1] = False
        neighbour_masks[-neighbour_distance:, :, 2] = False
        neighbour_masks[:neighbour_distance, :, 3] = False
        vector_norm = np.linalg.norm(self.field, axis=2)
        neighbour_strengths = np.zeros((self.num_rows, self.num_columns, 4))
        for neighbour_idx, neighbour in enumerate(neighbour_arrays):
            product_norm = vector_norm * np.linalg.norm(neighbour, axis=2)
            current_strength = np.maximum(
                (neighbour * self.field).sum(axis=2) - OrientationField.DENOISE_SUBTRACT_AMOUNT,
                0,
            )
            current_strength[product_norm != 0] /= product_norm[product_norm != 0]
            current_strength[np.isnan(current_strength)] = 0
            current_strength[~neighbour_masks[:, :, neighbour_idx]] = np.nan
            neighbour_strengths[:, :, neighbour_idx] = current_strength
        median_strength = np.nanmedian(neighbour_strengths, axis=2)
        combined_mask = vector_norm != 0
        denoised[:, :, 0][combined_mask] = (
            self.field[:, :, 0][combined_mask] / vector_norm[combined_mask] * median_strength[combined_mask]
        )
        denoised[:, :, 1][combined_mask] = (
            self.field[:, :, 1][combined_mask] / vector_norm[combined_mask] * median_strength[combined_mask]
        )

        return OrientationField(denoised)

    def count_nonzero(self) -> int:
        """Count the number of non-zero elements.

        Returns
        -------
        num_nonzero_elements : int
            The number of non-zero elements.

        """
        return int(np.count_nonzero(self.get_strengths()))


@benchmark
def generate_orientation_fields(
    image: _Array2D[_SCT_f],
    *,
    neighbour_distance: int,
    kernel_radius: int,
    num_orientation_field_levels: int,
) -> OrientationField:
    """Generate an orientation field for the given image.

    This includes a merging step and a denoising step.

    Parameters
    ----------
    image : Array2D[F]
        The image to generate an orientation field of.
    neighbour_distance : int
        The distance in pixels between a cell and its neighbour.
        Used when denoising.
    kernel_radius : int
        The radius of the orientation filter kernel in pixels.
    num_orientation_field_levels : int
        The number of orientation field levels to create and then join.

    Returns
    -------
    OrientationField
        The orientation field of the given image.

    """
    verify_data_is_2d(image)
    verify_data_is_normalized(image)
    verify_data_can_be_shrunk_orientation(
        image,
        num_orientation_field_levels=num_orientation_field_levels,
    )

    # Neighbour distance must be smaller than both dimensions
    if neighbour_distance >= image.shape[0] or neighbour_distance >= image.shape[1]:
        msg = "The neighbour distance must be strictly less than both of the image dimensions."
        raise ValueError(msg)

    # Generate all the different orientation field levels
    orientation_field_levels: list[OrientationField] = []
    for idx in range(num_orientation_field_levels):
        # Resize
        scale_factor: float = 1 / 2 ** (num_orientation_field_levels - idx - 1)
        resized_image = cast(_Array2D[_SCT_f], transform.rescale(image, scale_factor).astype(image.dtype))

        # Generate
        current_level = _generate_raw_orientation_field(
            resized_image,
            kernel_radius,
        )
        orientation_field_levels.append(current_level)

    # Merge orientation fields
    merged_field: OrientationField = reduce(
        lambda x, y: OrientationField.merge(x, y),
        orientation_field_levels,
    )

    # Denoise
    denoised_field = merged_field.denoise(
        neighbour_distance=neighbour_distance,
    )
    num_nonzero_elements = denoised_field.count_nonzero()

    log.info(
        "[green]DIAGNOST[/green]: Orientation field has %d non-zero elements.",
        num_nonzero_elements,
    )

    return denoised_field


def _generate_raw_orientation_field(
    image: _Array2D[_SCT_f],
    kernel_radius: op.CanInt,
) -> OrientationField:
    """Generate an orientation field for the given image with no merging or denoising steps.

    Parameters
    ----------
    image : Array2D[_SCT_f]
        The image to generate an orientation field of.
    kernel_radius : int
        The radius of the orientation filter kernel in pixels.

    Returns
    -------
    field : OrientationField
        The orientation field of the image.

    """
    # Filter the images using the orientation filters
    filtered_images = _generate_orientation_filtered_images(image, kernel_radius)

    # Clip negative values
    filtered_images[filtered_images < 0] = 0

    # Construct weighted sum of filtered images
    weights = np.square(filtered_images, dtype=np.complex128)
    for idx in range(9):
        weights[:, :, idx] = weights[:, :, idx] * np.exp(1j * 2 * idx * np.pi / 9)
    weighted_sum = np.sum(weights, axis=2)

    # Magnitude
    strengths = np.asarray(np.abs(weighted_sum), dtype=np.float64)
    # Angle
    angles = np.asarray(np.angle(weighted_sum)) / 2  # pyright:ignore[reportUnknownArgumentType]

    # Construct the orientation field
    return OrientationField.from_polar(strengths, angles.astype(strengths.dtype))


def _generate_orientation_filtered_images(
    image: _Array2D[_SCT_f],
    kernel_radius: op.CanInt,
) -> _Array3D[_SCT_f]:
    """Convolve the given image with 9 orientation filters and return all results.

    Parameters
    ----------
    image : Array2D[F]
        The 2D image to filter.
    kernel_radius : int
        The radius of the orientation filter kernel in pixels.

    Returns
    -------
    filtered_images : Array3D[F]
        The 3D array of the image filtered through the 9 orientation filters.

    """
    filtered_images = np.zeros((image.shape[0], image.shape[1], 9), dtype=image.dtype)
    for idx in range(9):
        angle = (idx * np.pi) / 9
        orientation_filter = _generate_orientation_filter_kernel(angle, kernel_radius)
        filtered_images[:, :, idx] = signal.convolve2d(
            image,
            orientation_filter,
            mode="same",
        )
    return filtered_images


def _generate_orientation_filter_kernel(
    theta: AnyReal,
    radius: op.CanInt,
) -> _Array2D_f64:
    """Generate orientation filter kernel.

    Technically this is 1D Ricker wavelet filter extended in 2D along an angle theta such that the filter response
    is strongest for that angle.

    Parameters
    ----------
    theta : float
        The angle in radians at which the filter is strongest.
    radius : int
        The radius of the kernel in pixels.

    Returns
    -------
    kernel : Array2D[f64]
        The filter kernel of size [2 * radius + 1, 2 * radius + 1]

    """
    radius = int(radius)

    # Mesh size in pixels
    num_pixels = int(2 * np.ceil(radius) + 1)
    # Sample from pixel centres
    max_value = np.pi * 2 * radius / (2 * radius + 1)
    # Sample from [-pi, pi] to create the filter
    x, y = np.meshgrid(
        np.linspace(-max_value, max_value, num_pixels),
        np.linspace(-max_value, max_value, num_pixels),
    )
    # Rotate by theta
    rotated_x = x * np.cos(theta) - y * np.sin(theta)
    rotated_y = x * np.sin(theta) + y * np.cos(theta)
    rotated_x_squared = np.square(rotated_x)
    rotated_y_squared = np.square(rotated_y)
    # Use Mexican hat wavelet as kernel
    wavelet = (1 - rotated_x_squared) * np.exp(-1 / 2 * rotated_x_squared)

    # Attenuate using a Gaussian function with sigma = max_value / 2
    sigma = max_value / 2
    gaussian_window = np.exp((-1 / (2 * (sigma**2))) * rotated_y_squared)

    # Construct filter
    kernel = wavelet * gaussian_window
    # Normalise
    kernel /= np.sqrt(np.sum(np.square(kernel)))
    return kernel.astype(np.float64)
