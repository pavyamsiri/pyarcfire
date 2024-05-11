"""Generates an orientation field from an image.
The algorithms used here are adapted from:
    1. Inferring Galaxy Morphology Through Texture Analysis (K. Au 2006).
    2. Automated Quantification of Arbitrary Arm-Segment Structure in Spiral Galaxies (D. Davis 2014).
and from the SpArcFiRe code [https://github.com/waynebhayes/SpArcFiRe]
"""

from __future__ import annotations

# Standard libraries
from functools import reduce

# External libraries
import numpy as np
from numpy import typing as npt
from rich import progress as rprogress
from scipy import signal
from skimage import transform

# Internal libraries
from .definitions import ImageArray, ImageArraySequence


class OrientationField:
    def __init__(self, field: ImageArray):
        assert (
            len(field.shape) == 3 and field.shape[2] == 2
        ), "OrientationFields are MxNx2 arrays."
        assert (
            field.shape[0] % 2 == 0
        ), "The height of an OrientationField must be even!"
        assert field.shape[1] % 2 == 0, "The width of an OrientationField must be even!"
        self._field: ImageArray = field

    @staticmethod
    def from_cartesian(x: ImageArray, y: ImageArray) -> OrientationField:
        field: ImageArray = np.zeros((x.shape[0], x.shape[1], 2))
        field[:, :, 0] = x
        field[:, :, 1] = y
        return OrientationField(field)

    @staticmethod
    def from_polar(magnitudes: ImageArray, angles: ImageArray) -> OrientationField:
        x = magnitudes * np.cos(angles)
        y = magnitudes * np.sin(angles)
        return OrientationField.from_cartesian(x, y)

    def __str__(self) -> str:
        return f"OrientationField(num_rows={self.num_rows}, num_columns={self.num_columns})"

    @property
    def num_cells(self) -> int:
        return self.num_rows * self.num_columns

    @property
    def num_rows(self) -> int:
        return self._field.shape[0]

    @property
    def num_columns(self) -> int:
        return self._field.shape[1]

    @property
    def shape(self) -> tuple[int, int, int]:
        assert self.field.shape[2] == 2
        return (self.num_rows, self.num_columns, self.field.shape[2])

    @property
    def field(self) -> ImageArray:
        return self._field

    @property
    def x(self) -> ImageArray:
        return self._field[:, :, 0]

    @property
    def y(self) -> ImageArray:
        return self._field[:, :, 1]

    def get_vector_at(
        self, row_index: int, column_index: int
    ) -> npt.NDArray[np.floating]:
        return self.field[row_index, column_index, :]

    def get_strengths(self) -> ImageArray:
        strengths = np.sqrt(np.square(self.x) + np.square(self.y))
        return strengths

    def get_directions(self) -> ImageArray:
        directions = np.arctan2(self.y, self.x) % np.pi
        return directions

    def rescale(self, scale_factor: float) -> OrientationField:
        resized_field = transform.rescale(self.field, scale_factor, channel_axis=2)
        return OrientationField(resized_field)

    def resize(self, new_width: int, new_height: int) -> OrientationField:
        return OrientationField(transform.resize(self.field, (new_height, new_width)))

    def merge(self, other: OrientationField) -> OrientationField:
        resized_coarse_field = np.zeros(other.shape)
        # NOTE: This implicitly requires that the finer field is twice the dimensions of the coarse field
        resized_coarse_field = self.rescale(2)
        resized_coarse_strengths = resized_coarse_field.get_strengths()
        fine_field_strengths = other.get_strengths()

        gains = fine_field_strengths
        gains[fine_field_strengths != 0] /= (
            fine_field_strengths + resized_coarse_strengths
        )[fine_field_strengths != 0]
        merged_field = resized_coarse_field.add(
            other.subtract(resized_coarse_field).scalar_field_multiply(gains)
        )
        return merged_field

    def scalar_field_multiply(self, scalar_field: ImageArray) -> OrientationField:
        assert (
            len(scalar_field.shape) == 2
        ), "The scalar field must be MxN as it is scalar."
        assert (
            scalar_field.shape[0] == self.num_rows
            and scalar_field.shape[1] == self.num_columns
        ), "The scalar field must have the same height and width as the OrientationField"
        result = self.field
        result[:, :, 0] *= scalar_field
        result[:, :, 1] *= scalar_field
        return OrientationField(result)

    def add(self, other: OrientationField) -> OrientationField:
        return self._add_or_subtract(other, add=True)

    def subtract(self, other: OrientationField) -> OrientationField:
        return self._add_or_subtract(other, add=False)

    def _add_or_subtract(self, other: OrientationField, add: bool) -> OrientationField:
        negative_vertical = other.y < 0
        b = other.field
        b[negative_vertical, 0] = -b[negative_vertical, 0]
        b[negative_vertical, 1] = -b[negative_vertical, 1]
        vector_sum = self.field + b
        vector_difference = self.field - b
        vector_sum_lengths = np.sqrt(np.sum(np.square(vector_sum), axis=2))
        vector_difference_lengths = np.sqrt(
            np.sum(np.square(vector_difference), axis=2)
        )
        sum_greater = vector_sum_lengths > vector_difference_lengths
        sum_greater = np.repeat(sum_greater[:, :, np.newaxis], 2, axis=2)
        result = np.zeros_like(self.field)
        if add:
            result[sum_greater] = vector_sum[sum_greater]
            result[~sum_greater] = vector_difference[~sum_greater]
        else:
            result[~sum_greater] = vector_sum[~sum_greater]
            result[sum_greater] = vector_difference[sum_greater]
        return OrientationField(result)

    def denoise(self, neighbour_distance: int = 5) -> OrientationField:
        # Performs the orientation field de-noising described in the PhD thesis
        # "Inferring Galaxy Morphology Through Texture Analysis" (K. Au 2006).
        # INPUTS:
        #   ofld: orientation field without de-noising (but already merged from 3
        #       resolution levels if the process in the thesis is followed)
        # OUTPUTS:
        #   dofld: de-noised orientation field
        denoised = np.zeros(self.shape)

        for row_idx in rprogress.track(range(self.num_rows)):
            for column_idx in range(self.num_columns):
                current_vector = self.get_vector_at(row_idx, column_idx)
                current_vector_norm = np.linalg.norm(current_vector)
                if current_vector_norm == 0:
                    continue
                neighbour_vectors = []
                # Check that the neighbour is not past the top left corner
                for row_offset, column_offset in (
                    (-neighbour_distance, -neighbour_distance),
                    (+neighbour_distance, -neighbour_distance),
                    (-neighbour_distance, +neighbour_distance),
                    (+neighbour_distance, +neighbour_distance),
                ):
                    target_row = row_idx + row_offset
                    target_column = column_idx + column_offset
                    if target_row < 0 or target_row >= self.num_rows:
                        continue
                    if target_column < 0 or target_column >= self.num_columns:
                        continue
                    neighbour_vectors.append(
                        self.get_vector_at(target_row, target_column)
                    )
                neighbour_sims = np.zeros(len(neighbour_vectors))

                subtract_amount = np.cos(np.pi / 4)
                for idx, neighbour in enumerate(neighbour_vectors):
                    neighbour_norm = np.linalg.norm(neighbour)
                    if neighbour_norm == 0:
                        continue
                    neighbour_sims[idx] = max(
                        np.dot(neighbour, current_vector) - subtract_amount, 0
                    ) / (current_vector_norm * neighbour_norm)

                denoised[row_idx, column_idx, :] = (
                    current_vector / current_vector_norm
                ) * np.median(neighbour_sims)
        return OrientationField(denoised)


def generate_orientation_fields(
    image: ImageArray, num_orientation_field_levels: int = 3
) -> OrientationField:
    """Generates an orientation field for the given image.
    This includes a merging step and a denoising step.

    Parameters
    ----------
    image : ImageArray
        The image to generate an orientation field of.
    num_orientation_field_levels: int, optional
        The number of orientation fields to create and merge. This number must be at least 1.
        The resolution shrinks by half for each additional level. The default is 3.

    Returns
    -------
    OrientationField
        The orientation field of the given image.
    """
    assert num_orientation_field_levels >= 1, "The number of levels must be at least 1."
    # The dimensions of the image must be divisible by the largest shrink factor
    maximum_shrink_factor: int = 2 ** (num_orientation_field_levels - 1)
    assert (
        image.shape[0] % maximum_shrink_factor == 0
    ), f"Image height must be divisible by {maximum_shrink_factor}"
    assert (
        image.shape[1] % maximum_shrink_factor == 0
    ), f"Image width must be divisible by {maximum_shrink_factor}"

    # Generate all the different orientation field levels
    orientation_field_levels: list[OrientationField] = []
    for idx in range(num_orientation_field_levels):
        # Resize
        scale_factor: float = 1 / 2 ** (num_orientation_field_levels - idx - 1)
        resized_image = transform.rescale(image, scale_factor)

        # Generate
        current_level = generate_single_orientation_field_level(resized_image)
        orientation_field_levels.append(current_level)

    # Merge orientation fields
    merged_field: OrientationField = reduce(
        lambda x, y: x.merge(y),
        orientation_field_levels,
    )

    # Denoise
    denoised_field = merged_field.denoise()
    return denoised_field


def generate_single_orientation_field_level(
    image: ImageArray,
) -> OrientationField:
    """Generates an orientation field for the given image with no merging
    or denoising steps.

    Parameters
    ----------
    image : ImageArray
        The image to generate an orientation field of.

    Returns
    -------
    field : OrientationField
        The orientation field of the image.

    """
    # Filter the images using the orientation filters
    filtered_images = generate_orientation_filtered_images(image)

    # Clip negative values
    filtered_images[filtered_images < 0] = 0

    # Construct weighted sum of filtered images
    weights = np.square(filtered_images, dtype=np.complex64)
    for idx in range(9):
        weights[:, :, idx] = weights[:, :, idx] * np.exp(1j * 2 * idx * np.pi / 9)
    weighted_sum = np.sum(weights, axis=2)

    # Magnitude
    strengths = np.abs(weighted_sum)
    # Angle
    angles = np.angle(weighted_sum) / 2

    # Construct the orientation field
    field = OrientationField.from_polar(strengths, angles)
    return field


def generate_orientation_filtered_images(
    image: ImageArray,
) -> ImageArraySequence:
    """Convolve the given image with 9 orientation filters and return all results.

    Parameters
    ----------
    image : ImageArray
        The 2D image to filter.

    Returns
    -------
    filtered_images : ImageArraySequence
        The 3D array of the image filtered through the 9 orientation filters.
    """
    filtered_images = np.zeros((image.shape[0], image.shape[1], 9))
    for idx in range(9):
        # TODO: The 9 filters are always the same so we should precompute this
        angle = (idx * np.pi) / 9
        orientation_filter = generate_orientation_filter_kernel(angle)
        filtered_images[:, :, idx] = signal.convolve2d(
            image, orientation_filter, mode="same"
        )
    return filtered_images


def generate_orientation_filter_kernel(theta: float, radius: int = 5) -> ImageArray:
    """The filter is a 1D Ricker wavelet filter extended in 2D along an angle theta, such
    that the filter response is strongest for that angle.

    Parameters
    ----------
    theta : float
        The angle in radians at which the filter is strongest.
    radius : int, optional
        The radius of the kernel in pixels. Default is 5 pixels.

    Returns
    -------
    kernel : ImageArray
        The filter kernel of size [2 * radius + 1, 2 * radius + 1]
    """
    assert radius > 0, f"Radius must be positive but it is instead {radius}"

    # Mesh size in pixels
    num_pixels: int = int(2 * np.ceil(radius) + 1)
    # Sample from pixel centres
    max_value: float = np.pi * 2 * radius / (2 * radius + 1)
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
    return kernel
