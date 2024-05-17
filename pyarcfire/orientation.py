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
from scipy import signal
from skimage import transform

# Internal libraries
from .definitions import ImageFloatArray, ImageArraySequence


class OrientationField:
    """The orientation field of an image.
    Each pixel in the image is given a corresponding orientation field strength and direction,
    dependent on how aligned it is with nearby pixels.
    """

    def __init__(self, field: ImageFloatArray) -> None:
        """Initiliases an orientation field from an array.

        Parameters
        ----------
        field : ImageArray
            The orientation field array. This is a 3D array of size MxNx2.
        """
        assert (
            len(field.shape) == 3 and field.shape[2] == 2
        ), "OrientationFields are MxNx2 arrays."
        assert (
            field.shape[0] % 2 == 0
        ), "The height of an OrientationField must be even!"
        assert field.shape[1] % 2 == 0, "The width of an OrientationField must be even!"
        self._field: ImageFloatArray = field

    @staticmethod
    def from_cartesian(x: ImageFloatArray, y: ImageFloatArray) -> OrientationField:
        """Creates an orientation field given the x and y components of the orientation field.

        Parameters
        ----------
        x : ImageArray
            The x-component of the orientation field.
        y : ImageArray
            The y-component of the orientation field.

        Returns
        -------
        OrientationField
            The orientation field.
        """
        field: ImageFloatArray = np.zeros((x.shape[0], x.shape[1], 2))
        field[:, :, 0] = x
        field[:, :, 1] = y
        return OrientationField(field)

    @staticmethod
    def from_polar(
        strengths: ImageFloatArray, directions: ImageFloatArray
    ) -> OrientationField:
        """Creates an orientation field given orientation strengths and directions.

        Parameters
        ----------
        strengths : ImageArray
            A scalar array of orientation field strengths.
        directions: ImageArray
            A 2D vector array of orientation field directions.

        Returns
        -------
        OrientationField
            The orientation field.
        """
        x = strengths * np.cos(directions)
        y = strengths * np.sin(directions)
        return OrientationField.from_cartesian(x, y)

    def __str__(self) -> str:
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
        assert self.field.shape[2] == 2
        return (self.num_rows, self.num_columns, self.field.shape[2])

    @property
    def field(self) -> ImageFloatArray:
        """ImageArray: The underlying field array."""
        return self._field

    @property
    def x(self) -> ImageFloatArray:
        """ImageArray: The x-component of the orientation."""
        return self._field[:, :, 0]

    @property
    def y(self) -> ImageFloatArray:
        """ImageArray: The y-component of the orientation."""
        return self._field[:, :, 1]

    def get_vector_at(
        self, row_index: int, column_index: int
    ) -> npt.NDArray[np.floating]:
        """The 2D orientation vector at the given indices.

        Parameters
        ----------
        row_index : int
            The row index.
        column_index : int
            The column index.

        Returns
        -------
        npt.NDArray[np.floating]
            The 2D orientation vector.
        """
        return self.field[row_index, column_index, :]

    def get_strengths(self) -> ImageFloatArray:
        """The orientation strength of each cell.

        Returns
        -------
        ImageArray
            The orientation strength as an array.
        """
        strengths = np.sqrt(np.square(self.x) + np.square(self.y))
        return strengths

    def get_directions(self) -> ImageFloatArray:
        """The orientation direction of each cell given as angles in the range [0, pi)

        Returns
        -------
        ImageArray
            The orientation directions in angles in the range [0, pi).
        """
        directions = np.arctan2(self.y, self.x) % np.pi
        return directions

    def resize(self, new_width: int, new_height: int) -> OrientationField:
        """Returns the orientation field resized via interpolation.

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
        return OrientationField(transform.resize(self.field, (new_height, new_width)))

    @staticmethod
    def merge(
        coarser_field: OrientationField, finer_field: OrientationField
    ) -> OrientationField:
        """Merge two orientation fields together. The first field must be of a lower resolution
        than the second field. The resultant field is the same resolution as that of the finer field.

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
            coarser_field.num_rows < finer_field.num_rows
            and coarser_field.num_columns < finer_field.num_columns
        )
        assert (
            is_finer_higher_resolution
        ), "The finer field must be a higher resolution than the coarser field."

        # Upscale coarse field to have same resolution as finer field
        resized_coarse_field = np.zeros(finer_field.shape)
        resized_coarse_field = coarser_field.resize(
            finer_field.num_columns, finer_field.num_rows
        )

        resized_coarse_strengths = resized_coarse_field.get_strengths()
        fine_field_strengths = finer_field.get_strengths()

        # gains = Sf / (Sf + Sc)
        gains = fine_field_strengths
        denominator = fine_field_strengths + resized_coarse_strengths
        gains[denominator != 0] /= denominator[denominator != 0]

        # Vf' = Vc + Sf / (Sf + Sc) * (Vf - Vc)
        merged_field = resized_coarse_field.add(
            finer_field.subtract(resized_coarse_field).scalar_field_multiply(gains)
        )
        return merged_field

    def scalar_field_multiply(self, scalar_field: ImageFloatArray) -> OrientationField:
        """Returns the orientation field mulitplied by a scalar field.

        Parameters
        ----------
        scalar_field : ImageArray
            The scalar field to multiply with.

        Returns
        -------
        OrientationField
            The multiplied orientation field.
        """
        assert (
            len(scalar_field.shape) == 2
        ), "The scalar field must be MxN as it is scalar."
        assert (
            scalar_field.shape[0] == self.num_rows
            and scalar_field.shape[1] == self.num_columns
        ), "The scalar field must have the same height and width as the OrientationField"

        # Multiply each component
        result = self.field
        result[:, :, 0] *= scalar_field
        result[:, :, 1] *= scalar_field
        return OrientationField(result)

    def add(self, other: OrientationField) -> OrientationField:
        """Returns the orientation field added with another field.

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
            self, other
        )
        result = np.zeros_like(vector_sum)
        result[sum_greater] = vector_sum[sum_greater]
        result[~sum_greater] = vector_difference[~sum_greater]
        return OrientationField(result)

    def subtract(self, other: OrientationField) -> OrientationField:
        """Returns the orientation field subtracted by another field.

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
            self, other
        )
        result = np.zeros_like(vector_sum)
        result[~sum_greater] = vector_sum[~sum_greater]
        result[sum_greater] = vector_difference[sum_greater]
        return OrientationField(result)

    @staticmethod
    def _prepare_sum(
        left: OrientationField, right: OrientationField
    ) -> tuple[ImageFloatArray, ImageFloatArray, npt.NDArray[np.bool_]]:
        """Calculate the necessary components to perform an orientation field sum or difference.

        Parameters
        ----------
        left : OrientationField
            The left hand side of the sum/difference.
        right : OrientationField
            The right hand side of the sum/difference.

        Returns
        -------
        vector_sum : ImageArray
            The vector sum of the two fields.
        vector_difference : ImageArray
            The vector difference of the two fields.
        sum_greater : npt.NDArray[np.bool_]
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
            np.sum(np.square(vector_difference), axis=2)
        )

        sum_greater = vector_sum_lengths > vector_difference_lengths
        sum_greater = np.repeat(sum_greater[:, :, np.newaxis], 2, axis=2)
        return (vector_sum, vector_difference, sum_greater)

    def denoise(self, neighbour_distance: int = 5) -> OrientationField:
        """Returns a denoised orientation field.

        Parameters
        ----------
        neighbour_distance : int, optional
            The distance between a pixel and its four cardinal neighbours.

        Returns
        -------
        OrientationField
            The denoised field.
        """
        SUBTRACT_AMOUNT: float = np.cos(np.pi / 4)
        # Allocate new field
        denoised = np.zeros(self.shape)

        # Iterate through every pixel
        for row_idx in range(self.num_rows):
            for column_idx in range(self.num_columns):
                # Calculate the norm of the orientation vector
                current_vector = self.get_vector_at(row_idx, column_idx)
                current_vector_norm = np.linalg.norm(current_vector)
                # The denoising equation requires non-zero norms
                if current_vector_norm == 0:
                    continue

                # Collect neighbours
                neighbour_vectors = []
                for row_offset, column_offset in (
                    (-neighbour_distance, -neighbour_distance),
                    (+neighbour_distance, -neighbour_distance),
                    (-neighbour_distance, +neighbour_distance),
                    (+neighbour_distance, +neighbour_distance),
                ):
                    target_row = row_idx + row_offset
                    target_column = column_idx + column_offset
                    # Only add neighbours within the image
                    if target_row < 0 or target_row >= self.num_rows:
                        continue
                    if target_column < 0 or target_column >= self.num_columns:
                        continue
                    neighbour_vectors.append(
                        self.get_vector_at(target_row, target_column)
                    )
                neighbour_strengths = np.zeros(len(neighbour_vectors))

                for idx, neighbour in enumerate(neighbour_vectors):
                    neighbour_norm = np.linalg.norm(neighbour)
                    # The denoising equation requires non-zero norms
                    if neighbour_norm == 0:
                        continue
                    # max(|V dot V'| - cos(pi/4), 0) / (|V| * |V'|)
                    neighbour_strengths[idx] = max(
                        np.dot(neighbour, current_vector) - SUBTRACT_AMOUNT, 0
                    ) / (current_vector_norm * neighbour_norm)

                # New orientation strength
                new_strength = np.median(neighbour_strengths)
                # Set new value
                denoised[row_idx, column_idx, :] = (
                    current_vector / current_vector_norm
                ) * new_strength
        # Create denoised field
        return OrientationField(denoised)

    # TODO: Do more thorough testing with non-vectorised version
    def denoise_vectorised(self, neighbour_distance: int = 5) -> OrientationField:
        """Returns a denoised orientation field.

        Parameters
        ----------
        neighbour_distance : int, optional
            The distance between a pixel and its four cardinal neighbours.

        Returns
        -------
        OrientationField
            The denoised field.
        """
        SUBTRACT_AMOUNT: float = np.cos(np.pi / 4)
        # Allocate new field
        denoised = np.zeros(self.shape)

        neighbour_arrays: list[ImageFloatArray] = [
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
                (neighbour * self.field).sum(axis=2) - SUBTRACT_AMOUNT, 0
            )
            current_strength[product_norm != 0] /= product_norm[product_norm != 0]
            current_strength[np.isnan(current_strength)] = 0
            current_strength[~neighbour_masks[:, :, neighbour_idx]] = np.nan
            neighbour_strengths[:, :, neighbour_idx] = current_strength
        median_strength = np.nanmedian(neighbour_strengths, axis=2)
        combined_mask = vector_norm != 0
        denoised[:, :, 0][combined_mask] = (
            self.field[:, :, 0][combined_mask]
            / vector_norm[combined_mask]
            * median_strength[combined_mask]
        )
        denoised[:, :, 1][combined_mask] = (
            self.field[:, :, 1][combined_mask]
            / vector_norm[combined_mask]
            * median_strength[combined_mask]
        )

        return OrientationField(denoised)


def generate_orientation_fields(
    image: ImageFloatArray, num_orientation_field_levels: int = 3
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
        current_level = _generate_raw_orientation_field(resized_image)
        orientation_field_levels.append(current_level)

    # Merge orientation fields
    merged_field: OrientationField = reduce(
        lambda x, y: OrientationField.merge(x, y),
        orientation_field_levels,
    )

    # Denoise
    denoised_field = merged_field.denoise_vectorised()
    return denoised_field


def _generate_raw_orientation_field(
    image: ImageFloatArray,
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
    filtered_images = _generate_orientation_filtered_images(image)

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


def _generate_orientation_filtered_images(
    image: ImageFloatArray,
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
        orientation_filter = _generate_orientation_filter_kernel(angle)
        filtered_images[:, :, idx] = signal.convolve2d(
            image, orientation_filter, mode="same"
        )
    return filtered_images


def _generate_orientation_filter_kernel(
    theta: float, radius: int = 5
) -> ImageFloatArray:
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
