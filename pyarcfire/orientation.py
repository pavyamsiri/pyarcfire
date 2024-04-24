from __future__ import annotations

# Standard libraries
from functools import reduce
from typing import Sequence

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
        return np.squeeze(self.field[row_index, column_index, :])

    def get_strengths(self) -> ImageArray:
        strengths = np.sqrt(np.square(self.x) + np.square(self.y))
        return strengths

    def get_directions(self) -> ImageArray:
        directions = np.arctan2(self.y, self.x) % np.pi
        return directions

    def rescale(self, scale_factor: float) -> OrientationField:
        resized_field = transform.rescale(self.field, scale_factor, channel_axis=2)
        return OrientationField(resized_field)

    def merge(self, other: OrientationField) -> OrientationField:
        print(self.shape)
        print(other.shape)
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
        b[negative_vertical, 0] -= b[negative_vertical, 0]
        b[negative_vertical, 1] -= b[negative_vertical, 1]
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
                current_vector_norm = np.sqrt(np.sum(np.square(current_vector)))
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
                    neighbour_norm = np.sqrt(np.sum(np.square(neighbour)))
                    neighbour_sims[idx] = max(
                        abs(neighbour * current_vector) - subtract_amount
                    ) / (current_vector_norm * neighbour_norm)

                denoised[row_idx, column_idx, :] = (
                    current_vector / current_vector_norm
                ) * np.median(neighbour_sims)
        return OrientationField(denoised)


def generate_orientation_fields(
    image: ImageArray, num_orientation_field_levels: int = 3
) -> tuple[ImageArray, ImageArray, ImageArray]:
    # Number of image scales to use when computing the orientation field.
    # The dimensions of the preprocessed image (see resizeDims) must be
    # divisible by 2^(numOrientationFieldLevels-1).

    orientation_field_levels: list[OrientationField] = []
    for idx in range(num_orientation_field_levels):
        scale_factor: float = 1 / 2 ** (num_orientation_field_levels - idx - 1)
        resized_image = transform.rescale(image, scale_factor)
        current_level = generate_single_orientation_field_level(resized_image)
        criteria = current_level.y < 0
        current_level.field[criteria] = -current_level.field[criteria]
        orientation_field_levels.append(current_level)

    # Now merge orientation fields
    merged_field: OrientationField = reduce(
        lambda x, y: x.merge(y),
        orientation_field_levels,
    )

    denoised_field = merged_field.denoise()
    orientation_vectors = denoised_field.field
    strengths = denoised_field.get_strengths()
    directions = denoised_field.get_directions()
    return orientation_vectors, strengths, directions


def generate_single_orientation_field_level(
    image: ImageArray,
    light_dark: int = 1,
) -> OrientationField:
    filtered_images = generate_orientation_filtered_images(image)

    if light_dark > 0:
        filtered_images[filtered_images < 0] = 0
    elif light_dark < 0:
        filtered_images[filtered_images > 0] = 0

    energies = np.square(filtered_images, dtype=np.complex64)
    for idx in range(9):
        energies[:, :, idx] = energies[:, :, idx] * np.exp(1j * 2 * idx * np.pi / 9)
    energy = np.sum(energies, axis=2)
    # Magnitude of complex valued energy
    strengths = np.abs(energy)
    # Angle of complex valued energy
    angles = np.angle(energy) / 2

    field = OrientationField.from_polar(strengths, angles)

    return field


def generate_orientation_filtered_images(
    image: ImageArray,
) -> ImageArraySequence:
    filtered_images = np.zeros((image.shape[0], image.shape[1], 9))

    for idx in range(9):
        angle = (idx * np.pi) / 9
        orientation_filter = generate_orientation_filter_fxn(angle)
        # NOTE: Matlab's conv2 and scipy's convolve2d produce different results for the same mode if the image has
        # an odd number of rows and/or columns. Let's assume for now that they are the same.
        filtered_images[:, :, idx] = signal.convolve2d(
            image, orientation_filter, mode="same"
        )
    return filtered_images


def generate_orientation_filter_fxn(
    angle: float, radius: float = 5, use_hilbert: bool = False
) -> ImageArray:
    # Generates an orientation field filter matrix described in the PhD thesis
    #  "Inferring Galaxy Morphology Through Texture Analysis" (K. Au 2006).
    #  The filter is a 1D LoG filter extended in 2D along an angle theta, such
    #  that the filter response is strongest for that angle.
    # INPUTS:
    #   theta: angle of the filter orientation
    #   radius: optional parameter specifying where to truncate the filter
    #    values; matrix size will be 2*radius+1
    #   hilbt: whether to use the Hilbert transform of the filter instead
    assert radius > 0, f"Radius must be positive but it is instead {radius}"

    # Mesh size in cells
    num_cells: int = int(2 * np.ceil(radius) + 1)
    # sample [-pi, pi], in pixel middles
    max_value: float = np.pi * 2 * radius / (2 * radius + 1)
    # Not sure what cVal and rVals exactly corresponds to
    columns, rows = np.meshgrid(
        np.linspace(-max_value, max_value, num_cells),
        np.linspace(-max_value, max_value, num_cells),
    )
    diagonals = rows * np.cos(angle) + columns * np.sin(angle)
    diagonals_squared = np.square(diagonals)
    filter_matrix = (
        (2 / np.sqrt(3))
        * (np.pi ** (-1 / 4))
        * (1 - diagonals_squared)
        * np.exp(-diagonals_squared / 2)
    )

    # TODO: Add Hilbert transform of filter
    # if use_hilbert:
    # filterH = imag(-cmhf(dVals)) / sqrt(3);
    # filterH = filterH .* (sum(abs(filter(:))) / sum(abs(filterH(:))));
    # filter = filterH;
    _ = use_hilbert

    sigma = max_value / 2
    weighted_sum = rows * np.cos(angle + np.pi / 2) + columns * np.sin(
        angle + np.pi / 2
    )
    gaussian_window = (1 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(
        (-1 / (2 * (sigma**2))) * np.square(weighted_sum)
    )

    filter_matrix *= gaussian_window
    # Normalise matrix
    filter_matrix /= np.sqrt(np.sum(np.square(filter_matrix)))
    return filter_matrix


def merge_orientation_fields(
    coarse_field: ImageArray,
    fine_field: ImageArray,
    fine_field_strengths: ImageArray,
) -> ImageArray:
    # Merges two orientation fields at different resolutions, as described in
    # the PhD thesis "Inferring Galaxy Morphology Through Texture Analysis"
    # (K. Au 2006).
    # The "fine" orientation field should have twice the resolution (in each
    # dimension) as the "coarse" orientation field
    # INPUTS:
    #   oriCoarse: the orientation field generated from a lower-resolution
    #       version of the image
    #   oriFine: the orientation field generated from a higher-resolution
    #       version of the image
    # OUTPUTS:
    #   oriMerged: result from merging the two orientation fields

    num_rows, num_columns, _ = fine_field.shape
    assert num_rows % 2 == 0, f"Number of rows is not even! {num_rows=}"
    assert num_columns % 2 == 0, f"Number of columns is not even! {num_columns=}"

    resized_coarse_field = np.zeros_like(fine_field)
    # Note: Might have to do it separately
    resized_coarse_field = transform.rescale(coarse_field, 2, channel_axis=2)
    resized_coarse_strengths = get_orientation_field_strengths(resized_coarse_field)

    gains = fine_field_strengths / (resized_coarse_strengths + fine_field_strengths)
    gains[np.isnan(gains)] = 0
    second_arg = np.repeat(gains[:, :, np.newaxis], 2, axis=2) * add_orientation_field(
        fine_field, resized_coarse_field, False
    )
    merged_field = add_orientation_field(
        resized_coarse_field,
        second_arg,
        True,
    )
    return merged_field


def get_orientation_field_strengths(orientation_field: ImageArray) -> ImageArray:
    strengths = np.sqrt(
        np.square(orientation_field[:, :, 0]) + np.square(orientation_field[:, :, 1])
    )
    return strengths


def get_orientation_field_directions(orientation_field: ImageArray) -> ImageArray:
    directions = (
        np.arctan(orientation_field[:, :, 1] / orientation_field[:, :, 0]) % np.pi
    )
    directions[orientation_field[:, :, 0] == 0] = 0
    return directions


def add_orientation_field(a: ImageArray, b: ImageArray, add: bool) -> ImageArray:
    # Adds a and b, interpreting the vectors as orientations. Unlike
    # vector subtraction, orientations differing by 180 degrees are considered
    # equivalent. a(i, j) and b(i, j) should be 2-element vectors.
    negative_vertical_b = b[:, :, 1] < 0
    negative_vertical_b = np.repeat(negative_vertical_b[:, :, np.newaxis], 2, axis=2)
    b[negative_vertical_b] = -b[negative_vertical_b]
    vector_sum = a + b
    vector_difference = a - b
    vector_sum_lengths = np.sqrt(np.sum(np.square(vector_sum), axis=2))
    vector_difference_lengths = np.sqrt(np.sum(np.square(vector_difference), axis=2))
    sum_greater = vector_sum_lengths > vector_difference_lengths
    sum_greater = np.repeat(sum_greater[:, :, np.newaxis], 2, axis=2)
    result = np.zeros_like(a)
    if add:
        result[sum_greater] = vector_sum[sum_greater]
        result[~sum_greater] = vector_difference[~sum_greater]
    else:
        result[~sum_greater] = vector_sum[~sum_greater]
        result[sum_greater] = vector_difference[sum_greater]
    return result


def denoise_orientation_field(orientation_field: ImageArray) -> ImageArray:
    # Performs the orientation field de-noising described in the PhD thesis
    # "Inferring Galaxy Morphology Through Texture Analysis" (K. Au 2006).
    # INPUTS:
    #   ofld: orientation field without de-noising (but already merged from 3
    #       resolution levels if the process in the thesis is followed)
    # OUTPUTS:
    #   dofld: de-noised orientation field

    assert len(orientation_field.shape) == 3 and orientation_field.shape[2] == 2

    # The neighbor distance is 5 in the thesis; if needed, this could be
    # changed here or made a parameter
    neighbour_distance: int = 5
    denoised = np.zeros_like(orientation_field)

    # TODO: vectorize this without losing generality
    num_rows, num_columns, _ = orientation_field.shape
    for row_idx in rprogress.track(range(num_rows)):
        for column_idx in range(num_columns):
            current_vector = np.squeeze(orientation_field[row_idx, column_idx, :])
            current_vector_norm = np.sqrt(np.sum(np.square(current_vector)))
            neighbour_vectors = []
            # Check that the neighbour is not past the top left corner
            if (
                row_idx - neighbour_distance >= 0
                and column_idx - neighbour_distance >= 0
            ):
                neighbour_vector = orientation_field[
                    row_idx - neighbour_distance, column_idx - neighbour_distance, :
                ]
                neighbour_vectors.append(neighbour_vector)
            if (
                row_idx + neighbour_distance < num_rows
                and column_idx - neighbour_distance >= 0
            ):
                neighbour_vector = orientation_field[
                    row_idx + neighbour_distance, column_idx - neighbour_distance, :
                ]
                neighbour_vectors.append(neighbour_vector)
            if (
                row_idx - neighbour_distance >= 0
                and column_idx + neighbour_distance < num_columns
            ):
                neighbour_vector = orientation_field[
                    row_idx - neighbour_distance, column_idx + neighbour_distance, :
                ]
                neighbour_vectors.append(neighbour_vector)
            if (
                row_idx + neighbour_distance < num_rows
                and column_idx + neighbour_distance < num_columns
            ):
                neighbour_vector = orientation_field[
                    row_idx + neighbour_distance, column_idx + neighbour_distance, :
                ]
                neighbour_vectors.append(neighbour_vector)
            neighbour_sims = np.zeros(len(neighbour_vectors))

            subtract_amount = np.cos(np.pi / 4)
            for idx, neighbour in enumerate(neighbour_vectors):
                neighbour_norm = np.sqrt(np.sum(np.square(neighbour)))
                neighbour_sims[idx] = max(
                    abs(neighbour * current_vector) - subtract_amount
                ) / (current_vector_norm * neighbour_norm)

            denoised[row_idx, column_idx, :] = (
                current_vector / current_vector_norm
            ) * np.median(neighbour_sims)
    return denoised
