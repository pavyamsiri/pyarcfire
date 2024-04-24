# Standard libraries
from typing import Sequence

# External libraries
import numpy as np
from numpy import typing as npt
from rich import progress as rprogress
from scipy import signal
from skimage import transform

# Internal libraries
from .definitions import ImageArray, ImageArraySequence


def generate_orientation_fields(
    image: ImageArray,
) -> tuple[ImageArray, ImageArray, ImageArray]:
    # Number of image scales to use when computing the orientation field.
    # The dimensions of the preprocessed image (see resizeDims) must be
    # divisible by 2^(numOrientationFieldLevels-1).
    num_orientation_field_levels: int = 3

    orientation_field_levels: list[tuple[ImageArray, ImageArray, ImageArray]] = []
    for idx in range(num_orientation_field_levels):
        scale_factor: float = 1 / 2 ** (num_orientation_field_levels - idx - 1)
        resized_image = transform.rescale(image, scale_factor)
        current_level = generate_single_orientation_field_level(resized_image)
        orientation_field_levels.append(current_level)

    # Now merge orientation fields
    merged_field = orientation_field_levels[0][0]
    for idx in range(1, num_orientation_field_levels):
        merged_field = merge_orientation_fields(
            merged_field,
            orientation_field_levels[idx][0],
            orientation_field_levels[idx][2],
        )

    orientation_vectors = denoise_orientation_field(merged_field)
    strengths = get_orientation_field_strengths(merged_field)
    directions = get_orientation_field_directions(merged_field)
    return orientation_vectors, strengths, directions


def generate_single_orientation_field_level(
    image: ImageArray,
    light_dark: int = 1,
) -> tuple[ImageArray, ImageArray, ImageArray]:
    filtered_images = generate_orientation_filtered_images(image)

    if light_dark > 0:
        filtered_images[filtered_images < 0] = 0
    elif light_dark < 0:
        filtered_images[filtered_images > 0] = 0

    energy = np.square(filtered_images, dtype=np.complex64)
    for idx in range(9):
        energy[:, :, idx] = energy[:, :, idx] * np.exp(1j * 2 * idx * np.pi / 9)
    energy = np.sum(energy, axis=2)
    strengths = np.abs(energy)
    directions = np.angle(energy) / 2

    orientation_vectors = np.zeros((energy.shape[0], energy.shape[1], 2))
    orientation_vectors[:, :, 0] = strengths * np.cos(directions)
    orientation_vectors[:, :, 1] = strengths * np.sin(directions)

    return orientation_vectors, energy, strengths


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
