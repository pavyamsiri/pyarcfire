"""Generates a similarity matrix from an orientation field, presumably derived from an image."""

from __future__ import annotations

# Standard libraries
import logging


# External libraries
import numpy as np
from numpy import typing as npt
from scipy import sparse


# Internal libraries
from .debug_utils import benchmark
from .matrix_utils import (
    is_sparse_matrix_hollow,
    is_sparse_matrix_symmetric,
)
from .orientation import OrientationField

log: logging.Logger = logging.getLogger(__name__)


@benchmark
def generate_similarity_matrix(
    orientation: OrientationField, similarity_cutoff: float
) -> sparse.coo_matrix:
    """Generates a sparse pixel-to-pixel similarity matrix from an orientation field.

    Parameters
    ----------
    orientation : OrientationField
        The orientation field derived from an image.
    similarity_cutoff : float, optional
        The minimum threshold value for a similarity score between pixels to be non-zero.
        Values below this threshold are set to zero similarity.

    Returns
    -------
    similarity_matrix : sparse.coo_matrix
        The similarity matrix expressed as a sparse matrix.

    Notes
    -----
    If the given orientation field is MxN then the resulting similarity matrix is of size OxO
    where O = M * N.
    This scaling is why the matrix is expressed as a sparse matrix.
    """
    (
        similarity_values,
        root_indices,
        child_indices,
        num_vecs,
    ) = _calculate_pixel_similarities(orientation, similarity_cutoff)

    similarity_matrix = sparse.coo_matrix(
        (similarity_values, (root_indices, child_indices)), shape=(num_vecs, num_vecs)
    )
    assert is_sparse_matrix_hollow(similarity_matrix)
    assert is_sparse_matrix_symmetric(similarity_matrix)
    return similarity_matrix


def _calculate_pixel_similarities(
    orientation: OrientationField,
    similarity_cutoff: float,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    int,
]:
    """Calculates similarities between pixels by measuring how well aligned their orientation vectors are.
    This function returns the non-zero similarity values along with its corresponding pixel indices and the
    number of total rows/columns of the corresponding similarity matrix.

    Parameters
    ----------
    orientation : OrientationField
        The orientation field derived from an image.
    similarity_cutoff : float
        The minimum threshold value for a similarity score between pixels to be non-zero.
        Values below this threshold are set to zero similarity.

    Returns
    -------
    similarity_values : FloatArray1D
        The non-zero similarity values between pixels.
    root_indices : IntegerArray1D
        The corresponding indices of root pixel.
    child_indices : IntegerArray1D
        The corresponding indices of child pixel.
    num_vectors : int
        The total number of rows/columns of the similarity matrix.
    """

    num_rows: int = orientation.num_rows
    num_columns: int = orientation.num_columns

    num_vectors = orientation.num_cells
    strengths = orientation.get_strengths()

    # Ox2 matrix where O is M * N, the number of vectors
    orientation_vectors = np.reshape(orientation.field, (num_vectors, 2))
    # Non-zero indices of flattened strengths matrix
    nonzero_indices = np.flatnonzero(strengths)
    num_nonzero_vectors = nonzero_indices.size

    # Each non-zero pixel has eight neighbours at distance 1 so we have eight pairs of similarity values.
    # This is a bit redundant but this implementation is relatively simple and it works.
    max_num_elements: int = 8 * num_nonzero_vectors
    similarity_values = np.full(max_num_elements, np.nan)

    # NOTE: Technically both indices are of equal footing, so these are kind of misnomers.
    # The root index of the similarity value pair
    root_indices = np.zeros(max_num_elements)
    # The child index of the similarity value pair
    child_indices = np.zeros(max_num_elements)

    # Offsets to get neighbour indices
    add_neighbour_row = np.array([-1, -1, -1, 0, 0, +1, +1, +1])
    add_neighbour_column = np.array([-1, 0, +1, -1, +1, -1, 0, +1])

    fill_idx: int = 0
    for idx in nonzero_indices:
        # Compute similarity values for each neighbour
        row_idx, column_idx = np.unravel_index(idx, (num_rows, num_columns))
        neighbour_row_indices = row_idx + add_neighbour_row
        neighbour_column_indices = column_idx + add_neighbour_column
        # Remove out of bounds accesses
        in_range = (
            (neighbour_column_indices >= 0)
            & (neighbour_column_indices < num_columns)
            & (neighbour_row_indices >= 0)
            & (neighbour_row_indices < num_rows)
        )
        # Convert to flat indices
        neighbour_indices = np.ravel_multi_index(
            (neighbour_row_indices[in_range], neighbour_column_indices[in_range]),
            (num_rows, num_columns),
        )
        assert isinstance(neighbour_indices, np.ndarray)
        # Collect neighbours' orientation vectors
        neighbours = orientation_vectors[neighbour_indices]

        # Compute similarities
        neighbour_similarities: npt.NDArray[np.floating] = np.abs(
            np.dot(neighbours, orientation_vectors[idx])
        )
        assert len(neighbour_similarities) == len(neighbours)

        # Perform similarity cut
        neighbour_similarities[neighbour_similarities < similarity_cutoff] = 0

        # Determine where to store values
        num_neighbours = len(neighbours)
        start_idx = 8 * fill_idx
        stride = num_neighbours
        assert stride <= 8
        assign_indices = slice(start_idx, start_idx + stride)

        # Store computed values in their arrays
        similarity_values[assign_indices] = neighbour_similarities
        root_indices[assign_indices] = idx
        child_indices[assign_indices] = neighbour_indices

        # Increment fill index
        fill_idx += 1

    # Ignore unused cells as it is likely we over allocated
    valid_indices = ~np.isnan(similarity_values)

    similarity_values = similarity_values[valid_indices]
    root_indices = root_indices[valid_indices]
    child_indices = child_indices[valid_indices]

    return (similarity_values, root_indices, child_indices, num_vectors)
