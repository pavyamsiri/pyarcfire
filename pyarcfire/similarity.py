from __future__ import annotations

# Standard libraries
import logging


# External libraries
import numpy as np
from numpy import typing as npt
from scipy import sparse


# Internal libraries
from .orientation import OrientationField

log = logging.getLogger(__name__)


def generate_similarity_matrix(
    orientation: OrientationField, similarity_cutoff: float = -np.inf
) -> sparse.coo_matrix:
    # Generates a sparse pixel-to-pixel similarity matrix using an orientation
    #   field derived from the input image.
    # INPUTS:
    #   img: image used to generate the orientation field
    #   oriMtx: orientation field generated from the image
    #   stgs: structure containing algorithm settings (see getDefaultSettings.m)
    #   simlCutoff: minimum value for a similarity score to be nonzero in the
    #       similarity matrix (this is often related to, but not necessarily
    #       the same as, the HAC stopping threshold in the 'stgs' struct)
    # OUTPUTS:
    #   simls: the sparse pixel-to-pixel similarity matrix

    (
        similarity_values,
        root_indices,
        child_indices,
        num_vecs,
    ) = calculate_pixel_similarities(orientation, similarity_cutoff)

    similarity_matrix = sparse.coo_matrix(
        (similarity_values, (root_indices, child_indices)), shape=(num_vecs, num_vecs)
    )
    return similarity_matrix


def calculate_pixel_similarities(
    orientation: OrientationField,
    similarity_cutoff: float,
    maximum_distance: int = 1,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    int,
]:
    # Calculates pixel similarities using dot products of neighbors'
    # orientation field vectors (similarities are zero if pixels are not within
    # an 8-neighborhood of each other). The outputs can be used directly, or
    # used to construct a similarity matrix.
    # INPUTS:
    #   img: image used to generate the orientation field
    #   oriMtx: orientation field generated from the image
    #   stgs: structure containing algorithm settings (see getDefaultSettings.m)
    #   simlCutoff: minimum value for a similarity score to be nonzero in the
    #       similarity matrix (this is often related to, but not necessarily
    #       the same as, the HAC stopping threshold in the 'stgs' struct)
    # OUTPUTS:
    #   fromInds, toInds, simlVals: nonzero pixel similarities, such that the
    #       similarity from pixel fromInds(ii) to pixel toInds(ii) is
    #       simlVals(ii)
    #   numElts: total number of similarities (generally a much larger value
    #       than the length of the other outputs, since only neighbors have
    #       nonzero similarities)

    num_rows: int = orientation.num_rows
    num_columns: int = orientation.num_columns

    num_vecs = orientation.num_cells
    strengths = orientation.get_strengths()

    # Yx2 matrix where Y is MxN, the number of vectors
    orientation_vectors = np.reshape(orientation.field, (num_vecs, 2))
    # Non-zero indices of flattened strengths matrix
    nonzero_indices = np.flatnonzero(strengths)
    num_nonzero_vectors = nonzero_indices.size

    max_num_elements: int = 9 * num_nonzero_vectors
    # Actually the root index?
    root_indices = np.zeros(max_num_elements)
    # Actually the child indices?
    child_indices = np.zeros(max_num_elements)
    similarity_values = np.zeros(max_num_elements)

    next_index: int = 1
    for idx in nonzero_indices:
        row_idx, column_idx = np.unravel_index(idx, (num_rows, num_columns))
        neighbour_column_indices = column_idx + maximum_distance * np.array(
            [-1, 0, 1, -1, 0]
        )
        neighbour_row_indices = row_idx + maximum_distance * np.array(
            [-1, -1, -1, 0, 0]
        )
        in_range = (
            (neighbour_column_indices >= 0)
            & (neighbour_column_indices < num_columns)
            & (neighbour_row_indices >= 0)
        )
        neighbour_indices = np.ravel_multi_index(
            (neighbour_row_indices[in_range], neighbour_column_indices[in_range]),
            (num_rows, num_columns),
        )
        assert isinstance(neighbour_indices, np.ndarray)

        neighbours = orientation_vectors[neighbour_indices]
        neighbour_similarities: npt.NDArray[np.floating] = np.dot(
            neighbours, orientation_vectors[idx]
        )
        assert len(neighbour_similarities) == len(neighbours)

        neighbour_similarities[neighbour_similarities < similarity_cutoff] = 0

        # Assign values for (up to) five neighbours including itself
        num_neighbours: int = len(neighbour_similarities)
        assign_indices = slice(next_index, next_index + num_neighbours)
        root_indices[assign_indices] = idx
        child_indices[assign_indices] = neighbour_indices
        similarity_values[assign_indices] = neighbour_similarities
        next_index += num_neighbours

        # Now assign values for last four neighbours
        num_lower_neighbours: int = num_neighbours - 1
        assign_indices = slice(next_index, next_index + num_lower_neighbours)
        root_indices[assign_indices] = neighbour_indices[:-1]
        child_indices[assign_indices] = idx
        similarity_values[assign_indices] = neighbour_similarities[:-1]
        next_index += num_lower_neighbours
    last_idx = np.flatnonzero(root_indices)[-1]

    root_indices = root_indices[: last_idx + 1]
    child_indices = child_indices[: last_idx + 1]
    similarity_values = similarity_values[: last_idx + 1]

    return (similarity_values, root_indices, child_indices, num_vecs)
