from __future__ import annotations

# Standard libraries
import logging


# External libraries
import numpy as np
from numpy import typing as npt
from rich import progress as rprogress
from scipy import sparse


# Internal libraries
from .orientation import OrientationField

log = logging.getLogger(__name__)


def generate_similarity_matrix(
    orientation: OrientationField, similarity_cutoff: float
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
    log.debug(
        f"Similarity matrix has {similarity_matrix.count_nonzero():,} nonzero elements."
    )
    is_hollow = np.allclose(similarity_matrix.diagonal(), 0)
    assert is_hollow, "Similarity matrix has non-zero diagonal values!"
    is_symmetric = (
        similarity_matrix - similarity_matrix.transpose()
    ).count_nonzero() == 0  # type:ignore
    assert is_symmetric, "Similarity matrix is not symmetric!"
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

    max_num_elements: int = 8 * num_nonzero_vectors
    # Actually the root index?
    similarity_values = np.full(max_num_elements, np.nan)
    root_indices = np.zeros(max_num_elements)
    # Actually the child indices?
    child_indices = np.zeros(max_num_elements)

    add_neighbour_row = maximum_distance * np.array([-1, -1, -1, 0, 0, +1, +1, +1])
    add_neighbour_column = maximum_distance * np.array([-1, 0, +1, -1, +1, -1, 0, +1])

    fill_idx: int = 0
    for idx in rprogress.track(nonzero_indices):
        # Compute similarity values for each neighbour
        row_idx, column_idx = np.unravel_index(idx, (num_rows, num_columns))
        neighbour_row_indices = row_idx + add_neighbour_row
        neighbour_column_indices = column_idx + add_neighbour_column
        in_range = (
            (neighbour_column_indices >= 0)
            & (neighbour_column_indices < num_columns)
            & (neighbour_row_indices >= 0)
            & (neighbour_row_indices < num_rows)
        )
        neighbour_indices = np.ravel_multi_index(
            (neighbour_row_indices[in_range], neighbour_column_indices[in_range]),
            (num_rows, num_columns),
        )
        assert isinstance(neighbour_indices, np.ndarray)
        neighbours = orientation_vectors[neighbour_indices]
        neighbour_similarities: npt.NDArray[np.floating] = np.abs(
            np.dot(neighbours, orientation_vectors[idx])
        )
        assert len(neighbour_similarities) == len(neighbours)

        neighbour_similarities[neighbour_similarities < similarity_cutoff] = 0

        num_neighbours = len(neighbours)
        start_idx = 8 * fill_idx
        stride = num_neighbours
        assert stride <= 8
        assign_indices = slice(start_idx, start_idx + stride)
        similarity_values[assign_indices] = neighbour_similarities
        root_indices[assign_indices] = idx
        child_indices[assign_indices] = neighbour_indices
        fill_idx += 1
    valid_indices = ~np.isnan(similarity_values)

    similarity_values = similarity_values[valid_indices]
    root_indices = root_indices[valid_indices]
    child_indices = child_indices[valid_indices]

    return (similarity_values, root_indices, child_indices, num_vecs)
