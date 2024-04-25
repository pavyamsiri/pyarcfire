from __future__ import annotations

# Standard libraries
import logging
from typing import Sequence


# External libraries
import numpy as np
from numpy import typing as npt
from scipy import sparse


# Internal libraries
from .definitions import ImageArray
from .orientation import OrientationField

log = logging.getLogger(__name__)


class Cluster:
    def __init__(self, merged_points: Sequence[int], merge_value: float):
        self._first_cluster_index: int | None = None
        self._second_cluster_index: int | None = None
        self._merge_value: float = merge_value
        self._merged_points: list[int] = list(merged_points)

    @staticmethod
    def empty() -> Cluster:
        return Cluster(merged_points=[], merge_value=0)

    def __str__(self) -> str:
        return f"Cluster(first={self._first_cluster_index}, second={self._second_cluster_index}, value={self._merge_value}, points={self._merged_points})"

    @property
    def size(self) -> int:
        return len(self._merged_points)

    def get_points(self) -> list[int]:
        return self._merged_points

    def set_clusters(self, first_index: int, second_index: int) -> None:
        self._first_cluster_index = first_index
        self._second_cluster_index = second_index


# def find_clusters(
#     orientation: OrientationField, similarity_cutoff: float = -np.inf
# ) -> None:
#     useMex = stgs.useMex
#     stopThres = stgs.stopThres
#     colorThres = stgs.clusSizeCutoff
#
#     simls = generate_similarity_matrix(orientation, similarity_cutoff)
#     clusters = generate_hac_tree(simls, img, barInfo, stgs)
#     return clusters


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

    similarity_values, root_indices, child_indices, num_vecs = (
        calculate_pixel_similarities(orientation, similarity_cutoff)
    )

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


def generate_hac_tree(
    similarity_matrix: sparse.coo_matrix,
    image: ImageArray,
    orientation: OrientationField,
    # barInfo,
    # ctrR,
    # ctrC,
    # stgs,
    # tracePts,
    # aviName,
    # colorThres,
    stop_threshold: float = 0.15,
    error_ratio_threshold: float = 2.5,
    merge_check_minimum_cluster_size: int = 25,
):
    # Performs HAC clustering and returns the resulting dendogram
    # INPUTS:
    #   simlMtx: the similarity matrix to use for clustering
    #   img: the image used to generate the similarity matrix
    #   barInfo:
    #   ctrR:
    #   ctrC:
    #   stgs: structure containing algorithm settings (see settings.m)
    #   tracePts: optional parameter specifying points (similarity values) at
    #       which the current clustering state should be saved as a frame in an
    #       AVI video file (default: empty; no AVI file)
    #   aviName: (optional) name of the AVI file, if such a file is to be saved
    #   colorThres: minimum size for clusters to get their own color (if an AVI
    #       is to be saved)
    # OUTPUTS:
    #   clusters: HAC dendogram, given as a structure array of root nodes for
    #       each of the cluster trees.  Each node contains indices of the two
    #       merged clusters, the similarity value between these clusters, and
    #       all points from these clusters

    # NOTE: Not sure what this means yet
    failed_to_revert = False

    num_rows: int = orientation.num_rows
    num_columns: int = orientation.num_columns

    # delete self-similarities
    similarity_matrix.setdiag(0)
    similarity_matrix.eliminate_zeros()
    # NOTE: This isn't in the original code, but it does assume that similarities are nonnegative
    similarity_matrix = abs(similarity_matrix)

    # Indices of non-zero columns
    select_points = similarity_matrix.sum(axis=0).nonzero()[1]
    select_rows, select_columns = np.unravel_index(
        select_points, (num_rows, num_columns)
    )
    select_points = select_points[image[select_rows, select_columns] > 0]
    # NOTE: After this similarity matrix's shape has changed if to len(select_points) x len(select_points)
    nonzero_similarity_matrix = similarity_matrix.tocsc()[select_points, :][
        :, select_points
    ]
    assert isinstance(nonzero_similarity_matrix, sparse.csc_matrix)
    num_points = nonzero_similarity_matrix.shape[1]
    # create a cluster for each point that has a nonzero similarity with at
    # least one other point
    clusters: list[Cluster] = []
    for select_point in select_points:
        cluster = Cluster(merged_points=(select_point,), merge_value=0)
        clusters.append(cluster)
    active_clusters = np.ones(num_points, dtype=np.bool_)
    arc_merge_checks: int = 0
    merge_stop_count: int = 0
    iteration_count: int = 0
    while True:
        iteration_count += 1
        max_idx = nonzero_similarity_matrix.argmax()
        row_idx, column_idx = np.unravel_index(max_idx, nonzero_similarity_matrix.shape)
        assert isinstance(row_idx, np.integer)
        assert isinstance(column_idx, np.integer)
        value = nonzero_similarity_matrix[row_idx, column_idx]
        assert isinstance(value, float)
        first_cluster: Cluster = clusters[row_idx]
        second_cluster: Cluster = clusters[column_idx]
        # Check that both clusters are past the threshold cluster size
        if (
            min(first_cluster.size, second_cluster.size)
            > merge_check_minimum_cluster_size
        ):
            # log.debug("PAST THE THRESHOLD")
            pass

        # Similarity value is high
        if value >= stop_threshold:
            # Create new merged cluster
            merged_cluster = Cluster(
                merged_points=first_cluster.get_points() + second_cluster.get_points(),
                merge_value=value,
            )
            merged_cluster.set_clusters(int(row_idx), int(column_idx))
            clusters[row_idx] = merged_cluster
            clusters[column_idx] = Cluster.empty()
            active_clusters[column_idx] = False

            new_similarities = np.maximum(
                nonzero_similarity_matrix[:, row_idx].toarray(),
                nonzero_similarity_matrix[:, column_idx].toarray(),
            )
            new_similarities[row_idx] = 0
            first_nonzero_rows, _ = np.nonzero(new_similarities)
            second_nonzero_rows, _ = np.nonzero(
                nonzero_similarity_matrix[:, row_idx].toarray()
            )
            has_old_values = np.isin(first_nonzero_rows, second_nonzero_rows)
            change_values = new_similarities[first_nonzero_rows]
            assert not np.any((nonzero_similarity_matrix < 0).toarray())
            change_values[has_old_values] -= nonzero_similarity_matrix[
                second_nonzero_rows, row_idx
            ].toarray()
            # First nonzero rows - row idx
            change_row_indices = np.zeros(2 * len(first_nonzero_rows))
            change_row_indices[: len(first_nonzero_rows)] = first_nonzero_rows
            change_row_indices[len(first_nonzero_rows) :] = row_idx

            # Row idx - first nonzero rows
            change_column_indices = np.zeros(2 * len(first_nonzero_rows))
            change_column_indices[: len(first_nonzero_rows)] = row_idx
            change_column_indices[len(first_nonzero_rows) :] = first_nonzero_rows
            change_matrix = sparse.coo_matrix(
                (
                    np.tile(change_values, (2, 1)).flatten(),
                    (change_row_indices, change_column_indices),
                ),
                (num_points, num_points),
            )
            nonzero_similarity_matrix += change_matrix

            third_nonzero_rows, _ = np.nonzero(
                nonzero_similarity_matrix[:, column_idx].toarray()
            )
            # Third nonzero rows - column idx
            change_row_indices = np.zeros(2 * len(third_nonzero_rows))
            change_row_indices[: len(third_nonzero_rows)] = third_nonzero_rows
            change_row_indices[len(third_nonzero_rows) :] = column_idx

            # Column idx - third nonzero rows
            change_column_indices = np.zeros(2 * len(third_nonzero_rows))
            change_column_indices[: len(third_nonzero_rows)] = column_idx
            change_column_indices[len(third_nonzero_rows) :] = third_nonzero_rows
            change_values = -nonzero_similarity_matrix[
                third_nonzero_rows, column_idx
            ].toarray()
            change_matrix = sparse.coo_matrix(
                (
                    np.tile(change_values, (2, 1)).flatten(),
                    (change_row_indices, change_column_indices),
                ),
                (num_points, num_points),
            )
            nonzero_similarity_matrix += change_matrix
        else:
            break

    clusters = list(np.array(clusters)[active_clusters])
    log.debug(np.count_nonzero(active_clusters))
