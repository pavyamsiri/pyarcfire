from __future__ import annotations

# Standard libraries
import logging
from typing import Sequence


# External libraries
import numpy as np
from scipy import sparse


# Internal libraries
from .definitions import ImageArray
from .orientation import OrientationField

log = logging.getLogger(__name__)

# Clustering
# 1. Each pixel with non-zero similarity with other pixels gets assigned a cluster
# 2. This single pixel cluster has


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

    def get_mask(self, num_rows: int, num_columns: int) -> ImageArray:
        row_indices, column_indices = np.unravel_index(
            self.get_points(), (num_rows, num_columns)
        )
        mask = np.zeros((num_rows, num_columns))
        mask[row_indices, column_indices] = 1
        return mask


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
) -> Sequence[Cluster]:
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
            log.debug(
                f"Cluster sizes are {first_cluster.size} and {second_cluster.size}"
            )
            log.debug("PAST THE THRESHOLD")
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

    return clusters
