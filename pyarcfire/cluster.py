from __future__ import annotations

# Standard libraries
import dataclasses
from dataclasses import dataclass
import logging
from typing import Sequence
from queue import PriorityQueue


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
    def __init__(self, points: list[int]) -> None:
        self._points: list[int] = points

    def __str__(self) -> str:
        return f"Cluster(size={self.size}, hash={hash(self):X})"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def combine(first: Cluster, second: Cluster) -> Cluster:
        all_points = first.get_points() + second.get_points()
        return Cluster(all_points)

    @property
    def size(self) -> int:
        return len(self._points)

    def get_points(self) -> Sequence[int]:
        return self._points

    def add_point(self, seed: int) -> None:
        assert (
            seed not in self._points
        ), "Clusters must have a set of points i.e. no duplicates"
        self._points.append(seed)

    def get_mask(self, num_rows: int, num_columns: int) -> ImageArray:
        row_indices, column_indices = np.unravel_index(
            self.get_points(), (num_rows, num_columns)
        )
        mask = np.zeros((num_rows, num_columns))
        mask[row_indices, column_indices] = 1
        return mask


@dataclass
class ClusterPair:
    similarity: float
    first: Cluster
    second: Cluster

    def __eq__(self, other: object) -> bool:
        assert isinstance(
            other, ClusterPair
        ), f"Equality is only supported for {self.__class__}"
        return self.similarity == other.similarity

    def __lt__(self, other: object) -> bool:
        assert isinstance(
            other, ClusterPair
        ), f"Less than is only supported for {self.__class__}"
        return self.similarity > other.similarity


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
    similarity_matrix: sparse.csr_matrix,
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

    # Indices of pixels which have a non-zero similarity with another pixel
    points = similarity_matrix.sum(axis=0).nonzero()[1]
    clusters = {idx: Cluster([idx]) for idx in points}
    while True:
        max_idx = similarity_matrix.argmax()
        unraveled_idx = np.unravel_index(max_idx, similarity_matrix.get_shape())
        first_idx = int(unraveled_idx[0])
        second_idx = int(unraveled_idx[1])
        value = float(similarity_matrix[first_idx, second_idx])  # type:ignore
        log.debug(f"{first_idx} <-> {second_idx}")
        log.debug(f"Value = {value}")
        if value < stop_threshold:
            break

        first_cluster = clusters[first_idx]
        second_cluster = clusters[second_idx]
        target_cluster = Cluster.combine(first_cluster, second_cluster)
        clusters[first_idx] = target_cluster
        log.debug(f"Merged cluster has new size = {clusters[first_idx].size}")
        del clusters[second_idx]

        # Merged cluster similarity is the maximum possible similarity from the set of cluster points
        old_similarity_values = similarity_matrix[first_idx, :]
        new_similarity_values = similarity_matrix[first_idx, :].maximum(
            similarity_matrix[second_idx, :]
        )
        # Set self similarity to 0 i.e. points within same merged cluster should have 0 similarity
        log.debug(f"First row = {similarity_matrix.getrow(first_idx)}")
        log.debug(f"First column = {similarity_matrix.getcol(first_idx)}")
        new_similarity_values[0, first_idx] = 0
        new_similarity_values_array = np.asarray(
            new_similarity_values[new_similarity_values.nonzero()]
        ).flatten()

        # Add a term so that the resultant matrix has new similarity values at row/column = first_idx without duplicating additions
        _, old_nonzero_indices = old_similarity_values.nonzero()
        _, new_nonzero_indices = new_similarity_values.nonzero()
        has_old_values = new_nonzero_indices[
            np.isin(new_nonzero_indices, old_nonzero_indices)
        ]
        set_max_values_matrix_values = new_similarity_values
        set_max_values_matrix_values[0, has_old_values] -= old_similarity_values[
            0, has_old_values
        ]
        _, set_max_values_matrix_indices = set_max_values_matrix_values.nonzero()
        num_nonzero = len(set_max_values_matrix_indices)
        set_max_values_matrix_row_indices = np.tile(set_max_values_matrix_indices, 2)
        set_max_values_matrix_row_indices[num_nonzero:] = first_idx
        set_max_values_matrix_column_indices = np.tile(set_max_values_matrix_indices, 2)
        set_max_values_matrix_column_indices[:num_nonzero] = first_idx
        set_max_values_matrix_values_expanded = np.asarray(
            np.tile(
                np.asarray(
                    set_max_values_matrix_values[set_max_values_matrix_values.nonzero()]
                ),  # type:ignore
                2,
            )
        ).flatten()
        log.debug(f"Values = {set_max_values_matrix_values_expanded}")
        set_max_values_matrix = sparse.coo_matrix(
            (
                set_max_values_matrix_values_expanded,
                (
                    set_max_values_matrix_row_indices,
                    set_max_values_matrix_column_indices,
                ),
            ),
            shape=similarity_matrix.shape,
        )
        similarity_matrix += set_max_values_matrix
        first_row = np.asarray(
            similarity_matrix[first_idx, :][similarity_matrix[first_idx, :].nonzero()]
        ).flatten()
        assert np.allclose(first_row, new_similarity_values_array), "Not on target"

        clear_nonzero_indices = np.asarray(
            similarity_matrix[second_idx, :].nonzero()[1]
        )
        num_nonzero = len(clear_nonzero_indices)
        set_clear_values_matrix_row_indices = np.tile(clear_nonzero_indices, 2)
        set_clear_values_matrix_row_indices[num_nonzero:] = second_idx
        set_clear_values_matrix_column_indices = np.tile(clear_nonzero_indices, 2)
        set_clear_values_matrix_column_indices[:num_nonzero] = second_idx
        set_clear_values_matrix_values = -np.asarray(
            np.tile(similarity_matrix[second_idx, clear_nonzero_indices].toarray(), 2)
        ).flatten()
        set_clear_values_matrix = sparse.coo_matrix(
            (
                set_clear_values_matrix_values,
                (
                    set_clear_values_matrix_row_indices,
                    set_clear_values_matrix_column_indices,
                ),
            ),
            shape=similarity_matrix.shape,
        )
        similarity_matrix += set_clear_values_matrix

        is_symmetric = (
            similarity_matrix - similarity_matrix.transpose()
        ).count_nonzero() == 0
        assert is_symmetric, "Similarity matrix is not symmetric!"
    log.debug(f"Number of clusters = {len(clusters)}")
    return list(clusters.values())
