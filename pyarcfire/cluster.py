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
    def __init__(self, seed: int) -> None:
        self._points: list[int] = [seed]

    def __str__(self) -> str:
        return f"Cluster(size={self.size})"

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
    clusters = {idx: Cluster(idx) for idx in points}
    row_indices, column_indices = similarity_matrix.nonzero()
    cluster_queue: PriorityQueue[ClusterPair] = PriorityQueue()
    for row_idx, column_idx in zip(row_indices, column_indices):
        similarity = similarity_matrix[row_idx, column_idx]
        assert isinstance(similarity, float)
        pair = ClusterPair(
            similarity=similarity, first=clusters[row_idx], second=clusters[column_idx]
        )
        cluster_queue.put(pair)
    log.debug(cluster_queue.get())
    return []
