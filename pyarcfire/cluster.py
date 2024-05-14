from __future__ import annotations

# Standard libraries
import logging
from typing import Sequence

# External libraries
import numpy as np
from numpy import typing as npt
from scipy import sparse

# Internal libraries
from .definitions import ImageArray, ImageArraySequence
from .merge import calculate_arc_merge_error
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
    def from_2d_array(array: ImageArray) -> Cluster:
        (points,) = np.nonzero(array.flatten())
        return Cluster(list(points.astype(int)))

    @staticmethod
    def from_3d_array(array: ImageArraySequence) -> Sequence[Cluster]:
        num_clusters: int = array.shape[2]
        clusters = []
        for cluster_idx in range(num_clusters):
            clusters.append(Cluster.from_2d_array(array[:, :, cluster_idx]))
        return clusters

    @staticmethod
    def list_to_array(
        clusters: Sequence[Cluster], image: ImageArray
    ) -> ImageArraySequence:
        array_list = [cluster.get_masked_image(image) for cluster in clusters]
        return np.dstack(array_list)

    @staticmethod
    def combine(first: Cluster, second: Cluster) -> Cluster:
        all_points = first._points + second._points
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

    def get_masked_image(self, image: ImageArray) -> ImageArray:
        num_rows, num_columns = image.shape
        row_indices, column_indices = np.unravel_index(
            self.get_points(), (num_rows, num_columns)
        )
        mask = np.ones_like(image, dtype=np.bool_)
        mask[row_indices, column_indices] = False
        masked_image = image.copy()
        masked_image[mask] = 0
        return masked_image

    def get_mask(self, num_rows: int, num_columns: int) -> npt.NDArray[np.bool_]:
        row_indices, column_indices = np.unravel_index(
            self.get_points(), (num_rows, num_columns)
        )
        mask = np.zeros((num_rows, num_columns), dtype=np.bool_)
        mask[row_indices, column_indices] = True
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
    similarity_matrix: sparse.csr_matrix,
    image: ImageArray,
    orientation: OrientationField,
    stop_threshold: float,
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

    # Diagnostics
    check_arc_merge_count: int = 0
    merge_stop_count: int = 0

    # Indices of pixels which have a non-zero similarity with another pixel
    points = similarity_matrix.sum(axis=0).nonzero()[1]
    clusters = {idx: Cluster([idx]) for idx in points}
    while True:
        max_idx = similarity_matrix.argmax()
        unraveled_idx = np.unravel_index(max_idx, similarity_matrix.get_shape())
        first_idx = int(unraveled_idx[0])
        second_idx = int(unraveled_idx[1])

        # Leave loop if there is none left
        value = float(similarity_matrix[first_idx, second_idx])  # type:ignore
        if value < stop_threshold:
            break

        first_cluster = clusters[first_idx]
        second_cluster = clusters[second_idx]
        if (
            min(first_cluster.size, second_cluster.size)
            > merge_check_minimum_cluster_size
        ):
            check_arc_merge_count += 1
            first_cluster_array = first_cluster.get_masked_image(image)
            second_cluster_array = second_cluster.get_masked_image(image)
            merge_error = calculate_arc_merge_error(
                first_cluster_array, second_cluster_array
            )
            if merge_error > error_ratio_threshold:
                merge_stop_count += 1
                similarity_matrix[first_idx, second_idx] = 0
                similarity_matrix[second_idx, first_idx] = 0
                continue

        target_cluster = Cluster.combine(first_cluster, second_cluster)
        clusters[first_idx] = target_cluster
        del clusters[second_idx]

        similarity_matrix = _update_similarity_matrix(
            similarity_matrix, first_idx, second_idx
        )
        similarity_matrix = _clear_similarity_matrix_row_column(
            similarity_matrix, second_idx
        )

    log.debug(f"Number of clusters = {len(clusters)}")
    log.debug(f"Checked {check_arc_merge_count} possible cluster arc merges")
    log.debug(f"Stopped {merge_stop_count} arc merges")
    return list(clusters.values())


def _update_similarity_matrix(
    similarity_matrix: sparse.csr_matrix, target_idx: int, source_idx: int
) -> sparse.csr_matrix:
    # Merged cluster similarity is the maximum possible similarity from the set of cluster points
    old_similarity_values = similarity_matrix[target_idx, :]
    # The updated values max(Ti, Si)
    new_similarity_values = similarity_matrix[target_idx, :].maximum(
        similarity_matrix[source_idx, :]
    )
    # Remove self-similarity
    new_similarity_values[0, target_idx] = 0

    desired_target_array = np.asarray(
        new_similarity_values[new_similarity_values.nonzero()]
    ).flatten()
    # AIM: Construct matrix such that when added, the target row and column is updated to the desired values
    # NOTE: Have to handle when the target row/column already have the desired value

    add_matrix_values = new_similarity_values - old_similarity_values
    if add_matrix_values.count_nonzero() == 0:  # type:ignore
        new_matrix = similarity_matrix
    else:
        _, add_matrix_indices = add_matrix_values.nonzero()
        add_matrix_values = np.asarray(
            add_matrix_values[add_matrix_values.nonzero()]
        ).flatten()
        num_nonzero = len(add_matrix_indices)
        add_matrix_row_indices = np.tile(add_matrix_indices, 2)
        add_matrix_row_indices[num_nonzero:] = target_idx
        add_matrix_column_indices = np.tile(add_matrix_indices, 2)
        add_matrix_column_indices[:num_nonzero] = target_idx
        add_matrix = sparse.coo_matrix(
            (
                np.tile(add_matrix_values, 2),
                (
                    add_matrix_row_indices,
                    add_matrix_column_indices,
                ),
            ),
            shape=similarity_matrix.shape,
        )
        new_matrix = similarity_matrix + add_matrix

    target_row = np.asarray(
        new_matrix.getrow(target_idx)[new_matrix.getrow(target_idx).nonzero()]
    ).flatten()
    target_column = np.asarray(
        new_matrix.getcol(target_idx)[new_matrix.getcol(target_idx).nonzero()]
    ).flatten()
    row_on_target = np.allclose(target_row, desired_target_array)
    assert row_on_target, "Row not on target"
    column_on_target = np.allclose(target_column, desired_target_array)
    assert column_on_target, "Column not on target"

    return new_matrix


def _clear_similarity_matrix_row_column(
    similarity_matrix: sparse.csr_matrix, clear_idx: int
) -> sparse.csr_matrix:
    values = np.asarray(
        similarity_matrix.getrow(clear_idx)[
            similarity_matrix.getrow(clear_idx).nonzero()
        ]
    ).flatten()
    indices = similarity_matrix.getrow(clear_idx).nonzero()[1]

    num_nonzero = len(indices)
    clear_matrix_row_indices = np.tile(indices, 2)
    clear_matrix_row_indices[num_nonzero:] = clear_idx
    clear_matrix_column_indices = np.tile(indices, 2)
    clear_matrix_column_indices[:num_nonzero] = clear_idx
    clear_matrix = sparse.coo_matrix(
        (
            np.tile(values, 2),
            (
                clear_matrix_row_indices,
                clear_matrix_column_indices,
            ),
        ),
        shape=similarity_matrix.shape,
    )
    new_matrix = similarity_matrix - clear_matrix

    is_symmetric = (new_matrix - new_matrix.transpose()).count_nonzero() == 0  # type:ignore
    assert is_symmetric, "Similarity matrix is not symmetric!"
    return new_matrix
