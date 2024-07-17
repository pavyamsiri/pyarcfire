"""Functions to merge clusters together by considering how spirals will fit them."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.ndimage import distance_transform_edt

from .debug_utils import benchmark
from .merge import calculate_arc_merge_error

log: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import MutableSequence, Sequence

    from numpy.typing import NDArray

    FloatType = TypeVar("FloatType", np.float32, np.float64)


@dataclass
class MergeClustersByFitSettings:
    """Settings to configure merge_clusters_by_fit.

    Attributes
    ----------
    stop_threshold : float
        The maximum merge error ratio before stopping merges.

    """

    stop_threshold: float = 2.5


DEFAULT_MERGE_CLUSTER_BY_FIT_SETTINGS: MergeClustersByFitSettings = MergeClustersByFitSettings()


@benchmark
def merge_clusters_by_fit(
    clusters: Sequence[NDArray[FloatType]],
    stop_threshold: float,
) -> Sequence[NDArray[FloatType]]:
    """Merge clusters by if they are fit spirals decently well when combined.

    Parameters
    ----------
    clusters : Sequence[NDArray[FloatType]]
        The clusters stored as series of masked images.
    stop_threshold : float
        The maximum allowed distance between clusters to be merged.

    Returns
    -------
    merged_clusters : Sequence[NDArray[FloatType]]
        The clusters after being merged.

    """
    assert len(clusters) > 0
    # Maximum pixel distance
    num_rows, num_columns = clusters[0].shape
    max_pixel_distance = np.mean([num_rows, num_columns]).astype(float) / 20

    # Fit spirals to each cluster
    num_clusters: int = len(clusters)
    cluster_list: MutableSequence[NDArray[FloatType] | None] = list(clusters)

    # Compute distances between each cluster
    cluster_distances = np.full((num_clusters, num_clusters), np.inf, dtype=np.float32)
    for source_idx in range(num_clusters):
        for target_idx in range(source_idx + 1, num_clusters):
            left_array = cluster_list[source_idx]
            right_array = cluster_list[target_idx]
            assert left_array is not None, "Should not be None because it was just set."
            assert right_array is not None, "Should not be None because it was just set."
            cluster_distances[source_idx, target_idx] = _calculate_cluster_distance(
                left_array,
                right_array,
                max_pixel_distance,
            )

    num_merges: int = 0
    while True:
        # Pop the smallest distance
        min_idx = cluster_distances.argmin()
        unravelled_index = np.unravel_index(min_idx, cluster_distances.shape)
        first_idx = int(unravelled_index[0])
        second_idx = int(unravelled_index[1])
        value = cluster_distances[first_idx, second_idx]

        # Distance value too large
        if value > stop_threshold:
            break

        # Merge clusters
        num_merges += 1

        first_cluster_array = cluster_list[first_idx]
        second_cluster_array = cluster_list[second_idx]
        assert first_cluster_array is not None, "Deleted clusters should not have a finite similarity!"
        assert second_cluster_array is not None, "Deleted clusters should not have a finite similarity!"
        combined_cluster_array = first_cluster_array + second_cluster_array
        cluster_list[second_idx] = None
        cluster_distances[:, second_idx] = np.inf
        cluster_distances[second_idx, :] = np.inf

        # Update cluster dictionary
        cluster_list[first_idx] = combined_cluster_array.astype(
            first_cluster_array.dtype,
        )

        # Update distances
        for other_idx in range(num_clusters):
            if cluster_list[other_idx] is None or other_idx == first_idx:
                continue
            left_idx = min(first_idx, other_idx)
            right_idx = max(first_idx, other_idx)
            left_array = cluster_list[left_idx]
            right_array = cluster_list[right_idx]
            assert left_array is not None, "Accessing a deleted cluster should be impossible"
            assert right_array is not None, "Accessing a deleted cluster should be impossible"
            cluster_distances[left_idx, right_idx] = _calculate_cluster_distance(
                left_array,
                right_array,
                max_pixel_distance,
            )
    log.info("[green]DIAGNOST[/green]: Merged %d clusters by fit", num_merges)
    # Combined clusters into arrays
    return [cluster for cluster in cluster_list if cluster is not None]


def _calculate_cluster_distance(
    first_cluster_array: NDArray[FloatType],
    second_cluster_array: NDArray[FloatType],
    max_pixel_distance: float,
) -> float:
    """Calculate the "distance" between two clusters.

    Parameters
    ----------
    first_cluster_array : NDArray[FloatType]
        The first cluster in the form of an array.
    second_cluster_array : NDArray[FloatType]
        The second cluster in the form of an array.
    max_pixel_distance : float
        The maximum allowed distance in pixels between the two clusters for them
        to have finite distance.

    Returns
    -------
    distance : float
        The distance between the two clusters.

    Notes
    -----
    The distance here is the merge error ratio of the two clusters. This is a measure of how
    well a merged cluster fits a spiral compared to the two clusters fitted separately.

    """
    # Compute pixel distances to first cluster
    distances = distance_transform_edt(first_cluster_array == 0, return_distances=True)
    # Mask the distance matrix using the second cluster as a mask
    distances = distances[second_cluster_array > 0]

    # Only compute if the second cluster is close enough to the first cluster
    if len(distances) > 0 and distances.min() <= max_pixel_distance:
        return calculate_arc_merge_error(first_cluster_array, second_cluster_array)
    return np.inf
