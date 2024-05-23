"""This module contains functions to merge clusters together by considering how spirals will fit them."""

# Standard libraries
import logging

# External libraries
import numpy as np
from scipy.ndimage import distance_transform_edt


# Internal libraries
from .debug_utils import benchmark
from .definitions import ImageFloatArray, ImageFloatArraySequence
from .merge import calculate_arc_merge_error

log = logging.getLogger(__name__)


@benchmark
def merge_clusters_by_fit(
    clusters: ImageFloatArraySequence, stop_threshold: float = 2.5
) -> ImageFloatArraySequence:
    """Merge clusters by if they are fit spirals decently well when combined.

    Parameters
    ----------
    clusters : ImageFloatArraySequence
        The clusters stored as series of masked images.
    stop_threshold : float
        The maximum allowed distance between clusters to be merged.

    Returns
    -------
    merged_clusters : ImageFloatArraySequence
        The clusters after being merged.
    """
    # Maximum pixel distance
    max_pixel_distance = (
        np.mean([clusters.shape[0], clusters.shape[1]]).astype(float) / 20
    )

    # Fit spirals to each cluster
    cluster_dict: dict[int, ImageFloatArray] = {}
    num_clusters: int = clusters.shape[2]
    for cluster_idx in range(num_clusters):
        cluster_dict[cluster_idx] = clusters[:, :, cluster_idx]

    # Compute distances between each cluster
    cluster_distances = np.full((num_clusters, num_clusters), np.inf, dtype=np.float32)
    for first_idx in range(num_clusters):
        for second_idx in range(first_idx + 1, num_clusters):
            left_array = cluster_dict[first_idx][1]
            right_array = cluster_dict[second_idx][1]
            cluster_distances[first_idx, second_idx] = _calculate_cluster_distance(
                left_array, right_array, max_pixel_distance
            )

    num_merges: int = 0
    while True:
        # Pop the smallest distance
        min_idx = cluster_distances.argmin()
        first_idx, second_idx = np.unravel_index(min_idx, cluster_distances.shape)
        value = cluster_distances[first_idx, second_idx]

        # Distance value too large
        if value > stop_threshold:
            break

        # Merge clusters
        num_merges += 1

        first_cluster_array = cluster_dict[first_idx][1]
        second_cluster_array = cluster_dict[second_idx][1]
        combined_cluster_array = first_cluster_array + second_cluster_array
        del cluster_dict[second_idx]
        cluster_distances[:, second_idx] = np.inf
        cluster_distances[second_idx, :] = np.inf

        # Update cluster dictionary
        cluster_dict[first_idx] = combined_cluster_array

        # Update distances
        for other_idx in range(num_clusters):
            if other_idx not in cluster_dict:
                continue
            left_idx = min(first_idx, other_idx)
            right_idx = max(first_idx, other_idx)
            left_array = cluster_dict[int(left_idx)][1]
            right_array = cluster_dict[int(right_idx)][1]
            cluster_distances[left_idx, right_idx] = _calculate_cluster_distance(
                left_array, right_array, max_pixel_distance
            )
    log.debug(f"Merged {num_merges} clusters")
    # Combined clusters into arrays
    merged_clusters: ImageFloatArraySequence = np.dstack(
        [cluster_dict[cluster_idx] for cluster_idx in cluster_dict]
    )
    return merged_clusters


def _calculate_cluster_distance(
    first_cluster_array: ImageFloatArray,
    second_cluster_array: ImageFloatArray,
    max_pixel_distance: float,
) -> float:
    """Calculates the "distance" between two clusters.

    Parameters
    ----------
    first_cluster_array : ImageFloatArray
        The first cluster in the form of an array.
    second_cluster_array : ImageFloatArray
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
    assert isinstance(distances, np.ndarray)
    # Mask the distance matrix using the second cluster as a mask
    distances = distances[second_cluster_array > 0]

    # Only compute if the second cluster is close enough to the first cluster
    if len(distances) > 0 and distances.min() <= max_pixel_distance:
        merge_error = calculate_arc_merge_error(
            first_cluster_array, second_cluster_array
        )
        return merge_error
    return np.inf
