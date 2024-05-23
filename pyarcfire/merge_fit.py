# Standard libraries
import logging

# External libraries
import numpy as np
from scipy.ndimage import distance_transform_edt


# Internal libraries
from .arc import LogSpiralFitResult, fit_spiral_to_image
from .debug_utils import benchmark
from .definitions import ImageFloatArray, ImageFloatArraySequence
from .merge import calculate_arc_merge_error

log = logging.getLogger(__name__)


@benchmark
def merge_clusters_by_fit(
    clusters: ImageFloatArraySequence, stop_threshold: float = 2.5
) -> ImageFloatArraySequence:
    max_pixel_distance = (
        np.mean([clusters.shape[0], clusters.shape[1]]).astype(float) / 20
    )
    cluster_dict: dict[int, tuple[LogSpiralFitResult, ImageFloatArray]] = {}
    num_clusters: int = clusters.shape[2]
    for cluster_idx in range(num_clusters):
        fit = fit_spiral_to_image(clusters[:, :, cluster_idx])
        cluster_dict[cluster_idx] = (fit, clusters[:, :, cluster_idx])
    num_clusters = len(cluster_dict)
    cluster_distances = np.full((num_clusters, num_clusters), np.inf, dtype=np.float32)
    for first_idx in range(num_clusters):
        for second_idx in range(first_idx + 1, num_clusters):
            left_array = cluster_dict[first_idx][1]
            right_array = cluster_dict[second_idx][1]
            log.debug(f"Calculating {first_idx} vs {second_idx}")
            cluster_distances[first_idx, second_idx] = _calculate_cluster_distance(
                left_array, right_array, max_pixel_distance
            )
    num_merges: int = 0
    while True:
        min_idx = cluster_distances.argmin()
        first_idx, second_idx = np.unravel_index(min_idx, cluster_distances.shape)
        value = cluster_distances[first_idx, second_idx]
        if value <= stop_threshold:
            first_cluster_array = cluster_dict[first_idx][1]
            second_cluster_array = cluster_dict[second_idx][1]
            combined_cluster_array = first_cluster_array + second_cluster_array
            del cluster_dict[second_idx]
            cluster_distances[:, second_idx] = np.inf
            cluster_distances[second_idx, :] = np.inf
            num_merges += 1
            new_fit = fit_spiral_to_image(combined_cluster_array)
            cluster_dict[first_idx] = (new_fit, combined_cluster_array)

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
        else:
            break
    log.debug(f"Merged {num_merges} clusters")
    merged_clusters: ImageFloatArraySequence = np.dstack(
        [cluster_dict[cluster_idx][1] for cluster_idx in cluster_dict]
    )
    return merged_clusters


def _calculate_cluster_distance(
    first_cluster_array: ImageFloatArray,
    second_cluster_array: ImageFloatArray,
    max_pixel_distance: float,
) -> float:
    distances = distance_transform_edt(first_cluster_array == 0, return_distances=True)
    assert isinstance(distances, np.ndarray)
    distances = distances[second_cluster_array > 0]

    if len(distances) > 0 and distances.min() <= max_pixel_distance:
        merge_error = calculate_arc_merge_error(
            first_cluster_array, second_cluster_array
        )
        return merge_error
    return np.inf
