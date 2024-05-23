# Standard libraries
import os
from typing import Sequence

# External libraries
import numpy as np
import scipy.io
from skimage import filters

# Internal libraries
from .debug_utils import benchmark
from .definitions import ImageFloatArray, ImageFloatArraySequence
from .orientation import OrientationField, generate_orientation_fields
from .similarity import generate_similarity_matrix
from .cluster import Cluster, generate_hac_tree
from .merge_fit import merge_clusters_by_fit


class ClusterSpiralResult:
    def __init__(
        self,
        image: ImageFloatArray,
        unsharp_image: ImageFloatArray,
        field: OrientationField,
        arrays: ImageFloatArraySequence,
    ) -> None:
        self._image: ImageFloatArray = image
        self._unsharp_image: ImageFloatArray = unsharp_image
        self._arrays: ImageFloatArraySequence = arrays
        self._field: OrientationField = field
        self._sizes: Sequence[int] = tuple(
            [
                np.count_nonzero(self._arrays[:, :, idx])
                for idx in range(self._arrays.shape[2])
            ]
        )

    def get_image(self) -> ImageFloatArray:
        return self._image

    def get_unsharp_image(self) -> ImageFloatArray:
        return self._unsharp_image

    def get_field(self) -> OrientationField:
        return self._field

    def get_sizes(self) -> Sequence[int]:
        return self._sizes

    def get_cluster_array(self, cluster_idx: int) -> tuple[ImageFloatArray, int]:
        assert cluster_idx in range(self._arrays.shape[2])
        return (self._arrays[:, :, cluster_idx], self._sizes[cluster_idx])

    def get_cluster_arrays(self) -> ImageFloatArraySequence:
        return self._arrays

    def dump(self, path: str) -> None:
        extension = os.path.splitext(path)[1].lstrip(".")
        match extension:
            case "npy":
                np.save(path, self._arrays)
            case "mat":
                scipy.io.savemat(path, {"image": self._arrays})
            case _:
                pass


@benchmark
def detect_spirals_in_image(
    image: ImageFloatArray,
    unsharp_mask_radius: float,
    unsharp_mask_amount: float,
    stop_threshold: float,
) -> ClusterSpiralResult:
    unsharp_image = filters.unsharp_mask(
        image, radius=unsharp_mask_radius, amount=unsharp_mask_amount
    )
    field = generate_orientation_fields(unsharp_image)
    matrix = generate_similarity_matrix(field, stop_threshold)
    clusters = generate_hac_tree(matrix.tocsr(), image, stop_threshold)  # type:ignore
    cluster_arrays = Cluster.list_to_array(clusters, image)
    merged_clusters = merge_clusters_by_fit(cluster_arrays)
    return ClusterSpiralResult(image, unsharp_image, field, merged_clusters)
