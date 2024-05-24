# Standard libraries
import logging
import os
from typing import Sequence

# External libraries
import numpy as np
import scipy.io
from skimage import filters

# Internal libraries
from .debug_utils import benchmark
from .definitions import Array2D, Array3D
from .orientation import OrientationField, generate_orientation_fields
from .similarity import generate_similarity_matrix
from .cluster import generate_clusters
from .merge_fit import merge_clusters_by_fit


log: logging.Logger = logging.getLogger(__name__)


class ClusterSpiralResult:
    def __init__(
        self,
        image: Array2D,
        unsharp_image: tuple[Array2D, float, float],
        field: OrientationField,
        cluster_masks: Array3D,
    ) -> None:
        self._image: Array2D = image
        self._unsharp_image: Array2D = unsharp_image[0]
        self._unsharp_radius: float = unsharp_image[1]
        self._unsharp_amount: float = unsharp_image[2]
        self._cluster_masks: Array3D = cluster_masks
        self._field: OrientationField = field
        self._sizes: Sequence[int] = tuple(
            [
                np.count_nonzero(self._cluster_masks[:, :, idx])
                for idx in range(self._cluster_masks.shape[2])
            ]
        )

    def get_image(self) -> Array2D:
        return self._image

    def get_unsharp_image(self) -> Array2D:
        return self._unsharp_image

    def get_unsharp_mask_properties(self) -> tuple[float, float]:
        return (self._unsharp_radius, self._unsharp_amount)

    def get_field(self) -> OrientationField:
        return self._field

    def get_sizes(self) -> Sequence[int]:
        return self._sizes

    def get_cluster_array(self, cluster_idx: int) -> tuple[Array2D, int]:
        return (self._cluster_masks[:, :, cluster_idx], self._sizes[cluster_idx])

    def get_cluster_arrays(self) -> Array3D:
        return self._cluster_masks

    def dump(self, path: str) -> None:
        extension = os.path.splitext(path)[1].lstrip(".")
        if extension == "npy":
            np.save(path, self._cluster_masks)
        elif extension == "mat":
            scipy.io.savemat(path, {"image": self._cluster_masks})
        else:
            log.warning(f"Unknown extension {extension}. Not dumping.")


@benchmark
def detect_spirals_in_image(
    image: Array2D,
    unsharp_mask_radius: float = 25,
    unsharp_mask_amount: float = 6,
    stop_threshold: float = 0.15,
) -> ClusterSpiralResult:
    unsharp_image = filters.unsharp_mask(
        image, radius=unsharp_mask_radius, amount=unsharp_mask_amount
    )
    log.info("[cyan]PROGRESS[/cyan]: Generating orientation field...")
    field = generate_orientation_fields(unsharp_image)
    log.info("[cyan]PROGRESS[/cyan]: Done generating orientation field.")
    log.info("[cyan]PROGRESS[/cyan]: Generating similarity matrix...")
    matrix = generate_similarity_matrix(field, stop_threshold)
    log.info("[cyan]PROGRESS[/cyan]: Done generating similarity matrix.")

    log.info(
        f"[green]DIAGNOST[/green]: Similarity matrix has {matrix.count_nonzero():,} nonzero elements."
    )  # type:ignore

    # TODO: Delete cluster containing the centre
    log.info("[cyan]PROGRESS[/cyan]: Generating clusters...")
    cluster_arrays = generate_clusters(image, matrix.tocsr(), stop_threshold)  # type:ignore
    log.info("[cyan]PROGRESS[/cyan]: Done generating clusters.")

    log.info("[cyan]PROGRESS[/cyan]: Merging clusters by fit...")
    merged_clusters = merge_clusters_by_fit(cluster_arrays)
    log.info("[cyan]PROGRESS[/cyan]: Done merging clusters by fit.")

    return ClusterSpiralResult(
        image,
        (unsharp_image, unsharp_mask_radius, unsharp_mask_amount),
        field,
        merged_clusters,
    )
