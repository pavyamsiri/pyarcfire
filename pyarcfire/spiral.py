# Standard libraries
from dataclasses import dataclass
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
from .orientation import (
    GenerateOrientationFieldSettings,
    OrientationField,
    generate_orientation_fields,
)
from .similarity import GenerateSimilarityMatrixSettings, generate_similarity_matrix
from .cluster import GenerateClustersSettings, generate_clusters
from .merge_fit import MergeClustersByFitSettings, merge_clusters_by_fit


log: logging.Logger = logging.getLogger(__name__)


@dataclass
class UnsharpMaskSettings:
    radius: float = 25
    amount: float = 6


class ClusterSpiralResult:
    def __init__(
        self,
        image: Array2D,
        field: OrientationField,
        cluster_masks: Array3D,
    ) -> None:
        self._image: Array2D = image
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
            log.warning(
                f"[yellow]FILESYST[/yellow]: Can not dump due to unknown extension [yellow]{extension}[/yellow]"
            )
            return
        log.info(f"[yellow]FILESYST[/yellow]: Dumped masks to [yellow]{path}[/yellow]")


@benchmark
def detect_spirals_in_image(
    image: Array2D,
    unsharp_mask_settings: UnsharpMaskSettings,
    orientation_field_settings: GenerateOrientationFieldSettings,
    similarity_matrix_settings: GenerateSimilarityMatrixSettings,
    generate_clusters_settings: GenerateClustersSettings,
    merge_clusters_by_fit_settings: MergeClustersByFitSettings,
) -> ClusterSpiralResult:
    # Unsharp phase
    unsharp_image = filters.unsharp_mask(
        image, radius=unsharp_mask_settings.radius, amount=unsharp_mask_settings.amount
    )

    # Generate orientation fields
    log.info("[cyan]PROGRESS[/cyan]: Generating orientation field...")
    field = generate_orientation_fields(
        unsharp_image,
        num_orientation_field_levels=orientation_field_settings.num_orientation_field_levels,
        neighbour_distance=orientation_field_settings.neighbour_distance,
        kernel_radius=orientation_field_settings.kernel_radius,
    )
    log.info("[cyan]PROGRESS[/cyan]: Done generating orientation field.")

    # Generate similarity matrix
    log.info("[cyan]PROGRESS[/cyan]: Generating similarity matrix...")
    matrix = generate_similarity_matrix(
        field, similarity_matrix_settings.similarity_cutoff
    )
    log.info("[cyan]PROGRESS[/cyan]: Done generating similarity matrix.")

    # Merge clusters via HAC
    log.info("[cyan]PROGRESS[/cyan]: Generating clusters...")
    cluster_arrays = generate_clusters(
        image,
        matrix.tocsr(),
        stop_threshold=generate_clusters_settings.stop_threshold,
        error_ratio_threshold=generate_clusters_settings.error_ratio_threshold,
        merge_check_minimum_cluster_size=generate_clusters_settings.merge_check_minimum_cluster_size,
        minimum_cluster_size=generate_clusters_settings.minimum_cluster_size,
        remove_central_cluster=generate_clusters_settings.remove_central_cluster,
    )
    log.info("[cyan]PROGRESS[/cyan]: Done generating clusters.")

    # Do some final merges based on fit
    log.info("[cyan]PROGRESS[/cyan]: Merging clusters by fit...")
    merged_clusters = merge_clusters_by_fit(
        cluster_arrays, merge_clusters_by_fit_settings.stop_threshold
    )
    log.info("[cyan]PROGRESS[/cyan]: Done merging clusters by fit.")

    return ClusterSpiralResult(
        image,
        field,
        merged_clusters,
    )
