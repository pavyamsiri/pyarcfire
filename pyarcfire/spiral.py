import logging
import os
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.io
from numpy.typing import NDArray
from skimage import filters

from pyarcfire.arc.utils import get_polar_coordinates

from .arc import LogSpiralFitResult, fit_spiral_to_image
from .cluster import GenerateClustersSettings, generate_clusters
from .debug_utils import benchmark
from .merge_fit import MergeClustersByFitSettings, merge_clusters_by_fit
from .orientation import (
    GenerateOrientationFieldSettings,
    OrientationField,
    generate_orientation_fields,
)
from .similarity import GenerateSimilarityMatrixSettings, generate_similarity_matrix

log: logging.Logger = logging.getLogger(__name__)


FloatType = np.float32


@dataclass
class UnsharpMaskSettings:
    radius: float = 25
    amount: float = 6


class ClusterSpiralResult:
    def __init__(
        self,
        image: NDArray[FloatType],
        unsharp_image: NDArray[FloatType],
        field: OrientationField,
        cluster_masks: NDArray[FloatType],
        unsharp_mask_settings: UnsharpMaskSettings,
        orientation_field_settings: GenerateOrientationFieldSettings,
        similarity_matrix_settings: GenerateSimilarityMatrixSettings,
        generate_cluster_settings: GenerateClustersSettings,
        merge_clusters_by_fit_settings: MergeClustersByFitSettings,
    ) -> None:
        self._image: NDArray[FloatType] = image
        self._unsharp_image: NDArray[FloatType] = unsharp_image
        self._cluster_masks: NDArray[FloatType] = cluster_masks
        self._field: OrientationField = field
        self._sizes: tuple[int, ...] = tuple(
            [
                np.count_nonzero(self._cluster_masks[:, :, idx])
                for idx in range(self._cluster_masks.shape[2])
            ]
        )

        # Settings
        self._unsharp_mask_settings: UnsharpMaskSettings = unsharp_mask_settings
        self._orientation_field_settings: GenerateOrientationFieldSettings = (
            orientation_field_settings
        )
        self._similarity_matrix_settings: GenerateSimilarityMatrixSettings = (
            similarity_matrix_settings
        )
        self._generate_cluster_settings: GenerateClustersSettings = (
            generate_cluster_settings
        )
        self._merge_clusters_by_fit_settings: MergeClustersByFitSettings = (
            merge_clusters_by_fit_settings
        )

        # Cache
        self._spiral_cache: dict[int, LogSpiralFitResult[FloatType]] = {}

    def __str__(self) -> str:
        return f"ClusterSpiralResult(num_clusters={self.get_num_clusters()})"

    @property
    def unsharp_mask_settings(self) -> UnsharpMaskSettings:
        return self._unsharp_mask_settings

    @property
    def orientation_field_settings(self) -> GenerateOrientationFieldSettings:
        return self._orientation_field_settings

    @property
    def similarity_matrix_settings(self) -> GenerateSimilarityMatrixSettings:
        return self._similarity_matrix_settings

    @property
    def generate_cluster_settings(self) -> GenerateClustersSettings:
        return self._generate_cluster_settings

    @property
    def merge_clusters_by_fit_settings(self) -> MergeClustersByFitSettings:
        return self._merge_clusters_by_fit_settings

    def get_num_clusters(self) -> int:
        return self._cluster_masks.shape[2]

    def get_image(self) -> NDArray[FloatType]:
        return self._image

    def get_unsharp_image(self) -> NDArray[FloatType]:
        return self._unsharp_image

    def get_field(self) -> OrientationField:
        return self._field

    def get_sizes(self) -> tuple[int, ...]:
        return self._sizes

    def get_cluster_array(self, cluster_idx: int) -> tuple[NDArray[FloatType], int]:
        return (self._cluster_masks[:, :, cluster_idx], self._sizes[cluster_idx])

    def get_cluster_arrays(self) -> NDArray[FloatType]:
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

    def get_spiral_of(
        self, cluster_idx: int, num_points: int = 100
    ) -> tuple[NDArray[FloatType], NDArray[FloatType]]:
        if cluster_idx not in range(self.get_num_clusters()):
            raise ValueError("Expected a valid cluster index!")

        if cluster_idx not in self._spiral_cache:
            current_array, _ = self.get_cluster_array(cluster_idx)
            self._spiral_cache[cluster_idx] = fit_spiral_to_image(current_array)
        spiral_fit = self._spiral_cache[cluster_idx]
        x, y = spiral_fit.calculate_cartesian_coordinates(num_points)
        return x, y

    def _get_fit(self, cluster_idx: int) -> LogSpiralFitResult[FloatType]:
        if cluster_idx not in self._spiral_cache:
            current_array, _ = self.get_cluster_array(cluster_idx)
            self._spiral_cache[cluster_idx] = fit_spiral_to_image(current_array)
        return self._spiral_cache[cluster_idx]

    def get_spirals(
        self, num_points: int
    ) -> Generator[tuple[NDArray[FloatType], NDArray[FloatType]], None, None]:
        num_clusters: int = self.get_num_clusters()
        for cluster_idx in range(num_clusters):
            spiral_fit = self._get_fit(cluster_idx)
            x, y = spiral_fit.calculate_cartesian_coordinates(num_points)
            yield x, y

    def get_arc_bounds(self, cluster_idx: int) -> tuple[float, float]:
        current_fit = self._get_fit(cluster_idx)
        start_angle = current_fit.offset
        end_angle = start_angle + current_fit.arc_bounds[1]
        return (start_angle, end_angle)

    def calculate_fit_error_to_cluster(
        self,
        calculate_radii: Callable[[NDArray[FloatType]], NDArray[FloatType]],
        cluster_idx: int,
    ) -> NDArray[FloatType]:
        current_array, _ = self.get_cluster_array(cluster_idx)
        radii, theta, weights = get_polar_coordinates(current_array)
        result = np.multiply(
            np.sqrt(weights),
            (radii - calculate_radii(theta)),
        )
        return result


@benchmark
def detect_spirals_in_image(
    image: NDArray[FloatType],
    unsharp_mask_settings: UnsharpMaskSettings = UnsharpMaskSettings(),
    orientation_field_settings: GenerateOrientationFieldSettings = GenerateOrientationFieldSettings(),
    similarity_matrix_settings: GenerateSimilarityMatrixSettings = GenerateSimilarityMatrixSettings(),
    generate_clusters_settings: GenerateClustersSettings = GenerateClustersSettings(),
    merge_clusters_by_fit_settings: MergeClustersByFitSettings = MergeClustersByFitSettings(),
) -> Optional[ClusterSpiralResult]:
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
    cluster_arrays: Optional[NDArray[FloatType]] = generate_clusters(
        image,
        matrix.tocsr(),
        stop_threshold=generate_clusters_settings.stop_threshold,
        error_ratio_threshold=generate_clusters_settings.error_ratio_threshold,
        merge_check_minimum_cluster_size=generate_clusters_settings.merge_check_minimum_cluster_size,
        minimum_cluster_size=generate_clusters_settings.minimum_cluster_size,
        remove_central_cluster=generate_clusters_settings.remove_central_cluster,
    )
    log.info("[cyan]PROGRESS[/cyan]: Done generating clusters.")
    if cluster_arrays is None:
        return None

    # Do some final merges based on fit
    log.info("[cyan]PROGRESS[/cyan]: Merging clusters by fit...")
    merged_clusters = merge_clusters_by_fit(
        cluster_arrays, merge_clusters_by_fit_settings.stop_threshold
    )
    log.info("[cyan]PROGRESS[/cyan]: Done merging clusters by fit.")

    return ClusterSpiralResult(
        image=image,
        unsharp_image=unsharp_image,
        field=field,
        cluster_masks=merged_clusters,
        unsharp_mask_settings=unsharp_mask_settings,
        orientation_field_settings=orientation_field_settings,
        similarity_matrix_settings=similarity_matrix_settings,
        generate_cluster_settings=generate_clusters_settings,
        merge_clusters_by_fit_settings=merge_clusters_by_fit_settings,
    )
