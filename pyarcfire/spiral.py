"""Public API."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io
from numpy import int32
from skimage import filters
from typing_extensions import assert_never

from pyarcfire.arc import Chirality

from .arc import FitErrorKind, LogSpiralFitResult, fit_spiral_to_image
from .arc.utils import get_polar_coordinates
from .assert_utils import (
    verify_data_is_2d,
    verify_data_is_normalized,
)
from .cluster import (
    DEFAULT_CLUSTER_SETTINGS,
    GenerateClustersSettings,
    generate_clusters,
)
from .debug_utils import benchmark
from .merge_fit import (
    DEFAULT_MERGE_CLUSTER_BY_FIT_SETTINGS,
    MergeClustersByFitSettings,
    merge_clusters_by_fit,
)
from .orientation import (
    DEFAULT_ORIENTATION_FIELD_SETTINGS,
    GenerateOrientationFieldSettings,
    OrientationField,
    generate_orientation_fields,
)
from .preprocess import preprocess_image
from .similarity import (
    DEFAULT_SIMILARITY_MATRIX_SETTINGS,
    GenerateSimilarityMatrixSettings,
    generate_similarity_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


FloatType = np.float32

MAX_SIZE_BEFORE_WARN: int = 256


@dataclass
class UnsharpMaskSettings:
    """Settings to determine how the unsharp mask is applied.

    Attributes
    ----------
    radius : float
        The radius of the unsharp mask in pixels.
    amount : float
        The amount of unsharpening.

    """

    radius: float = 25
    amount: float = 6


DEFAULT_UNSHARP_MASK: UnsharpMaskSettings = UnsharpMaskSettings()


@dataclass
class FitErrorSettings:
    """Filter out clusters with poor fit errors.

    Attributes
    ----------
    kind : FitErrorKind
        The normalisation scheme to use.
    max_error : float
        The maximum allowed error.
    pixel_to_distance : float
        The unit conversion factor to translate from pixels to the user's desired units.

    """

    kind: FitErrorKind
    max_error: float
    pixel_to_distance: float


class ClusterSpiralResult:
    """Data structure that contains the results of the spiral finding algorithm."""

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
        """Initialize a ClusterSpiralResult instance.

        Parameters
        ----------
        image : NDArray[FloatType]
            The original input image.
        unsharp_image : NDArray[FloatType]
            The image after applying an unsharp mask.
        field : OrientationField
            The orientation field derived from the input image.
        cluster_masks : NDArray[FloatType]
            A 3D NumPy array where each slice along the third dimension represents a cluster mask.
        unsharp_mask_settings : UnsharpMaskSettings
            Settings used for generating the unsharp mask image.
        orientation_field_settings : GenerateOrientationFieldSettings
            Settings used for generating the orientation field.
        similarity_matrix_settings : GenerateSimilarityMatrixSettings
            Settings used for generating the similarity matrix.
        generate_cluster_settings : GenerateClustersSettings
            Settings used for generating and merging clusters.
        merge_clusters_by_fit_settings : MergeClustersByFitSettings
            Settings used for merging clusters based on fit criteria.

        """
        self._image: NDArray[FloatType] = image
        self._unsharp_image: NDArray[FloatType] = unsharp_image
        self._cluster_masks: NDArray[FloatType] = cluster_masks
        self._field: OrientationField = field
        self._sizes: tuple[int, ...] = tuple(
            [np.count_nonzero(self._cluster_masks[:, :, idx]) for idx in range(self._cluster_masks.shape[2])],
        )

        # Settings
        self._unsharp_mask_settings: UnsharpMaskSettings = unsharp_mask_settings
        self._orientation_field_settings: GenerateOrientationFieldSettings = orientation_field_settings
        self._similarity_matrix_settings: GenerateSimilarityMatrixSettings = similarity_matrix_settings
        self._generate_cluster_settings: GenerateClustersSettings = generate_cluster_settings
        self._merge_clusters_by_fit_settings: MergeClustersByFitSettings = merge_clusters_by_fit_settings

        # Cache
        self._spiral_cache: dict[int, LogSpiralFitResult[FloatType]] = {}

    def __str__(self) -> str:
        """Return a string representation of ClusterSpiralResult."""
        return f"ClusterSpiralResult(num_clusters={self.get_num_clusters()})"

    @property
    def unsharp_mask_settings(self) -> UnsharpMaskSettings:
        """UnsharpMaskSettings: Settings for the unsharp mask."""
        return self._unsharp_mask_settings

    @property
    def orientation_field_settings(self) -> GenerateOrientationFieldSettings:
        """GenerateOrientationFieldSettings: Settings for orientation field generation."""
        return self._orientation_field_settings

    @property
    def similarity_matrix_settings(self) -> GenerateSimilarityMatrixSettings:
        """GenerateSimilarityMatrixSettings: Settings for similarity matrix generation."""
        return self._similarity_matrix_settings

    @property
    def generate_cluster_settings(self) -> GenerateClustersSettings:
        """GenerateClusterSettings: Settings for cluster merging and generation."""
        return self._generate_cluster_settings

    @property
    def merge_clusters_by_fit_settings(self) -> MergeClustersByFitSettings:
        """MergeClustersByFitSettings: Settings for cluster merging by fit."""
        return self._merge_clusters_by_fit_settings

    def get_num_clusters(self) -> int:
        """Return the number of clusters found.

        Returns
        -------
        int
            The number of clusters found.

        """
        return self._cluster_masks.shape[2]

    def get_image_height(self) -> int:
        """Return the original image height.

        Returns
        -------
        width : int
            The image height in pixels.

        """
        return self._image.shape[0]

    def get_image_width(self) -> int:
        """Return the original image width.

        Returns
        -------
        width : int
            The image width in pixels.

        """
        return self._image.shape[1]

    def get_image(self) -> NDArray[FloatType]:
        """Return the original image.

        Returns
        -------
        NDArray[FloatType]
            The original image.

        """
        return self._image

    def get_unsharp_image(self) -> NDArray[FloatType]:
        """Return the unsharpened image.

        Returns
        -------
        NDArray[FloatType]
            The unsharpened image.

        """
        return self._unsharp_image

    def get_field(self) -> OrientationField:
        """Return the orientation field.

        Returns
        -------
        OrientationField
            The orientation field.

        """
        return self._field

    def get_sizes(self) -> tuple[int, ...]:
        """Return the sizes of each cluster.

        Returns
        -------
        tuple[int, ...]
            The size of each cluster.

        """
        return self._sizes

    def get_cluster_mask(self) -> NDArray[int32]:
        """Return the overall cluster mask.

        The cluster mask being an array of integer values corresponding to the indices
        of the clusters the pixels belong to.

        Returns
        -------
        cluster_mask : NDArray[int]
            The clusters as an array where non-negative values correspond to the index of the
            cluster they belong to. Negative values indicate that the pixel belongs to no cluster.

        """
        mask = np.full((self.get_image_height(), self.get_image_width()), -1, dtype=int32)
        for cluster_index in range(self.get_num_clusters()):
            current_mask = self.get_cluster_array(cluster_index)[0]
            mask[current_mask != 0] = cluster_index
        return mask

    def get_cluster_array(self, cluster_idx: int) -> tuple[NDArray[FloatType], int]:
        """Return the cluster as an array along with its size.

        Returns
        -------
        cluster_array : NDArray[FloatType]
            The cluster as an array where the non-zero values are where the cluster exists.
        size : int
            The size of the cluster.

        """
        return (self._cluster_masks[:, :, cluster_idx], self._sizes[cluster_idx])

    def get_cluster_arrays(self) -> NDArray[FloatType]:
        """Return clusters as a 3D array where each slice is a cluster.

        Returns
        -------
        NDArray[FloatType]
            The array containing every cluster as a 2D slice.

        """
        return self._cluster_masks

    def dump(self, path: str) -> None:
        """Dump the cluster array into one of the supported formats.

        Parameters
        ----------
        path : str
            The path to write to.

        Notes
        -----
        The supported formats are currently:
        - npy
            - numpy array file.
        - mat
            - MatLab mat file.

        """
        extension = Path(path).suffix.lstrip(".")
        if extension == "npy":
            np.save(path, self._cluster_masks)
        elif extension == "mat":
            scipy.io.savemat(path, {"image": self._cluster_masks})
        else:
            log.warning(
                "[yellow]FILESYST[/yellow]: Can not dump due to unknown extension [yellow]%s[/yellow]",
                extension,
            )
            return
        log.info(
            "[yellow]FILESYST[/yellow]: Dumped masks to [yellow]%s[/yellow]",
            extension,
        )

    def get_spiral_of(
        self,
        cluster_idx: int,
        num_points: int = 100,
        pixel_to_distance: float = 1,
        *,
        flip_y: bool = False,
    ) -> tuple[NDArray[FloatType], NDArray[FloatType]]:
        """Return the x and y coordinates of a cluster's spiral arc.

        Parameters
        ----------
        cluster_idx : int
            The index of the cluster to get the arc coordinates of.
        num_points : int
            The number of points to approximate the arc with.
        pixel_to_distance : float
            The unit conversion factor to translate from pixels to the user's desired units.
        flip_y : bool
            If the flag is set, flip the y coordinate.

        Returns
        -------
        x : NDArray[FloatType]
            The x coordinate of the arc.
        y : NDArray[FloatType]
            The y coordinate of the arc.

        """
        if cluster_idx not in range(self.get_num_clusters()):
            msg = "Expected a valid cluster index!"
            raise ValueError(msg)

        if cluster_idx not in self._spiral_cache:
            current_array, _ = self.get_cluster_array(cluster_idx)
            self._spiral_cache[cluster_idx] = fit_spiral_to_image(current_array)
        spiral_fit = self._spiral_cache[cluster_idx]
        x, y = spiral_fit.calculate_cartesian_coordinates(
            num_points,
            pixel_to_distance,
            flip_y=flip_y,
        )
        return x, y

    def get_fit(self, cluster_idx: int) -> LogSpiralFitResult[FloatType]:
        """Return the log spiral fit associated with the given cluster index.

        Parameters
        ----------
        cluster_idx : int
            The index of the cluster to get the fit of.

        Returns
        -------
        result : LogSpiralFitResult[FloatType]
            The log spiral fit result.

        """
        if cluster_idx not in self._spiral_cache:
            current_array, _ = self.get_cluster_array(cluster_idx)
            self._spiral_cache[cluster_idx] = fit_spiral_to_image(current_array)
        return self._spiral_cache[cluster_idx]

    def get_spirals(
        self,
        num_points: int,
        pixel_to_distance: float,
        *,
        flip_y: bool = False,
    ) -> Generator[tuple[NDArray[FloatType], NDArray[FloatType]], None, None]:
        """Generate Cartesian coordinates for spirals fitted to each cluster.

        Parameters
        ----------
        num_points : int
            The number of points to generate for each spiral.
        pixel_to_distance : float
            Conversion factor from pixel units to physical distance units.
        flip_y : bool, optional
            Whether to flip the y-coordinate, by default False.

        Yields
        ------
        x : NDArray[FloatType]
            The x coordinate of the arc.
        y : NDArray[FloatType]
            The y coordinate of the arc.

        """
        num_clusters: int = self.get_num_clusters()
        for cluster_idx in range(num_clusters):
            spiral_fit = self.get_fit(cluster_idx)
            x, y = spiral_fit.calculate_cartesian_coordinates(
                num_points,
                pixel_to_distance,
                flip_y=flip_y,
            )
            yield x, y

    def get_spirals_and_clusters(
        self,
        num_points: int,
        pixel_to_distance: float,
        *,
        flip_y: bool = False,
    ) -> Generator[tuple[NDArray[FloatType], NDArray[FloatType], NDArray[FloatType]], None, None]:
        """Generate Cartesian coordinates for spirals fitted to each cluster.

        Parameters
        ----------
        num_points : int
            The number of points to generate for each spiral.
        pixel_to_distance : float
            Conversion factor from pixel units to physical distance units.
        flip_y : bool, optional
            Whether to flip the y-coordinate, by default False.

        Yields
        ------
        x : NDArray[FloatType]
            The x coordinate of the arc.
        y : NDArray[FloatType]
            The y coordinate of the arc.
        cluster_array : NDArray[FloatType]
            The cluster in array format.

        """
        num_clusters: int = self.get_num_clusters()
        for cluster_idx in range(num_clusters):
            spiral_fit = self.get_fit(cluster_idx)
            x, y = spiral_fit.calculate_cartesian_coordinates(
                num_points,
                pixel_to_distance,
                flip_y=flip_y,
            )
            yield x, y, self.get_cluster_array(cluster_idx)[0]

    def get_arc_bounds(self, cluster_idx: int) -> tuple[float, float]:
        """Return the arc bounds in radians for a given cluster.

        Parameters
        ----------
        cluster_idx : int
            The index of the cluster.

        Returns
        -------
        arc_bounds : tuple[float, float]
            The arc bounds.

        """
        current_fit = self.get_fit(cluster_idx)
        return current_fit.arc_bounds

    def calculate_fit_error_to_cluster(
        self,
        calculate_radii: Callable[[NDArray[FloatType]], NDArray[FloatType]],
        cluster_idx: int,
        *,
        pixel_to_distance: float,
        fit_error_kind: FitErrorKind = FitErrorKind.NONORM,
    ) -> float:
        """Calculate the residuals of the given function with respect to a cluster.

        Parameters
        ----------
        calculate_radii : Callable[[NDArray[FloatType]], NDArray[FloatType]]
            A function that takes in an array of angles and returns radii.
        cluster_idx : int
            The index of the cluster.
        pixel_to_distance : float
            Conversion factor from pixel units to physical distance units.
        fit_error_kind : FitErrorKind
            The kind of normalisation scheme to apply to the fit error before returning it.

        Returns
        -------
        residuals : NDArray[FloatType]
            The residuals.

        """
        current_array, _ = self.get_cluster_array(cluster_idx)
        radii, theta, weights = get_polar_coordinates(current_array)
        residuals = np.multiply(
            np.sqrt(weights),
            (pixel_to_distance * radii - calculate_radii(theta)),
        )
        total_error = np.sum(np.square(residuals))
        if fit_error_kind == FitErrorKind.NONORM:
            pass
        elif fit_error_kind == FitErrorKind.NUM_PIXELS_NORM:
            num_pixels = np.count_nonzero(current_array)
            total_error /= num_pixels
        else:
            assert_never(fit_error_kind)
        return total_error

    def get_dominant_chirality(self) -> Chirality:
        """Determine the dominant chirality by arc length weighted vote.

        Returns
        -------
        dominant_chirality : Chirality
            The dominant chirality.

        """
        fits = [self.get_fit(cluster_idx) for cluster_idx in range(self.get_num_clusters())]
        arc_lengths = np.asarray([fit.arc_length for fit in fits])
        chiralities = np.asarray([fit.chirality_sign for fit in fits])
        result = np.sum(arc_lengths * chiralities)
        dominant_chirality: Chirality
        if result > 0:
            dominant_chirality = Chirality.CLOCKWISE
        elif result < 0:
            dominant_chirality = Chirality.COUNTER_CLOCKWISE
        else:
            dominant_chirality = Chirality.NONE
        return dominant_chirality

    def get_overall_pitch_angle(self) -> float:
        """Determine the overall pitch angle in radians.

        The overall pitch angle is the average pitch angle of all the arcs that agree with the
        dominant chirality.

        Returns
        -------
        overall_pitch_angle : float
            The overall pitch angle in radians.

        """
        dominant_chirality = self.get_dominant_chirality()
        fits = [self.get_fit(cluster_idx) for cluster_idx in range(self.get_num_clusters())]
        fits = [fit for fit in fits if fit.chirality == dominant_chirality]
        pitch_angles = np.asarray([fit.pitch_angle for fit in fits])
        return float(np.mean(pitch_angles))


@benchmark
def detect_spirals_in_image(
    image: NDArray[FloatType],
    *,
    unsharp_mask_settings: UnsharpMaskSettings = DEFAULT_UNSHARP_MASK,
    orientation_field_settings: GenerateOrientationFieldSettings = DEFAULT_ORIENTATION_FIELD_SETTINGS,
    similarity_matrix_settings: GenerateSimilarityMatrixSettings = DEFAULT_SIMILARITY_MATRIX_SETTINGS,
    generate_clusters_settings: GenerateClustersSettings = DEFAULT_CLUSTER_SETTINGS,
    merge_clusters_by_fit_settings: MergeClustersByFitSettings = DEFAULT_MERGE_CLUSTER_BY_FIT_SETTINGS,
    fit_error_settings: FitErrorSettings | None = None,
    preprocess: bool = False,
) -> ClusterSpiralResult | None:
    """Run the spiral arc finder algorithm on the given image.

    This function returns None if no suitable clusters can be found.

    Parameters
    ----------
    image : NDArray[FloatType]
        The input image as a NumPy array.
    unsharp_mask_settings : UnsharpMaskSettings, optional
        Settings for the unsharp mask, by default UnsharpMaskSettings().
    orientation_field_settings : GenerateOrientationFieldSettings, optional
        Settings for generating the orientation field, by default GenerateOrientationFieldSettings().
    similarity_matrix_settings : GenerateSimilarityMatrixSettings, optional
        Settings for generating the similarity matrix, by default GenerateSimilarityMatrixSettings().
    generate_clusters_settings : GenerateClustersSettings, optional
        Settings for generating clusters, by default GenerateClustersSettings().
    merge_clusters_by_fit_settings : MergeClustersByFitSettings, optional
        Settings for merging clusters based on fit criteria, by default MergeClustersByFitSettings().
    fit_error_settings : FitErrorSettings | None
        The maximum allowed fit error for a cluster and a normalisation scheme. If this is `None` then all
        clusters are allowed.
    preprocess : bool
        Set this flag to preprocess the image to ensure it is compatible.

    Returns
    -------
    ClusterSpiralResult | None
        The result of the spiral detection algorithm, or None if detection failed.

    Notes
    -----
    The input image must be normalized to the range [0, 1] before running this algorithm. Additionally it must also
    have dimensions divisible by 2^n where n is the number of orientation field levels (this is a setting you can adjust).

    """
    if preprocess:
        image = preprocess_image(image, num_orientation_field_levels=orientation_field_settings.num_orientation_field_levels)

    # Checks
    verify_data_is_normalized(image)
    verify_data_is_2d(image)

    log.info(r"[green]DIAGNOST[/green]: The image has dimensions %dx%d (width x height)", image.shape[1], image.shape[0])
    # Warn about large sizes
    max_size = max(image.shape)
    if max_size > MAX_SIZE_BEFORE_WARN:
        warning_msg = (
            "Note that the spiral arc finding algorithm scales really badly with image size. Please consider downscaling."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)

    # Unsharp phase
    unsharp_image = filters.unsharp_mask(
        image,
        radius=unsharp_mask_settings.radius,
        amount=unsharp_mask_settings.amount,
    )

    # Generate orientation fields
    log.info("[cyan]PROGRESS[/cyan]: Generating orientation field...")
    field = generate_orientation_fields(unsharp_image, orientation_field_settings)
    log.info("[cyan]PROGRESS[/cyan]: Done generating orientation field.")
    if field.count_nonzero() == 0:
        return None

    # Generate similarity matrix
    log.info("[cyan]PROGRESS[/cyan]: Generating similarity matrix...")
    matrix = generate_similarity_matrix(
        field,
        similarity_matrix_settings.similarity_cutoff,
    )
    log.info("[cyan]PROGRESS[/cyan]: Done generating similarity matrix.")
    if matrix.count_nonzero() == 0:
        return None

    # Merge clusters via HAC
    log.info("[cyan]PROGRESS[/cyan]: Generating clusters...")
    cluster_arrays: Sequence[NDArray[FloatType]] = generate_clusters(
        image,
        matrix.tocsr(),
        stop_threshold=generate_clusters_settings.stop_threshold,
        error_ratio_threshold=generate_clusters_settings.error_ratio_threshold,
        merge_check_minimum_cluster_size=generate_clusters_settings.merge_check_minimum_cluster_size,
        minimum_cluster_size=generate_clusters_settings.minimum_cluster_size,
        remove_central_cluster=generate_clusters_settings.remove_central_cluster,
    )
    log.info("[cyan]PROGRESS[/cyan]: Done generating clusters.")
    if len(cluster_arrays) == 0:
        return None

    # Do some final merges based on fit
    log.info("[cyan]PROGRESS[/cyan]: Merging clusters by fit...")
    cluster_list = merge_clusters_by_fit(
        cluster_arrays,
        merge_clusters_by_fit_settings.stop_threshold,
    )
    log.info("[cyan]PROGRESS[/cyan]: Done merging clusters by fit.")

    if fit_error_settings is not None:
        cluster_list = [
            current_cluster
            for current_cluster in cluster_list
            if fit_spiral_to_image(current_cluster).get_normalised_total_error(
                fit_error_settings.kind, pixel_to_distance=fit_error_settings.pixel_to_distance
            )
            <= fit_error_settings.max_error
        ]

    if len(cluster_list) == 0:
        return None

    detected_clusters = np.dstack(cluster_list)
    return ClusterSpiralResult(
        image=image,
        unsharp_image=unsharp_image,
        field=field,
        cluster_masks=detected_clusters,
        unsharp_mask_settings=unsharp_mask_settings,
        orientation_field_settings=orientation_field_settings,
        similarity_matrix_settings=similarity_matrix_settings,
        generate_cluster_settings=generate_clusters_settings,
        merge_clusters_by_fit_settings=merge_clusters_by_fit_settings,
    )
