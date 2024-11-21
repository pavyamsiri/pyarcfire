"""The spiral arc finder class."""

from __future__ import annotations

import logging
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np
import scipy.io
from numpy.lib.npyio import NpzFile
from typing_extensions import assert_never

from .arc import Chirality, FitErrorKind, LogSpiralFitResult, fit_spiral_to_image
from .arc.utils import get_polar_coordinates
from .assert_utils import verify_array_dtype, verify_data_is_2d, verify_data_is_3d
from .cluster import generate_clusters
from .merge_fit import merge_clusters_by_fit
from .orientation import OrientationField, generate_orientation_fields
from .preprocess import (
    ImageContrastBooster,
    ImageDivisibleResizer,
    ImageIdentityNormalizer,
    ImageLinearNormalizer,
    ImageNormalizer,
    ImageResizer,
    ImageUnsharpMaskBooster,
)
from .similarity import generate_similarity_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype as op
    from numpy._typing import _ArrayLikeFloat_co  # pyright:ignore[reportPrivateUsage]

    from ._typing import AnyReal


StrPath: TypeAlias = str | PathLike[str]

_SCT = TypeVar("_SCT", bound=np.generic)
_Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[_SCT]]
_Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_SCT]]
_Array1D_f64: TypeAlias = _Array1D[np.float64]
_Array2D_f64: TypeAlias = _Array2D[np.float64]
_Array2D_u32: TypeAlias = _Array2D[np.uint32]

_CalculateRadiiFn: TypeAlias = Callable[[_Array1D_f64], _Array1D_f64]

log: logging.Logger = logging.getLogger(__name__)


class SpiralFinderResult:
    """The result of the spiral finding algorithm."""

    def __init__(
        self, mask: _Array2D_u32, *, original_image: _Array2D_f64, processed_image: _Array2D_f64, field: OrientationField
    ) -> None:
        """Initialize the result.

        Parameters
        ----------
        mask : Array2D[u32]
            The cluster mask where non-zero values indicate the cluster id the pixel belongs to.
        original_image : Array2D[f64]
            The original image before preprocessing.
        processed_image : Array2D[f64]
            The image after preprocessing.
        field : OrientationField
            The orientation field of the processed image.

        """
        self._mask: _Array2D_u32 = mask
        self._original_image: _Array2D_f64 = original_image
        self._processed_image: _Array2D_f64 = processed_image
        self._field: OrientationField = field

        # Useful values
        self._num_clusters: int = int(np.max(self._mask))
        self._sizes: tuple[int, ...] = tuple(
            int(np.count_nonzero(self._mask == cluster_index)) for cluster_index in range(1, self._num_clusters + 1)
        )

        # Fits
        self._fits: Sequence[LogSpiralFitResult] = [
            fit_spiral_to_image(np.where(self._mask == cluster_index, self._processed_image, 0))
            for cluster_index in range(1, self._num_clusters + 1)
        ]

    @property
    def mask(self) -> _Array2D_u32:
        """Array2D[u32]: The cluster mask.

        Non-zero integers indicate the presence of a cluster and its value determines the cluster id.
        """
        return self._mask

    @property
    def original_image(self) -> _Array2D_f64:
        """Array2D[f64]: The original image before preprocessing."""
        return self._original_image

    @property
    def original_image_height(self) -> int:
        """int: The original image's pixel height."""
        return self._original_image.shape[0]

    @property
    def original_image_width(self) -> int:
        """int: The original image's pixel width."""
        return self._original_image.shape[1]

    @property
    def processed_image(self) -> _Array2D_f64:
        """Array2D[f64]: The image after preprocessing."""
        return self._processed_image

    @property
    def processed_image_height(self) -> int:
        """int: The processed image's pixel height."""
        return self._processed_image.shape[0]

    @property
    def processed_image_width(self) -> int:
        """int: The processed image's pixel width."""
        return self._processed_image.shape[1]

    @property
    def orientation_field(self) -> OrientationField:
        """OrientationField: The orientation field."""
        return self._field

    @property
    def num_clusters(self) -> int:
        """int: The number of clusters found."""
        return self._num_clusters

    @property
    def sizes(self) -> tuple[int, ...]:
        """tuple[int, ...]: The sizes of the clusters found."""
        return self._sizes

    def __str__(self) -> str:
        """Return the string representation.

        Returns
        -------
        str
            The string representation.

        """
        return f"{type(self).__qualname__}(num_clusters={self.num_clusters})"

    def get_dominant_chirality(self) -> Chirality:
        """Determine the dominant chirality by arc length weighted vote.

        Returns
        -------
        dominant_chirality : Chirality
            The dominant chirality.

        """
        arc_lengths = np.asarray([fit.arc_length for fit in self._fits])
        chiralities = np.asarray([fit.chirality_sign for fit in self._fits])
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
        if self._num_clusters == 0:
            log.warning("There are no clusters so the overall pitch angle is undefined!")
            return float("nan")
        dominant_chirality = self.get_dominant_chirality()
        fits = [fit for fit in self._fits if fit.chirality == dominant_chirality]
        pitch_angles = np.asarray([fit.pitch_angle for fit in fits])
        return float(np.mean(pitch_angles))

    def get_fit(self, cluster_index: op.CanIndex) -> LogSpiralFitResult:
        """Return the log spiral fit to a cluster.

        Parameters
        ----------
        cluster_index : int
            The index of the cluster to get the index of.

        Returns
        -------
        fit : LogSpiralFitResult
            The log spiral fit.

        """
        if cluster_index not in range(self.num_clusters):
            msg = f"Cluster index {cluster_index} is not in the range [0, {self.num_clusters})!"
            raise IndexError(msg)
        return self._fits[int(cluster_index)]

    def calculate_fit_error_to_cluster(
        self,
        calculate_radii: _CalculateRadiiFn,
        cluster_index: op.CanIndex,
        *,
        pixel_to_distance: AnyReal,
        fit_error_kind: FitErrorKind = FitErrorKind.NONORM,
    ) -> float:
        """Calculate the residuals of the given function with respect to a cluster.

        Parameters
        ----------
        calculate_radii : Callable[[Array1D[f64]], Array1D[f64]]
            A function that takes in an array of angles and returns radii.
        cluster_index : int
            The index of the cluster.
        pixel_to_distance : float
            Conversion factor from pixel units to physical distance units.
        fit_error_kind : FitErrorKind
            The kind of normalisation scheme to apply to the fit error before returning it.

        Returns
        -------
        error : float
            The error.

        """
        cluster_index = int(cluster_index)
        if cluster_index not in range(self.num_clusters):
            msg = f"Cluster index {cluster_index} is not in the range [0, {self.num_clusters})!"
            raise IndexError(msg)

        current_array, _ = np.where(self._mask == (cluster_index + 1), self._processed_image, 0)
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

    @staticmethod
    def load(path: StrPath) -> SpiralFinderResult | None:
        """Load the result from a file.

        Parameters
        ----------
        path : str
            The path to load from.

        Returns
        -------
        result : SpiralFinderResult | None
            The loaded result if the extension is supported.

        Notes
        -----
        The supported formats are currently:
        - npz
            - numpy archive file.
        - mat
            - MatLab mat file.

        """
        extension = Path(path).suffix.lstrip(".")
        if extension == "mat":
            file = scipy.io.loadmat(path)
            mask = verify_data_is_2d(verify_array_dtype(file["mask"], np.uint32))
            original_image = verify_data_is_2d(verify_array_dtype(file["original_image"], np.float64))
            processed_image = verify_data_is_2d(verify_array_dtype(file["processed_image"], np.float64))
            field = verify_data_is_3d(verify_array_dtype(file["field"], np.float64))
        elif extension == "npz":
            file = np.load(path)
            if not isinstance(file, NpzFile):
                msg = "Expected the file to be a npz archive!"
                raise TypeError(msg)

            mask = verify_data_is_2d(verify_array_dtype(file["mask"], np.uint32))
            original_image = verify_data_is_2d(verify_array_dtype(file["original_image"], np.float64))
            processed_image = verify_data_is_2d(verify_array_dtype(file["processed_image"], np.float64))
            field = verify_data_is_3d(verify_array_dtype(file["field"], np.float64))
        else:
            return None

        return SpiralFinderResult(
            mask=mask,
            original_image=original_image,
            processed_image=processed_image,
            field=OrientationField(field),
        )

    def dump(self, path: StrPath) -> None:
        """Dump the result into one of the supported formats.

        Parameters
        ----------
        path : str
            The path to write to.

        Notes
        -----
        The supported formats are currently:
        - npz
            - numpy archive file.
        - mat
            - MatLab mat file.

        """
        extension = Path(path).suffix.lstrip(".")
        if extension == "npz":
            np.savez_compressed(
                path,
                mask=self._mask,
                original_image=self._original_image,
                processed_image=self._processed_image,
                field=self._field.field,
            )
        elif extension == "mat":
            scipy.io.savemat(
                path,
                {
                    "mask": self._mask,
                    "original_image": self._original_image,
                    "processed_image": self._processed_image,
                    "field": self._field.field,
                },
            )
        else:
            log.warning(
                "[yellow]FILESYST[/yellow]: Can not dump due to unknown extension [yellow]%s[/yellow]",
                extension,
            )


class SpiralFinder:
    """Class that contains the parameters for the SpArcFiRe algorithm.

    Create an instance of this class and then call `extract` to run the algorithm.
    """

    def __init__(self) -> None:
        """Initialize the finder."""
        # Orientation field parameters
        self._field_neighbour_distance: int = 5
        self._field_kernel_radius: int = 5
        self._field_num_orientation_field_levels: int = 3
        # Similarity matrix parameters
        self._similarity_cutoff: float = 0.15
        # Clustering parameters
        self._error_ratio_threshold: float = 2.5
        self._merge_check_minimum_cluster_size: int = 25
        self._minimum_cluster_size: int = 150
        self._remove_central_cluster: bool = True
        # Merge clusters by fit
        self._merge_fit_stop_threshold: float = 2.5

        # Preprocessors
        self._normalizer: ImageNormalizer = ImageLinearNormalizer()
        self._resizer: ImageResizer | None = ImageDivisibleResizer(2**self._field_num_orientation_field_levels)
        self._booster: ImageContrastBooster | None = ImageUnsharpMaskBooster()

    @property
    def normalizer(self) -> ImageNormalizer:
        """ImageNormalizer: The image normalizer used."""
        return self._normalizer

    @property
    def resizer(self) -> ImageResizer | None:
        """ImageResizer | None: The resizer used if it is set otherwise `None`."""
        return self._resizer

    @property
    def booster(self) -> ImageContrastBooster | None:
        """ImageContrastBooster | None: The contrast booster used if it is set otherwise `None`."""
        return self._booster

    def extract(self, image: _ArrayLikeFloat_co) -> SpiralFinderResult:
        """Extract spiral arm segments from an image.

        Parameters
        ----------
        image : ArrayLike[float]
            A 2D array representing a black and white image centered on a galaxy or object with spiral structure.

        Returns
        -------
        result : SpiralFinderResult
            The result of the extraction algorithm.

        """
        # Step -1: Convert image to numpy array
        image_array = np.asarray(image)

        # Verify shape
        image_array = verify_data_is_2d(image_array)

        # Step 0: Preprocess the image
        processed_image = self._normalizer.normalize(image_array)
        if self._resizer is not None:
            processed_image = self._resizer.resize(processed_image)
        if self._booster is not None:
            processed_image = self._booster.boost(processed_image)

        # Step 1: Generate orientation field
        field = generate_orientation_fields(
            processed_image,
            neighbour_distance=self._field_neighbour_distance,
            kernel_radius=self._field_kernel_radius,
            num_orientation_field_levels=self._field_num_orientation_field_levels,
        )

        # Step 2: Construct similarity matrix
        sim_matrix = generate_similarity_matrix(
            field,
            self._similarity_cutoff,
        )

        # Step 3: Perform clustering
        cluster_list: Sequence[_Array2D_f64] = generate_clusters(
            processed_image,
            sim_matrix.tocsr(),
            stop_threshold=self._similarity_cutoff,
            error_ratio_threshold=self._error_ratio_threshold,
            merge_check_minimum_cluster_size=self._merge_check_minimum_cluster_size,
            minimum_cluster_size=self._minimum_cluster_size,
            remove_central_cluster=self._remove_central_cluster,
        )

        # Step 4: Merge clusters by spiral fits
        cluster_list = merge_clusters_by_fit(
            cluster_list,
            self._merge_fit_stop_threshold,
        )

        # Step 5: Combine clusters into 2D array labelled by cluster index
        cluster_mask = np.zeros_like(processed_image, dtype=np.uint32)

        for cluster_index, current_mask in enumerate(cluster_list):
            cluster_mask[current_mask != 0] = cluster_index + 1

        return SpiralFinderResult(
            mask=cluster_mask,
            original_image=image_array,
            processed_image=processed_image,
            field=field,
        )

    # Configuration functions
    def with_normalizer(self, normalizer: ImageNormalizer | None) -> SpiralFinder:
        """Set the current normalizer to the given normalizer.

        Parameters
        ----------
        normalizer : ImageNormalizer | None
            The new normalizer to use. If `None` is given then the normalizer will be the identity normalizer.

        Returns
        -------
        new_finder : SpiralFinder
            The finder with the new normalizer.

        """
        self._normalizer = normalizer if normalizer is not None else ImageIdentityNormalizer()
        return self

    def with_resizer(self, resizer: ImageResizer | None) -> SpiralFinder:
        """Set the current resizer to the given resizer.

        Parameters
        ----------
        resizer : ImageResizer | None
            The new resizer to use. If `None` is given then the image will not be resized.

        Returns
        -------
        new_finder : SpiralFinder
            The finder with the new resizer.

        """
        self._resizer = resizer
        return self

    def with_booster(self, booster: ImageContrastBooster | None) -> SpiralFinder:
        """Set the current booster to the given booster.

        Parameters
        ----------
        booster : ImageContrastBooster | None
            The new booster to use. If `None` is given then the image will not be boosted.

        Returns
        -------
        new_finder : SpiralFinder
            The finder with the new booster.

        """
        self._booster = booster
        return self

    def with_orientation_field_settings(
        self,
        *,
        neighbour_distance: op.CanInt | None = None,
        kernel_radius: op.CanInt | None = None,
        num_levels: op.CanInt | None = None,
    ) -> SpiralFinder:
        """Configure the settings for the orientation field creation step.

        Parameters
        ----------
        neighbour_distance : int | None
            The distance in pixels between a cell and its neighbour when denoising.
        kernel_radius : int | None
            The radius of the orientation filter kernel in pixels.
        num_levels : int | None
            The number of orientation field levels to create and then join.

        Returns
        -------
        new_finder : SpiralFinder
            The newly configured finder.

        Notes
        -----
        If `None` is given for any parameter then the current value is kept.

        """
        if neighbour_distance is not None:
            self._field_neighbour_distance = int(neighbour_distance)
        if kernel_radius is not None:
            self._field_kernel_radius = int(kernel_radius)
        if num_levels is not None:
            self._field_num_orientation_field_levels = int(num_levels)
            # Update resizer
            if self._resizer is not None:
                self._resizer.update(self._field_num_orientation_field_levels)
        return self

    def with_similarity_matrix_settings(self, *, cutoff: op.CanFloat | None = None) -> SpiralFinder:
        """Configure the settings for the similarity matrix creation step.

        Parameters
        ----------
        cutoff : float | None
            The minimum amount of similarity allowed before it is clipped to zero.


        Returns
        -------
        new_finder : SpiralFinder
            The newly configured finder.

        Notes
        -----
        If `None` is given for any parameter then the current value is kept.

        """
        if cutoff is not None:
            self._similarity_cutoff = float(cutoff)

        return self

    def with_clustering_settings(
        self,
        *,
        error_ratio_threshold: op.CanFloat | None = None,
        merge_check_minimum_cluster_size: op.CanInt | None = None,
        minimum_cluster_size: op.CanInt | None = None,
        remove_central_cluster: op.CanBool | None = None,
    ) -> SpiralFinder:
        """Configure the settings for the clustering step.

        Parameters
        ----------
        error_ratio_threshold : float | None
            The maximum error ratio allowed for a merge between two clusters
            to be permitted. This error ratio is the ratio of an arc fit's error to
            the merged cluster relative to the error of two arc fits to the clusters
            individually.
        merge_check_minimum_cluster_size : int | None
            The maximum size of each cluster before their merges become checked and
            potentially stopped.
        minimum_cluster_size : int | None
            The minimum cluster size allowed after all merges are complete.
            Clusters with sizes below this value are discarded.
        remove_central_cluster : bool | None
            Set this flag to remove clusters that touch the centre.


        Returns
        -------
        new_finder : SpiralFinder
            The newly configured finder.

        Notes
        -----
        If `None` is given for any parameter then the current value is kept.

        """
        if error_ratio_threshold is not None:
            self._error_ratio_threshold = float(error_ratio_threshold)
        if merge_check_minimum_cluster_size is not None:
            self._merge_check_minimum_cluster_size = int(merge_check_minimum_cluster_size)
        if minimum_cluster_size is not None:
            self._minimum_cluster_size = int(minimum_cluster_size)
        if remove_central_cluster is not None:
            self._remove_central_cluster = bool(remove_central_cluster)

        return self

    def with_merge_fit_settings(self, stop_threshold: op.CanFloat | None = None) -> SpiralFinder:
        """Configure the settings for the cluster merging via fit step.

        Parameters
        ----------
        stop_threshold : float | None
            The maximum merge error ratio before stopping merges.


        Returns
        -------
        new_finder : SpiralFinder
            The newly configured finder.

        Notes
        -----
        If `None` is given for any parameter then the current value is kept.

        """
        if stop_threshold is not None:
            self._merge_fit_stop_threshold = float(stop_threshold)
        return self
