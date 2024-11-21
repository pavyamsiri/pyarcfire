"""The spiral arc finder class."""

from __future__ import annotations

import logging
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, assert_never, cast

import numpy as np
import scipy.io
from skimage import filters, transform

from .arc import Chirality, FitErrorKind, LogSpiralFitResult, fit_spiral_to_image
from .arc.utils import get_polar_coordinates
from .assert_utils import verify_data_is_2d
from .cluster import generate_clusters
from .merge_fit import merge_clusters_by_fit
from .orientation import OrientationField, generate_orientation_fields
from .similarity import generate_similarity_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype as op
    from numpy._typing import _ArrayLikeFloat_co  # pyright:ignore[reportPrivateUsage]

    from ._typing import AnyReal


StrPath: TypeAlias = str | PathLike[str]

_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any])
_Array1D = np.ndarray[tuple[int], np.dtype[_SCT]]
_Array2D = np.ndarray[tuple[int, int], np.dtype[_SCT]]
_Array1D_f64 = _Array1D[np.float64]
_Array2D_f64 = _Array2D[np.float64]
_Array2D_u32 = _Array2D[np.uint32]

_CalculateRadiiFn: TypeAlias = Callable[[_Array1D_f64], _Array1D_f64]

log: logging.Logger = logging.getLogger(__name__)

# Preprocessors:
# - Normalizer
# - Resizer
# - Constrast booster


class ImagePreprocessor(Protocol):
    def preprocess(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]: ...


class ImageNormalizer(Protocol):
    def normalize(self, image: _Array2D[_SCT]) -> _Array2D_f64: ...


class ImageResizer(Protocol):
    def resize(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]: ...


class ImageContrastBooster(Protocol):
    def boost(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]: ...


class ImageLinearNormalizer:
    def __init__(self) -> None:
        pass

    def normalize(self, image: _Array2D[_SCT]) -> _Array2D_f64:
        float_image = image.astype(np.float64)
        # Map non-finite values to finite values
        float_image = np.nan_to_num(float_image, nan=0, posinf=1, neginf=0)

        min_value = np.min(float_image)
        max_value = np.max(float_image)
        # Array is the exactly the same value throughout
        if max_value == min_value:
            # Array should be all one
            if max_value != 0:
                return np.ones_like(float_image)
            # Array is all zero
            return np.zeros_like(float_image)
        return (float_image - min_value) / (max_value - min_value)


class ImageDivisibleResizer:
    def __init__(self, divisor: op.CanInt) -> None:
        self._divisor: int = int(divisor)

    def resize(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        height: int = image.shape[0]
        width: int = image.shape[1]
        compatible_height = self._closest_multiple(height, self._divisor)
        compatible_width = self._closest_multiple(width, self._divisor)
        return cast(_Array2D[_SCT_f], transform.resize(image, (compatible_height, compatible_width)).astype(image.dtype))  # pyright:ignore[reportUnknownMemberType]

    @staticmethod
    def _closest_multiple(num: int, divisor: int) -> int:
        quotient = num / divisor
        smaller_multiple = int(np.floor(quotient)) * divisor
        larger_multiple = int(np.ceil(quotient)) * divisor

        smaller_multiple_distance = num - smaller_multiple
        larger_multiple_distance = larger_multiple - num
        if smaller_multiple_distance <= larger_multiple_distance:
            return smaller_multiple
        return larger_multiple


class ImageUnsharpMaskBooster:
    def __init__(self, radius: op.CanFloat, amount: op.CanFloat) -> None:
        self._radius: float = float(radius)
        self._amount: float = float(amount)

    def boost(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        return cast(
            _Array2D[_SCT_f],
            filters.unsharp_mask(  # pyright:ignore[reportUnknownMemberType]
                image,
                radius=self._radius,
                amount=self._amount,
            ),
        )


class SpiralFinderResult:
    def __init__(
        self, mask: _Array2D_u32, *, original_image: _Array2D_f64, processed_image: _Array2D_f64, field: OrientationField
    ) -> None:
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
        return self._mask

    @property
    def original_image(self) -> _Array2D_f64:
        return self._original_image

    @property
    def original_image_height(self) -> int:
        return self._original_image.shape[0]

    @property
    def original_image_width(self) -> int:
        return self._original_image.shape[1]

    @property
    def processed_image_height(self) -> int:
        return self._processed_image.shape[0]

    @property
    def processed_image_width(self) -> int:
        return self._processed_image.shape[1]

    @property
    def processed_image(self) -> _Array2D_f64:
        return self._processed_image

    @property
    def orientation_field(self) -> OrientationField:
        return self._field

    @property
    def num_clusters(self) -> int:
        return self._num_clusters

    @property
    def sizes(self) -> tuple[int, ...]:
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
        dominant_chirality = self.get_dominant_chirality()
        fits = [fit for fit in self._fits if fit.chirality == dominant_chirality]
        pitch_angles = np.asarray([fit.pitch_angle for fit in fits])
        return float(np.mean(pitch_angles))

    def get_fit(self, cluster_index: op.CanIndex) -> LogSpiralFitResult:
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

    def dump(self, path: StrPath) -> None:
        """Dump the result into one of the supported formats.

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
            return
        log.info(
            "[yellow]FILESYST[/yellow]: Dumped masks to [yellow]%s[/yellow]",
            extension,
        )


class SpiralFinder:
    """Class that contains the parameters for the SpArcFiRe algorithm.

    Create this class and then call `extract` to run the algorithm.
    """

    def __init__(self) -> None:
        # Orientation field parameters
        self._field_neighbour_distance: int = 5
        self._field_kernel_radius: int = 5
        self._field_num_orientation_field_levels: int = 3
        self._size_divisor: int = 2**self._field_num_orientation_field_levels
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
        self._resizer: ImageResizer = ImageDivisibleResizer(self._size_divisor)

        self._unsharp_radius: float = 25
        self._unsharp_amount: float = 6
        self._booster: ImageContrastBooster = ImageUnsharpMaskBooster(25, 6)

    def extract(self, image: _ArrayLikeFloat_co) -> SpiralFinderResult:
        # Step -1: Convert image to numpy array
        image_array = np.asarray(image)

        # Verify shape
        image_array = verify_data_is_2d(image_array)

        # Step 0: Preprocess the image
        processed_image = self._normalizer.normalize(image_array)
        processed_image = self._resizer.resize(processed_image)
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
