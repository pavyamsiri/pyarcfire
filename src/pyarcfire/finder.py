"""The spiral arc finder class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import numpy as np
from skimage import filters, transform

from .assert_utils import verify_data_is_2d
from .cluster import GenerateClustersSettings, generate_clusters
from .merge_fit import MergeClustersByFitSettings, merge_clusters_by_fit
from .orientation import GenerateOrientationFieldSettings, generate_orientation_fields
from .similarity import GenerateSimilarityMatrixSettings, generate_similarity_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype as op
    from numpy._typing import _ArrayLikeFloat_co  # pyright:ignore[reportPrivateUsage]

_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any])
_Array2D = np.ndarray[tuple[int, int], np.dtype[_SCT]]
_Array2D_f64 = _Array2D[np.float64]

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


class SpiralFinder:
    """Class that contains the parameters for the SpArcFiRe algorithm.

    Create this class and then call `extract` to run the algorithm.
    """

    def __init__(self) -> None:
        self._normalizer: ImageNormalizer = ImageLinearNormalizer()
        self._resizer: ImageResizer = ImageDivisibleResizer(2**3)
        self._booster: ImageContrastBooster = ImageUnsharpMaskBooster(25, 6)

    def extract(self, image: _ArrayLikeFloat_co) -> None:
        # Step -1: Convert image to numpy array
        processed_image = np.asarray(image)

        # Verify shape
        processed_image = verify_data_is_2d(processed_image)

        # Step 0: Preprocess the image
        processed_image = self._normalizer.normalize(processed_image)
        processed_image = self._resizer.resize(processed_image)
        processed_image = self._booster.boost(processed_image)

        # Step 1: Generate orientation field
        orient_field = generate_orientation_fields(processed_image, GenerateOrientationFieldSettings())

        # Step 2: Construct similarity matrix
        sim_matrix = generate_similarity_matrix(
            orient_field,
            GenerateSimilarityMatrixSettings().similarity_cutoff,
        )

        generate_clusters_settings = GenerateClustersSettings()
        cluster_list: Sequence[_Array2D_f64] = generate_clusters(
            processed_image,
            sim_matrix.tocsr(),
            stop_threshold=generate_clusters_settings.stop_threshold,
            error_ratio_threshold=generate_clusters_settings.error_ratio_threshold,
            merge_check_minimum_cluster_size=generate_clusters_settings.merge_check_minimum_cluster_size,
            minimum_cluster_size=generate_clusters_settings.minimum_cluster_size,
            remove_central_cluster=generate_clusters_settings.remove_central_cluster,
        )

        merge_clusters_by_fit_settings = MergeClustersByFitSettings()
        cluster_list = merge_clusters_by_fit(
            cluster_list,
            merge_clusters_by_fit_settings.stop_threshold,
        )
        detected_clusters = np.dstack(cluster_list)

        # Create a new 2D array with the same height and width, initialized to 0
        result = np.zeros((detected_clusters.shape[0], detected_clusters.shape[1]), dtype=np.int32)

        # Iterate over slices in the 3rd dimension
        for index in range(detected_clusters.shape[2]):
            # Update the result: set indices where the current slice is non-zero
            result[detected_clusters[:, :, index] != 0] = index + 1  # Use 1-based indexing if needed
