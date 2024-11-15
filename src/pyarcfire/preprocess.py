"""Preprocess images to be suitable."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from skimage import transform

from .assert_utils import verify_data_is_2d

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy.typing import NDArray

    FloatType = TypeVar("FloatType", np.float32, np.float64)


log: logging.Logger = logging.getLogger(__name__)


def preprocess_image(image: NDArray[FloatType], *, num_orientation_field_levels: int) -> NDArray[FloatType]:
    """Process an image so that is compatible with the pyarcfire package.

    Parameters
    ----------
    image : NDArray[FloatType]
        The image to process.
    num_orientation_field_levels : int
        The number of orientation field levels.

    Returns
    -------
    processed_image : NDArray[FloatType]
        The processed image.

    """
    log.debug("[green]DIAGNOST[/green]: Standardizing image...")
    warning_msg = "This function is a WIP and may not produce satisfactory results. Image preprocessing is best done yourself."
    warnings.warn(warning_msg, UserWarning, stacklevel=1)
    processed_image = image.copy()
    verify_data_is_2d(processed_image)

    # Adjust size to be compatible with orientation field generation
    maximum_shrink_factor: int = 2**num_orientation_field_levels
    has_incompatible_size = image.shape[0] % maximum_shrink_factor != 0 or image.shape[1] % maximum_shrink_factor != 0
    if has_incompatible_size:
        processed_image = _resize_image(processed_image, maximum_shrink_factor)

    # Normalize
    return _normalize_image(processed_image)


# TODO(pavyamsiri): Add more normalization options
def _normalize_image(image: NDArray[FloatType]) -> NDArray[FloatType]:
    log.debug("[green]DIAGNOST[/green]: Normalizing...")
    # Remove nans
    normalized_image = np.nan_to_num(image, nan=0)

    min_value = np.min(image)
    max_value = np.max(image)
    # Array is the exactly the same value throughout
    if max_value == min_value:
        # Array is non-zero
        if max_value != 0:
            return np.divide(image, max_value)
        # Array is all zero
        return image
    return np.divide(np.subtract(normalized_image, min_value), max_value - min_value)


def _resize_image(image: NDArray[FloatType], divisor: int) -> NDArray[FloatType]:
    # TODO(pavyamsiri): Make this more sophisicated and add more resizing algorithms
    height: int = image.shape[0]
    width: int = image.shape[1]
    compatible_height = _closest_multiple(height, divisor)
    compatible_width = _closest_multiple(width, divisor)
    log.debug("[green]DIAGNOST[/green]: Resizing image to %dx%d...", compatible_height, compatible_width)
    return transform.resize(image, (compatible_height, compatible_width)).astype(image.dtype)


def _closest_multiple(num: int, divisor: int) -> int:
    quotient = num / divisor
    smaller_multiple = int(np.floor(quotient)) * divisor
    larger_multiple = int(np.ceil(quotient)) * divisor

    smaller_multiple_distance = num - smaller_multiple
    larger_multiple_distance = larger_multiple - num
    if smaller_multiple_distance <= larger_multiple_distance:
        return smaller_multiple
    return larger_multiple
