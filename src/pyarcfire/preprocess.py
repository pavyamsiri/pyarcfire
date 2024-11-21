"""Preprocess images to be suitable."""

from __future__ import annotations

__all__ = [
    # Normalizers
    "ImageNormalizer",
    "ImageLinearNormalizer",
    "ImageIdentityNormalizer",
    # Resizers
    "ImageResizer",
    "ImageDivisibleResizer",
    # Boosters
    "ImageContrastBooster",
    "ImageUnsharpMaskBooster",
]

import logging
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, cast

import numpy as np
from skimage import filters, transform

from .assert_utils import verify_data_is_normalized

if TYPE_CHECKING:
    import optype as op

    from ._typing import AnyReal

_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any])
_Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_SCT]]


log: logging.Logger = logging.getLogger(__name__)


class ImageNormalizer(Protocol):
    """The interface for an image data normalizer."""

    def normalize(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        """Normalize the input image into the range [0, 1].

        Parameters
        ----------
        image : Array2D[F]
            The image to preprocess.

        Returns
        -------
        normalized_image : Array2D[F]
            The normalized image.

        """
        ...


class ImageResizer(Protocol):
    """The interface for an image resizer."""

    def resize(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        """Resize an image.

        Parameters
        ----------
        image : Array2D[F]
            The image to resize.

        Returns
        -------
        resized_image  : Array2D[F]
            The resized image.

        """
        ...

    def update(self, num_levels: int) -> None:
        """Update the parameters of the resizer given the number of orientation field levels.

        Parameters
        ----------
        num_levels : int
            The number of orientation field levels.

        """
        ...


class ImageContrastBooster(Protocol):
    """The interface for an image contrast booster."""

    def boost(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        """Contrast boost an image.

        Parameters
        ----------
        image : Array2D[F]
            The image to contrast boost.

        Returns
        -------
        boosted_image : Array2D[F]
            The contrast boosted image.

        """
        ...


class ImageLinearNormalizer:
    """Normalizes an image using a simple linear normalization scheme."""

    def __init__(self, *, vmin: AnyReal | None = None, vmax: AnyReal | None = None) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        vmin : float | None
            The minimum value in the linear scale if given.
        vmax : float | None
            The maximum value in the linear scale if given.

        Notes
        -----
        If `None` is given for either `vmin` and `vmax` then they will be dynmically determined by finding
        the minimum and maximum value of the given images.

        """
        self._vmin: float | None = float(vmin) if vmin is not None else None
        self._vmax: float | None = float(vmax) if vmax is not None else None

    def normalize(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        """Normalize the input image into the range [0, 1].

        Parameters
        ----------
        image : Array2D[F]
            The image to preprocess.

        Returns
        -------
        normalized_image : Array2D[F]
            The normalized image.

        """
        # Map non-finite values to finite values
        finite_image = np.nan_to_num(image, nan=0, posinf=1, neginf=0)

        min_value = np.min(finite_image) if self._vmin is None else self._vmin
        max_value = np.max(finite_image) if self._vmax is None else self._vmax

        if max_value < min_value:
            msg = f"Given vmin {min_value} that is greater than vmax {max_value}!"
            raise ValueError(msg)

        # Array is the exactly the same value throughout
        if max_value == min_value:
            # Array should be all one
            if max_value != 0:
                return np.ones_like(finite_image)
            # Array is all zero
            return np.zeros_like(finite_image)
        return ((finite_image - min_value) / (max_value - min_value)).astype(image.dtype)


class ImageIdentityNormalizer:
    """Validates an image is normalized to the range [0, 1].

    This does not do any normalization itself.
    """

    def __init__(self) -> None:
        """Initialize the normalizer."""

    def normalize(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        """Ensure that the input image is in the range [0, 1].

        Parameters
        ----------
        image : Array2D[F]
            The image to validate.

        Returns
        -------
        normalized_image : Array2D[F]
            The normalized image.

        """
        verify_data_is_normalized(image)
        return image


class ImageDivisibleResizer:
    """An image resizer that resizes images so that they are divisible by a given number."""

    def __init__(self, divisor: op.CanInt) -> None:
        """Initialize the resizer.

        Parameters
        ----------
        divisor : int
            The number that that image dimensions must be divisible by.

        """
        self._divisor: int = int(divisor)

    def update(self, num_levels: int) -> None:
        """Update the parameters of the resizer given the number of orientation field levels.

        Parameters
        ----------
        num_levels : int
            The number of orientation field levels.

        """
        self._divisor = 2**num_levels

    def resize(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        """Resize an image.

        Parameters
        ----------
        image : Array2D[F]
            The image to resize.

        Returns
        -------
        resized_image  : Array2D[F]
            The resized image.

        """
        height: int = image.shape[0]
        width: int = image.shape[1]
        compatible_height = self._closest_multiple(height, self._divisor)
        compatible_width = self._closest_multiple(width, self._divisor)
        return cast(_Array2D[_SCT_f], transform.resize(image, (compatible_height, compatible_width)).astype(image.dtype))

    @staticmethod
    def _closest_multiple(num: int, divisor: int) -> int:
        """Find the closest multiple of `divisor` to `num`.

        Parameters
        ----------
        num : int
            The number to get the closest multiple of `divisor` of.
        divisor : int
            The base factor of the multiple.

        Returns
        -------
        closest_multiple : int
            The closest multiple of `divisor` to `num`.

        """
        quotient = num / divisor
        smaller_multiple = int(np.floor(quotient)) * divisor
        larger_multiple = int(np.ceil(quotient)) * divisor

        smaller_multiple_distance = num - smaller_multiple
        larger_multiple_distance = larger_multiple - num
        if smaller_multiple_distance <= larger_multiple_distance:
            return smaller_multiple
        return larger_multiple


class ImageUnsharpMaskBooster:
    """Boost the contrast of a normalized image using an unsharp mask."""

    def __init__(self, radius: op.CanFloat = 25, amount: op.CanFloat = 6) -> None:
        """Initialize the unsharp mask booster.

        Parameters
        ----------
        radius : float
            The radius of the unsharp mask. Must be non-negative.
        amount : float
            The amplification factor of the unsharp mask. Must be non-negative.

        """
        self._radius: float = float(radius)
        self._amount: float = float(amount)

    def boost(self, image: _Array2D[_SCT_f]) -> _Array2D[_SCT_f]:
        """Contrast boost an image.

        Parameters
        ----------
        image : Array2D[F]
            The image to contrast boost.

        Returns
        -------
        boosted_image : Array2D[F]
            The contrast boosted image.

        """
        return cast(
            _Array2D[_SCT_f],
            filters.unsharp_mask(
                image,
                radius=self._radius,
                amount=self._amount,
            ),
        )

    def __str__(self) -> str:
        """Return the string representation.

        Returns
        -------
        str
            The string representation.

        """
        return f"{type(self).__qualname__}(radius={self._radius:.2f}, amount={self._amount:.2f})"
