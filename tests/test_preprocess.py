"""Tests for image preprocessers."""

import numpy as np
import optype.numpy as onp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pyarcfire.preprocess import ImageUnsharpMaskBooster


@given(
    arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(2, 256), st.integers(2, 256)),
        elements=st.floats(0, 1),
    )
)
def test_unsharp_mask_transpose_invariance_property(image: onp.Array2D[np.float32]) -> None:
    """Test that an unsharp mask is invariant over transpose.

    Parameters
    ----------
    image : Array2D[f32]
        The image to test.

    """
    booster = ImageUnsharpMaskBooster()
    processed_image = booster.boost(image)
    processed_transpose_image = booster.boost(image.T)
    assert np.count_nonzero(processed_image) == np.count_nonzero(processed_transpose_image)
