import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from pyarcfire import orientation


def valid_images(
    levels: int, min_multiple: int, max_multiple: int
) -> st.SearchStrategy[NDArray[np.float32]]:
    factor: int = 2**levels
    elements = st.floats(
        width=32, min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False
    )
    return st.integers(min_multiple, max_multiple).flatmap(
        lambda n: arrays(
            dtype=np.float32, shape=(n * factor, n * factor), elements=elements
        )
    )


@given(valid_images(levels=3, min_multiple=2, max_multiple=16))
def test_generation(arr: NDArray[np.float32]) -> None:
    field = orientation.generate_orientation_fields(
        arr, num_orientation_field_levels=3, neighbour_distance=5, kernel_radius=5
    )
    assert field.shape[0] == arr.shape[0]
    assert field.shape[1] == arr.shape[1]
