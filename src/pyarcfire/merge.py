"""Functions relating to calculating merging errors."""

from __future__ import annotations

import logging
from typing import Any, TypeAlias, TypeVar

import numpy as np

from .arc import fit_spiral_to_image

_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any])
_Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_SCT]]

log: logging.Logger = logging.getLogger(__name__)


def calculate_arc_merge_error(
    first_cluster_array: _Array2D[_SCT_f],
    second_cluster_array: _Array2D[_SCT_f],
) -> float:
    """Calculate the arc merge error ratio for two clusters.

    This is a measure of how well the merged cluster of the two clusters given fit to a log spiral
    compared to fitting the two clusters separately.

    Parameters
    ----------
    first_cluster_array : NDArray[FloatType]
        The first cluster in the form of an array.
    second_cluster_array : NDArray[FloatType]
        The second cluster in the form of an array.

    Returns
    -------
    merge_error_ratio : float
        The arc merge error ratio.

    """
    first_sum = first_cluster_array.sum()
    second_sum = second_cluster_array.sum()
    # Verify we don't have division by zero or weird negative weights
    if first_sum <= 0 or second_sum <= 0:
        msg = "The cluster arrays must sum to a positive value."
        raise ValueError(msg)
    total_sum = first_sum + second_sum
    # Adjust weights
    first_reweighted_array = first_cluster_array * total_sum / first_sum
    second_reweighted_array = second_cluster_array * total_sum / second_sum

    # Fit spirals to each cluster individually
    first_fit = fit_spiral_to_image(first_reweighted_array)
    second_fit = fit_spiral_to_image(second_reweighted_array)

    # Fit a spiral to both clusters at the same time
    combined_cluster_array = first_reweighted_array + second_reweighted_array
    first_merged_fit = fit_spiral_to_image(
        combined_cluster_array,
        initial_growth_factor=first_fit.growth_factor,
    )
    second_merged_fit = fit_spiral_to_image(
        combined_cluster_array,
        initial_growth_factor=second_fit.growth_factor,
    )
    first_fit_is_better = first_merged_fit.total_error <= second_merged_fit.total_error
    merged_fit = first_merged_fit if first_fit_is_better else second_merged_fit
    first_cluster_indices = (first_reweighted_array > 0)[combined_cluster_array > 0]
    # Get the sse of the merged spiral for each individual cluster
    first_cluster_error = np.square(merged_fit.errors[first_cluster_indices]).sum()
    second_cluster_error = np.square(merged_fit.errors[~first_cluster_indices]).sum()

    # Calculate the arc merge error ratio for each cluster
    ratios = (
        (first_cluster_error / first_fit.total_error),
        (second_cluster_error / second_fit.total_error),
    )
    # Return the worst error ratio
    return max(ratios)
