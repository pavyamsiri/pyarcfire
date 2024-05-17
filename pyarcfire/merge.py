# Standard libraries
import logging

# External libraries

# Internal libraries
from .arc import fit_spiral_to_image
from .definitions import ImageFloatArray


log = logging.getLogger(__name__)


def calculate_arc_merge_error(
    first_cluster_array: ImageFloatArray, second_cluster_array: ImageFloatArray
) -> float:
    first_sum = first_cluster_array.sum()
    second_sum = second_cluster_array.sum()
    assert first_sum > 0
    assert second_sum > 0
    total_sum = first_sum + second_sum
    # Adjust weights
    first_cluster_array *= total_sum / first_sum
    second_cluster_array *= total_sum / second_sum

    # Fit spirals to each cluster individually
    first_fit = fit_spiral_to_image(first_cluster_array)
    second_fit = fit_spiral_to_image(second_cluster_array)

    combined_cluster_array = first_cluster_array + second_cluster_array
    # Fit a spiral to both clusters at the same time
    first_merged_fit = fit_spiral_to_image(
        combined_cluster_array, initial_pitch_angle=first_fit.pitch_angle
    )
    second_merged_fit = fit_spiral_to_image(
        combined_cluster_array, initial_pitch_angle=second_fit.pitch_angle
    )
    if first_merged_fit.total_error <= second_merged_fit.total_error:
        merged_fit = first_merged_fit
    else:
        merged_fit = second_merged_fit

    first_cluster_indices = (first_cluster_array > 0)[combined_cluster_array > 0]
    # Get the error of the merged spiral for each individual cluster
    first_cluster_errors = merged_fit.errors[first_cluster_indices].sum()
    second_cluster_errors = merged_fit.errors[~first_cluster_indices].sum()

    # Readjust errors from normalised cluster arrays
    first_cluster_error_weighted = first_fit.total_error / first_sum
    second_cluster_error_weighted = second_fit.total_error / second_sum

    # Calculate the arc merge error ratio for each cluster
    ratios = (
        ((first_cluster_errors / first_sum) / first_cluster_error_weighted),
        ((second_cluster_errors / second_sum) / second_cluster_error_weighted),
    )
    # Return the worst error ratio
    return max(ratios)
