# Standard libraries
import logging


# Internal libraries
from .arc import fit_spiral_to_image
from .definitions import ImageArray


log = logging.getLogger(__name__)


def calculate_arc_merge_error(
    first_cluster_array: ImageArray, second_cluster_array: ImageArray
) -> float:
    first_fit = fit_spiral_to_image(first_cluster_array)
    second_fit = fit_spiral_to_image(second_cluster_array)
    first_sum = first_cluster_array.sum()
    second_sum = second_cluster_array.sum()
    assert first_sum > 0
    assert second_sum > 0
    total_sum = first_sum + second_sum
    # Adjust weights
    first_cluster_array *= total_sum / first_sum
    second_cluster_array *= total_sum / second_sum
    combined_cluster_array = first_cluster_array + second_cluster_array
    merged_fit = fit_spiral_to_image(combined_cluster_array)

    first_merge_error = merged_fit.error / first_fit.error
    second_merge_error = merged_fit.error / second_fit.error
    return max(first_merge_error, second_merge_error)
