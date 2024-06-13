"""Routines to generate clusters representing spiral arm segments from a similarity matrix."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy import sparse

from .array_utils import get_origin_points_unnested
from .debug_utils import benchmark
from .matrix_utils import get_nonzero_values, is_sparse_matrix_symmetric
from .merge import calculate_arc_merge_error

if TYPE_CHECKING:
    from numpy.typing import NDArray

    SparseMatrix = TypeVar(
        "SparseMatrix",
        sparse.coo_array,
        sparse.bsr_array,
        sparse.csc_array,
        sparse.csr_array,
        sparse.dia_array,
        sparse.dok_array,
        sparse.lil_array,
        sparse.coo_matrix,
        sparse.bsr_matrix,
        sparse.csc_matrix,
        sparse.csr_matrix,
        sparse.dia_matrix,
        sparse.dok_matrix,
        sparse.lil_matrix,
    )
    SparseArray = TypeVar(
        "SparseArray",
        sparse.coo_array,
        sparse.bsr_array,
        sparse.csc_array,
        sparse.csr_array,
        sparse.dia_array,
        sparse.dok_array,
        sparse.lil_array,
    )
    SparseMatrixSupportsIndex = TypeVar(
        "SparseMatrixSupportsIndex",
        sparse.csc_array,
        sparse.csr_array,
        sparse.dok_array,
        sparse.lil_array,
        sparse.csc_matrix,
        sparse.csr_matrix,
        sparse.dok_matrix,
        sparse.lil_matrix,
    )
    SparseArraySupportsIndex = TypeVar(
        "SparseArraySupportsIndex",
        sparse.csc_array,
        sparse.csr_array,
        sparse.dok_array,
        sparse.lil_array,
    )


log: logging.Logger = logging.getLogger(__name__)


FloatType = np.float32


@dataclass
class GenerateClustersSettings:
    """Settings to configure generate_orientation_field.

    Attributes
    ----------
    stop_threshold: float
        The minimum similarity allowed between clusters before stopping
        merges.
    error_ratio_threshold : float
        The maximum error ratio allowed for a merge between two clusters
        to be permitted. This error ratio is the ratio of an arc fit's error to
        the merged cluster relative to the error of two arc fits to the clusters
        individually.
    merge_check_minimum_cluster_size : int
        The maximum size of each cluster before their merges become checked and
        potentially stopped.
    remove_central_cluster : bool
        Set this flag to remove clusters that touch the centre.

    """

    stop_threshold: float = 0.15
    error_ratio_threshold: float = 2.5
    merge_check_minimum_cluster_size: int = 25
    minimum_cluster_size: int = 150
    remove_central_cluster: bool = True


DEFAULT_CLUSTER_SETTINGS: GenerateClustersSettings = GenerateClustersSettings()


class Cluster:
    """A cluster of pixels grouped by their similarity and spatial distance.

    A cluster should represent a spiral arm segment and be fit well by a log spiral.
    """

    def __init__(self, points: Iterable[int]) -> None:
        """Create a cluster from a given set of points.

        Parameters
        ----------
        points : Iterable[int]
            The points the cluster consists of.

        """
        self.points: list[int] = list(points)

    def __str__(self) -> str:
        """Return a string representation of OrientationField."""
        return f"Cluster(size={self.size}, hash={hash(self):X})"

    def __repr__(self) -> str:
        """Return a string representation of OrientationField."""
        return str(self)

    def __contains__(self, item: Iterable[int] | int) -> bool:
        """Check whether a point or a set of points is contained by the cluster.

        Parameters
        ----------
        item : Iterable[int] | int
            Either a single index or an iterable of point indices.

        Returns
        -------
        bool
            This is `True` if the point is contained or all points are contained within
            the cluster.

        """
        if isinstance(item, Iterable):
            elements: Iterable[Any] = item
            return any(elem in self.points for elem in elements)
        return item in self.points

    @property
    def size(self) -> int:
        """int: The number of points the cluster contains."""
        return len(self.points)

    def get_masked_image(self, image: NDArray[FloatType]) -> NDArray[FloatType]:
        """Return the given image masked by the cluster.

        Parameter
        ---------
        image : NDArray[FloatType]
            The image to mask.

        Returns
        -------
        masked_image : NDArray[FloatType]
            The masked image.

        """
        num_rows, num_columns = image.shape
        row_indices, column_indices = np.unravel_index(self.points, (num_rows, num_columns))
        mask = np.ones_like(image, dtype=np.bool_)
        mask[row_indices, column_indices] = False
        masked_image = image.copy()
        masked_image[mask] = 0
        return masked_image

    @staticmethod
    def list_to_array(image: NDArray[FloatType], clusters: Iterable[Cluster]) -> NDArray[FloatType] | None:
        """Convert a list of clusters and an image into an array of the same image masked by each different cluster.

        Parameters
        ----------
        image : NDArray[FloatType]
            The image to mask.
        clusters : Sequence[Cluster]
            The clusters to mask the image with.


        Returns
        -------
        NDArray[FloatType] | None
            The image masked by the different clusters. This returns None if there
            are no clusters given.

        """
        array_list = [cluster.get_masked_image(image) for cluster in clusters]
        if len(array_list) == 0:
            return None
        return np.dstack(array_list)

    @staticmethod
    def combine(left: Cluster, right: Cluster) -> Cluster:
        """Return the merging of two clusters.

        Parameters
        ----------
        left : Cluster
            The cluster to merge.
        right : Cluster
            The other cluster to merge.

        Returns
        -------
        Cluster
            The merged cluster.

        """
        all_points = left.points + right.points
        return Cluster(all_points)


@benchmark
def generate_clusters(
    image: NDArray[FloatType],
    input_similarity_matrix: SparseArray,
    stop_threshold: float,
    error_ratio_threshold: float,
    merge_check_minimum_cluster_size: int,
    minimum_cluster_size: int,
    *,
    remove_central_cluster: bool,
) -> NDArray[FloatType] | None:
    """Perform single linkage clustering on an image given its corresponding similarity matrix.

    The clusters are merged using single linkage clustering with a single modification. Sufficiently
    large cluster pairs will first be checked if their resulting merged cluster will fit a spiral
    sufficiently well compared to the clusters individually before being merged. If the merged cluster
    fits sufficiently poorly then the clusters will not be merged.

    Parameters
    ----------
    image : NDArray[FloatType]
        The image to find clusters in.
    input_similarity_matrix : SparseArray
        The similarity matrix representing pixel similarities in the image.
    stop_threshold : float
        The minimum similarity value where cluster merging can happen.
    error_ratio_threshold : float, optional
        The maximum merge error ratio allowed for a cluster merge to happen. Default is 2.5.
    merge_check_minimum_cluster_size : int, optional
        The minimum size for each cluster when performing a merge check. Default is 25.
    minimum_cluster_size : int, optional
        The minimum size a cluster is allowed to be after merging. Default is 150.
    remove_central_cluster : bool, optional
        If this flag is set, the cluster that contains the origin will be removed. Default is true.

    Returns
    -------
    cluster_arrays : Optional[NDArray[FloatType]]
        The image masked by each cluster. This function may not find any suitable clusters in which case
        it will return None.

    """
    # The image must be 2D
    if len(image.shape) != 2:
        msg = "The image must be a 2D array!"
        raise ValueError(msg)

    # Verify symmetry and implicitly squareness
    if not is_sparse_matrix_symmetric(input_similarity_matrix):
        msg = "Similarity matrix must be symmetric!"
        raise ValueError(msg)

    # TODO(pavyamsiri): Should check that the input matrix is not self-similar

    # Change to an index that supports indexing
    similarity_matrix: sparse.csr_array = input_similarity_matrix.tocsr()
    similarity_matrix.eliminate_zeros()
    num_pixels_from_matrix = similarity_matrix.shape[0]
    num_pixels_from_image = image.shape[0] * image.shape[1]
    if num_pixels_from_image != num_pixels_from_matrix:
        msg = "The similarity matrix's size is inconsistent with the image's size."
        raise ValueError(msg)

    # Diagnostics
    check_arc_merge_count: int = 0
    merge_stop_count: int = 0

    # Indices of pixels which have a non-zero similarity with another pixel
    points = np.flatnonzero(similarity_matrix.sum(axis=0))
    # Assign a cluster to each pixel with non-zero similarity to another pixel
    clusters = {idx: Cluster([idx]) for idx in points}

    # Merge clusters via single-linkage clustering
    while True:
        # Pop the pair with the largest similarity
        max_idx = similarity_matrix.argmax()
        unraveled_idx = np.unravel_index(max_idx, similarity_matrix.shape)
        first_idx = int(unraveled_idx[0])
        second_idx = int(unraveled_idx[1])

        # Leave loop if similarity value is below the stop threshold
        value = similarity_matrix[first_idx, second_idx]
        if value < stop_threshold:
            break

        # Get the clusters
        first_cluster = clusters[first_idx]
        second_cluster = clusters[second_idx]
        # Perform a merge check on sufficiently large cluster pairs
        if min(first_cluster.size, second_cluster.size) > merge_check_minimum_cluster_size:
            check_arc_merge_count += 1
            first_cluster_array = first_cluster.get_masked_image(image)
            second_cluster_array = second_cluster.get_masked_image(image)
            # Check if merging will result in a sufficently worse spiral fit than
            # if the clusters were separate.
            merge_error = calculate_arc_merge_error(first_cluster_array, second_cluster_array)
            # Error after merging is sufficiently bad to halt merging
            if merge_error > error_ratio_threshold:
                merge_stop_count += 1
                # Remove links
                similarity_matrix[first_idx, second_idx] = 0
                similarity_matrix[second_idx, first_idx] = 0
                continue

        # Combine clusters together
        target_cluster = Cluster.combine(first_cluster, second_cluster)
        # Assign cluster to first cluster
        clusters[first_idx] = target_cluster
        # Remove the second cluster
        del clusters[second_idx]

        # Update the similarity matrix with the inclusion of the merged cluster
        similarity_matrix = _update_similarity_matrix(similarity_matrix, first_idx, second_idx)
        # Remove the links associated with the deleted cluster
        similarity_matrix = _clear_similarity_matrix_row_column(similarity_matrix, second_idx)

    # Remove clusters below minimum size
    clusters_list = list(clusters.values())
    clusters_list = [cluster for cluster in clusters_list if cluster.size >= minimum_cluster_size]

    # Remove cluster that contains the centre
    if remove_central_cluster:
        log.info("[green]DIAGNOST[/green]: Removing central cluster...")
        central_points = get_origin_points_unnested(image)
        central_idx = np.ravel_multi_index(central_points, image.shape)
        clusters_list = [cluster for cluster in clusters_list if central_idx not in cluster]

    cluster_arrays = Cluster.list_to_array(image, clusters_list)

    # Show diagnostics of merging step
    log.info("[green]DIAGNOST[/green]: Number of clusters = %d", len(clusters))
    log.info("[green]DIAGNOST[/green]: Checked %d possible cluster merges", check_arc_merge_count)
    log.info("[green]DIAGNOST[/green]: Stopped %d cluster merges", merge_stop_count)
    return cluster_arrays


def _update_similarity_matrix(
    similarity_matrix: sparse.csr_array,
    target_idx: int,
    source_idx: int,
) -> sparse.csr_array:
    """Return the similarity matrix updated to have new similarity values after the merging of two clusters.

    The target cluster's row and column is updated to have the maximum similarity value between the two clusters.
    Note that this function does not remove the other cluster's similarity values.

    Parameters
    ----------
    similarity_matrix : sparse.csr_array
        The similarity matrix to update.
    target_idx : int
        The index of the row and column to put the updated similarity values into.
        This represents the index of the target cluster (the cluster that does not get deleted).
    source_idx : int
        The index of the row and column to take similarity values from.
        This represents the index of the source cluster (the cluster that does get deleted).

    Returns
    -------
    updated_matrix : sparse.csr_array
        The matrix with its rows and columns updated with the new similarity values.

    """
    # Merged cluster similarity is the maximum possible similarity from the set of cluster points
    old_similarity_values = similarity_matrix[[target_idx], :]
    # The updated values max(Ti, Si)
    new_similarity_values = similarity_matrix[[target_idx], :].maximum(similarity_matrix[[source_idx], :])
    # Remove self-similarity
    new_similarity_values[0, [target_idx]] = 0

    # AIM: Construct matrix such that when added, the target row and column is updated to the desired values
    add_matrix_values = new_similarity_values - old_similarity_values
    # The target row/column already has the right similarity values
    updated_matrix: sparse.csr_array
    if add_matrix_values.count_nonzero() == 0:
        updated_matrix = similarity_matrix
    # The target row/column must be updated
    else:
        # Construct a sparse matrix with the only non-zero row/column being our target row/column
        _, add_matrix_indices = add_matrix_values.nonzero()
        add_matrix_values_array = get_nonzero_values(add_matrix_values)
        num_nonzero = len(add_matrix_indices)
        add_matrix_row_indices = np.tile(add_matrix_indices, 2)
        add_matrix_row_indices[num_nonzero:] = target_idx
        add_matrix_column_indices = np.tile(add_matrix_indices, 2)
        add_matrix_column_indices[:num_nonzero] = target_idx
        add_matrix = sparse.coo_matrix(
            (
                np.tile(add_matrix_values_array, 2),
                (
                    add_matrix_row_indices,
                    add_matrix_column_indices,
                ),
            ),
            shape=similarity_matrix.shape,
        )
        # Add matrix
        updated_matrix = similarity_matrix + add_matrix
    return updated_matrix


def _clear_similarity_matrix_row_column(
    similarity_matrix: SparseMatrixSupportsIndex,
    clear_idx: int,
) -> SparseMatrixSupportsIndex:
    """Return the similarity matrix with the row and column at the given index cleared to zero.

    Parameters
    ----------
    similarity_matrix : SparseMatrixSupportsIndex
        The similarity matrix to update.
    clear_idx: int
        The index of the row and column to clear/zero.

    Returns
    -------
    updated_matrix : sparse.csr_array
        The matrix with one of its rows and columns cleared.

    """
    # Construct matrix such that when added the target row and column are cleared
    values = get_nonzero_values(similarity_matrix[[clear_idx], :])
    indices: NDArray[np.int32] = similarity_matrix[[clear_idx], :].nonzero()[1]
    num_nonzero = len(indices)
    clear_matrix_row_indices = np.tile(indices, 2)
    clear_matrix_row_indices[num_nonzero:] = clear_idx
    clear_matrix_column_indices = np.tile(indices, 2)
    clear_matrix_column_indices[:num_nonzero] = clear_idx
    clear_matrix = sparse.coo_matrix(
        (
            np.tile(values, 2),
            (
                clear_matrix_row_indices,
                clear_matrix_column_indices,
            ),
        ),
        shape=similarity_matrix.shape,
    )

    # Add matrix
    return similarity_matrix - clear_matrix
