"""This module contains utilities regarding matrices."""

# Standard libraries
from typing import Union

# External libraries
import numpy as np
from scipy import sparse

# Internal libraries


def is_sparse_matrix_hollow(
    matrix: Union[sparse.coo_matrix, sparse.csr_matrix, sparse.csc_matrix],
) -> bool:
    """Utility used to assert that a sparse matrix is hollow i.e. no non-zero values in diagonal.

    Parameters
    ----------
    matrix : sparse.coo_matrix | sparse.csr_matrix | sparse.csc_matrix
        The matrix to assert about.

    Returns
    -------
    is_hollow : bool
        Returns true if the matrix is hollow otherwise false.
    """
    is_hollow = np.allclose(matrix.diagonal(), 0)
    return is_hollow


def is_sparse_matrix_symmetric(
    matrix: Union[sparse.coo_matrix, sparse.csr_matrix, sparse.csc_matrix],
) -> bool:
    """Utility used to assert that a sparse matrix is symmetric i.e. it is equal to its transpose.

    Parameters
    ----------
    matrix : sparse.coo_matrix | sparse.csr_matrix | sparse.csc_matrix
        The matrix to assert about.

    Returns
    -------
    is_symmetric : bool
        Returns true if the matrix is symmetric otherwise false.
    """
    is_symmetric = (matrix - matrix.transpose()).count_nonzero() == 0  # type:ignore
    return is_symmetric
