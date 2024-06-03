"""This module contains utilities regarding matrices."""

from typing import cast, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

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

SparseMatrixSupportsIndex = TypeVar(
    "SparseMatrixSupportsIndex",
    sparse.bsr_array,
    sparse.csc_array,
    sparse.csr_array,
    sparse.dok_array,
    sparse.lil_array,
    sparse.bsr_matrix,
    sparse.csc_matrix,
    sparse.csr_matrix,
    sparse.dok_matrix,
    sparse.lil_matrix,
)


def is_sparse_matrix_hollow(matrix: SparseMatrix) -> bool:
    """Utility used to assert that a sparse matrix is hollow i.e. no non-zero values in diagonal.

    Parameters
    ----------
    matrix : SparseMatrix
        The matrix to assert about.

    Returns
    -------
    is_hollow : bool
        Returns true if the matrix is hollow otherwise false.
    """
    is_hollow = np.allclose(matrix.diagonal(), 0)
    return is_hollow


def is_sparse_matrix_symmetric(matrix: SparseMatrix) -> bool:
    """Utility used to assert that a sparse matrix is symmetric i.e. it is equal to its transpose.

    Parameters
    ----------
    matrix : SparseMatrix
        The matrix to assert about.

    Returns
    -------
    is_symmetric : bool
        Returns true if the matrix is symmetric otherwise false.
    """
    is_symmetric = cast(bool, (matrix - matrix.transpose()).count_nonzero() == 0)
    return is_symmetric


def get_nonzero_values(matrix: SparseMatrixSupportsIndex) -> NDArray[np.float32]:
    """Returns all non-zero values as a flat array.

    Parameters
    ----------
    matrix : SparseMatrixSupportsIndex
        The matrix to assert about.

    Returns
    -------
    NDArray[np.float32]
        The non-zero values.
    """
    return np.asarray(matrix[matrix.nonzero()], dtype=np.float32).flatten()
