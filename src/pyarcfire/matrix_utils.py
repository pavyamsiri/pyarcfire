"""Utilities regarding matrices."""

from typing import TypeAlias, TypeVar

import numpy as np
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
    sparse.csc_array,
    sparse.csr_array,
    sparse.dok_array,
    sparse.lil_array,
    sparse.csc_matrix,
    sparse.csr_matrix,
    sparse.dok_matrix,
    sparse.lil_matrix,
)


_SCT = TypeVar("_SCT", bound=np.generic)
_Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[_SCT]]
_Array1D_f64: TypeAlias = _Array1D[np.float64]


def is_sparse_matrix_hollow(matrix: SparseMatrix) -> bool:
    """Check that the sparse matrix is hollow i.e. diagonal is all zero.

    Parameters
    ----------
    matrix : SparseMatrix
        The matrix to assert about.

    Returns
    -------
    is_hollow : bool
        Returns true if the matrix is hollow otherwise false.

    """
    return np.allclose(matrix.diagonal(), 0)


def is_sparse_matrix_symmetric(matrix: SparseMatrix) -> bool:
    """Check that the sparse matrix is symmetric i.e. it is equal to its transpose.

    Parameters
    ----------
    matrix : SparseMatrix
        The matrix to assert about.

    Returns
    -------
    is_symmetric : bool
        Returns true if the matrix is symmetric otherwise false.

    """
    return (matrix - matrix.transpose()).count_nonzero() == 0


def get_nonzero_values(matrix: SparseMatrixSupportsIndex) -> _Array1D_f64:
    """Return all non-zero values as a flat array.

    Parameters
    ----------
    matrix : SparseMatrixSupportsIndex
        The matrix to assert about.

    Returns
    -------
    values : Array1D[f64]
        The non-zero values.

    """
    return np.asarray(matrix[matrix.nonzero()], dtype=np.float64).flatten()
