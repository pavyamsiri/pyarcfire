"""This module contains useful type definitions, especially pertaining to arrays.
Note that despite the type names referencing array shape, this is not reflected in
the python type system, they are just for hinting to the reader.
"""

# Internal libraries
from typing import TypeAlias

# External libraries
import numpy as np
from numpy import typing as npt


# 1D Arrays
FloatArray1D: TypeAlias = npt.NDArray[np.floating]
IntegerArray1D: TypeAlias = npt.NDArray[np.integer]
BoolArray1D: TypeAlias = npt.NDArray[np.bool_]
NumberArray1D: TypeAlias = FloatArray1D | IntegerArray1D
Array1D: TypeAlias = BoolArray1D | NumberArray1D


# 2D Arrays
FloatArray2D: TypeAlias = npt.NDArray[np.floating]
IntegerArray2D: TypeAlias = npt.NDArray[np.integer]
BoolArray2D: TypeAlias = npt.NDArray[np.bool_]
NumberArray2D: TypeAlias = FloatArray2D | IntegerArray2D
Array2D: TypeAlias = BoolArray2D | NumberArray2D


# 3D Arrays
FloatArray3D: TypeAlias = npt.NDArray[np.floating]
IntegerArray3D: TypeAlias = npt.NDArray[np.integer]
BoolArray3D: TypeAlias = npt.NDArray[np.bool_]
NumberArray3D: TypeAlias = FloatArray3D | IntegerArray3D
Array3D: TypeAlias = BoolArray3D | NumberArray3D
