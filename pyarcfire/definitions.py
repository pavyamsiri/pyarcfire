"""This module contains useful type definitions, especially pertaining to arrays.
Note that despite the type names referencing array shape, this is not reflected in
the python type system, they are just for hinting to the reader.
"""

# Standard libraries
from typing import Union

# External libraries
import numpy as np
from numpy import typing as npt


# 1D Arrays
FloatArray1D = npt.NDArray[np.floating]
IntegerArray1D = npt.NDArray[np.integer]
BoolArray1D = npt.NDArray[np.bool_]
NumberArray1D = Union[FloatArray1D, IntegerArray1D]
Array1D = Union[BoolArray1D, NumberArray1D]


# 2D Arrays
FloatArray2D = npt.NDArray[np.floating]
IntegerArray2D = npt.NDArray[np.integer]
BoolArray2D = npt.NDArray[np.bool_]
NumberArray2D = Union[FloatArray2D, IntegerArray2D]
Array2D = Union[BoolArray2D, NumberArray2D]


# 3D Arrays
FloatArray3D = npt.NDArray[np.floating]
IntegerArray3D = npt.NDArray[np.integer]
BoolArray3D = npt.NDArray[np.bool_]
NumberArray3D = Union[FloatArray3D, IntegerArray3D]
Array3D = Union[BoolArray3D, NumberArray3D]
