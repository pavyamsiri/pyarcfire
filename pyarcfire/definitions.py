# Internal libraries
from typing import TypeAlias

# External libraries
import numpy as np
from numpy import typing as npt


FloatArray1D: TypeAlias = npt.NDArray[np.floating]
IntegerArray1D: TypeAlias = npt.NDArray[np.integer]
BoolArray1D: TypeAlias = npt.NDArray[np.bool_]
ImageFloatArray: TypeAlias = npt.NDArray[np.floating]
ImageIntegerArray: TypeAlias = npt.NDArray[np.integer]
ImageBoolArray: TypeAlias = npt.NDArray[np.bool_]
ImageArrayUnion: TypeAlias = ImageFloatArray | ImageIntegerArray | ImageBoolArray
ImageFloatArraySequence: TypeAlias = npt.NDArray[np.floating]
