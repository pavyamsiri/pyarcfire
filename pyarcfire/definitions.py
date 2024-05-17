# Internal libraries
from typing import TypeAlias

# External libraries
import numpy as np
from numpy import typing as npt


FloatArray1D: TypeAlias = npt.NDArray[np.floating]
IntegerArray1D: TypeAlias = npt.NDArray[np.integer]
BoolArray1D: TypeAlias = npt.NDArray[np.bool_]
ImageArray: TypeAlias = npt.NDArray[np.floating]
ImageArraySequence: TypeAlias = npt.NDArray[np.floating]
