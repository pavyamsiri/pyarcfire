# Internal libraries
from typing import TypeAlias

# External libraries
import numpy as np
from numpy import typing as npt


FloatArray1D: TypeAlias = npt.NDArray[np.floating]
ImageArray: TypeAlias = npt.NDArray[np.floating]
ImageArraySequence: TypeAlias = npt.NDArray[np.floating]
