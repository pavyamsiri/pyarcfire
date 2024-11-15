"""Common type aliases."""

from typing import Any, Literal, TypeAlias

import numpy as np

AnyBool: TypeAlias = bool | np.bool_ | Literal[0, 1]
AnyInt: TypeAlias = int | np.integer[Any] | np.bool_
AnyReal: TypeAlias = int | float | np.floating[Any] | np.integer[Any] | np.bool_
AnyComplex: TypeAlias = int | float | complex | np.number[Any] | np.bool_
AnyChar: TypeAlias = str | bytes  # `np.str_ <: builtins.str` and `np.bytes_ <: builtins.bytes`
AnyScalar: TypeAlias = int | float | complex | AnyChar | np.generic
