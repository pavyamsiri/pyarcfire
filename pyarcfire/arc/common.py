from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .functions import log_spiral

FloatType = TypeVar("FloatType", np.float32, np.float64)


@dataclass
class LogSpiralFitResult(Generic[FloatType]):
    offset: float
    pitch_angle: float
    initial_radius: float
    arc_bounds: tuple[float, float]
    total_error: float
    errors: NDArray[FloatType]
    has_multiple_revolutions: bool

    def calculate_cartesian_coordinates(
        self,
        num_points: int,
        pixel_to_distance: float,
        *,
        flip_y: bool = False,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        y_flip_factor: float = 1.0 if not flip_y else -1.0
        start_angle = self.offset
        end_angle = start_angle + self.arc_bounds[1]

        theta = np.linspace(start_angle, end_angle, num_points, dtype=np.float32)
        radii = pixel_to_distance * log_spiral(
            theta,
            self.offset,
            self.pitch_angle,
            self.initial_radius,
            use_modulo=not self.has_multiple_revolutions,
        )
        x = np.multiply(radii, np.cos(theta))
        y = y_flip_factor * np.multiply(radii, np.sin(theta))
        return (x, y)
