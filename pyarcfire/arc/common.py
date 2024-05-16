# Standard libraries
from dataclasses import dataclass

# External libraries
import numpy as np

# Internal libraries
from pyarcfire.definitions import FloatArray1D
from .functions import log_spiral


@dataclass
class LogSpiralFitResult:
    offset: float
    pitch_angle: float
    initial_radius: float
    arc_bounds: tuple[float, float]
    total_error: float
    errors: FloatArray1D

    def calculate_cartesian_coordinates(
        self, num_points: int
    ) -> tuple[FloatArray1D, FloatArray1D]:
        start_angle = self.offset
        end_angle = start_angle + self.arc_bounds[1]

        theta = np.linspace(start_angle, end_angle, num_points)
        radii = log_spiral(theta, self.offset, self.pitch_angle, self.initial_radius)
        x = radii * np.cos(theta)
        y = radii * np.sin(theta)
        return (x, y)
