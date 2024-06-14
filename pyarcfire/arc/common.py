"""A dataclass that stores the results of a log spiral fit."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy import float32, float64
from numpy.typing import NDArray

from .functions import log_spiral

FloatType = TypeVar("FloatType", float32, float64)


@dataclass
class LogSpiralFitResult(Generic[FloatType]):
    """The result of a log spiral fit to a cluster.

    A log spiral is a curve of the form
        R = R0 * exp(-a * (theta - phi))
    where R0 is the initial radius, a is the pitch angle and phi
    is the offset. R and theta are the radial and polar coordinate
    respectively.

    Attributes
    ----------
    offset : float
        The offset in radians.
    pitch_angle : float
        The pitch angle. TODO: I think the pitch angle is actually arctan(a)
    initial_radius : float
        The initial radius in pixels.
    arc_bounds : tuple[float, float]
        The azimuthal bounds of the arc.
    total_error : float
        The sum of the square residuals.
    errors : NDArray[FloatType]
        The residuals.
    has_multiple_revolutions : bool
        The arc revolves fully at least once if this is `True`.

    """

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
    ) -> tuple[NDArray[float32], NDArray[float32]]:
        """Return the x and y Cartesian coordinates of the log spiral.

        Parameters
        ----------
        num_points : int
            The number of points to approximate the log spiral with.
        pixel_to_distance : float
            The unit conversion factor to convert pixels to another unit.
        flip_y : bool
            Set this flag to flip the y coordinates.

        Returns
        -------
        x : NDArray[float32]
            The x coordinates.
        z : NDArray[float32]
            The z coordinates.

        """
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
