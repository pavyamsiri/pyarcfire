"""A dataclass that stores the results of a log spiral fit."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, TypeVar

import numpy as np
from numpy import float32, float64
from numpy.typing import NDArray

from .functions import log_spiral

FloatType = TypeVar("FloatType", float32, float64)


class Chirality(Enum):
    """Chirality of a log spiral.

    Variants
    --------
    CLOCKWISE
        The spiral winds clockwise or "Z"-wise.
    COUNTER_CLOCKWISE
        The spiral winds counter-clockwise or "S"-wise.
    NONE
        The spiral does not wind at all i.e. when pitch angle is exactly zero.

    """

    CLOCKWISE = auto()
    COUNTER_CLOCKWISE = auto()
    NONE = auto()


@dataclass
class LogSpiralFitResult(Generic[FloatType]):
    """The result of a log spiral fit to a cluster.

    A log spiral is a curve of the form
        R = R0 * exp(-k * (theta - phi))
    where R0 is the initial radius, k is the growth factor and phi
    is the offset. R and theta are the radial and polar coordinate
    respectively.

    Attributes
    ----------
    offset : float
        The offset in radians.
    growth_factor : float
        The growth factor.
    initial_radius : float
        The initial radius in pixels.
    arc_extent : float
        The azimuthal extent of the arc in radians.
    total_error : float
        The sum of the square residuals.
    errors : NDArray[FloatType]
        The residuals.
    has_multiple_revolutions : bool
        The arc revolves fully at least once if this is `True`.

    """

    offset: float
    growth_factor: float
    initial_radius: float
    arc_extent: float
    total_error: float
    errors: NDArray[FloatType]
    has_multiple_revolutions: bool

    def __post_init__(self) -> None:
        """Calculate properties of the log spiral."""
        # Pitch angle
        self._pitch_angle = np.arctan(self.growth_factor)
        # Arc length
        start_angle = self.offset
        end_angle = start_angle + self.arc_extent
        lengths = log_spiral(
            np.asarray(
                [start_angle, end_angle],
            ),
            self.offset,
            self.growth_factor,
            self.initial_radius,
            use_modulo=self.has_multiple_revolutions,
        )
        self._arc_length: float
        if np.isclose(np.sin(self._pitch_angle), 0):
            self._arc_length = self.initial_radius * self.arc_extent
        else:
            self._arc_length = abs(lengths[1] - lengths[0]) / np.sin(self._pitch_angle)
        # Winding direction
        winding_direction: float = np.sign(self._pitch_angle)
        self._chirality: Chirality
        if np.isclose(winding_direction, 0):
            self._chirality = Chirality.NONE
        elif winding_direction > 0:
            self._chirality = Chirality.CLOCKWISE
        else:
            self._chirality = Chirality.COUNTER_CLOCKWISE

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
        end_angle = start_angle + self.arc_extent

        theta = np.linspace(start_angle, end_angle, num_points, dtype=np.float32)
        radii = pixel_to_distance * log_spiral(
            theta,
            self.offset,
            self.growth_factor,
            self.initial_radius,
            use_modulo=not self.has_multiple_revolutions,
        )
        x = np.multiply(radii, np.cos(theta))
        y = y_flip_factor * np.multiply(radii, np.sin(theta))
        return (x, y)

    @property
    def pitch_angle_in_degrees(self) -> float:
        """float: The pitch angle in degrees from [-180, 180]."""
        return np.rad2deg(self._pitch_angle)

    @property
    def arc_length(self) -> float:
        """float: The arc length in pixel units."""
        return self._arc_length

    @property
    def chirality(self) -> Chirality:
        """Chirality: The chirality of the spiral."""
        return self._chirality
