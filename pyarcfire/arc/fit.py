# Standard libraries

# External libraries

# Internal libraries
from pyarcfire.definitions import ImageArray
from .common import LogSpiralFitResult
from .multiple_revolution import fit_spiral_to_image_multiple_revolution
from .single_revolution import fit_spiral_to_image_single_revolution


def fit_spiral_to_image(
    image: ImageArray,
    initial_pitch_angle: float = 0,
    allow_multiple_revolutions: bool = True,
) -> LogSpiralFitResult:
    if allow_multiple_revolutions:
        return fit_spiral_to_image_multiple_revolution(
            image, initial_pitch_angle=initial_pitch_angle
        )
    else:
        return fit_spiral_to_image_single_revolution(
            image, initial_pitch_angle=initial_pitch_angle
        )
