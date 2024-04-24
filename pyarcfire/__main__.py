# Standard libraries
import argparse
import logging
from typing import Sequence

# External libraries
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters

# Internal libraries
from .log_utils import setup_logging

log = logging.getLogger(__name__)


def main(raw_args: Sequence[str]) -> None:
    args = _parse_args(raw_args)
    # Load image
    image = np.asarray(Image.open(args.input_path).convert("L"))

    UNSHARP_MASK_RADIUS: float = 25
    UNSHARP_MASK_AMOUNT: float = 6
    contrast_image = filters.unsharp_mask(
        image, radius=UNSHARP_MASK_RADIUS, amount=UNSHARP_MASK_AMOUNT
    )

    from .orientation import generate_orientation_fields

    field, strengths, directions = generate_orientation_fields(contrast_image)
    print(strengths.shape)
    print(field.shape)
    print(directions.shape)

    fig = plt.figure()
    left_axis = fig.add_subplot(121)
    left_axis.imshow(directions, cmap="gray")
    left_axis.set_title("Original image")
    left_axis.set_axis_off()

    right_axis = fig.add_subplot(122)
    right_axis.imshow(contrast_image, cmap="gray")
    right_axis.set_title(
        rf"Unsharp image $\text{{Radius}} = {UNSHARP_MASK_RADIUS}, \; \text{{Amount}} = {UNSHARP_MASK_AMOUNT}$"
    )
    right_axis.set_axis_off()

    fig.tight_layout()
    plt.show()
    plt.close()


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyarcfire",
        description="Python port of SpArcFiRe, a program that finds and reports spiral features in images.",
    )
    parser.add_argument(
        "-i",
        "--i",
        type=str,
        dest="input_path",
        help="Path to the input image.",
        required=True,
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    import sys

    setup_logging()
    main(sys.argv[1:])
