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
from .orientation import generate_orientation_fields

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

    field = generate_orientation_fields(contrast_image)
    field = field.resize(128, 128)
    strengths = field.get_strengths()
    nonzero_cells = np.count_nonzero(strengths)
    total_cells = strengths.size
    print(
        f"Strengths has {nonzero_cells}/{total_cells} |{100*(nonzero_cells/total_cells):.2f}%| nonzero cells"
    )

    fig = plt.figure()
    original_axis = fig.add_subplot(131)
    original_axis.imshow(image, cmap="gray")
    original_axis.set_title("Original image")
    original_axis.set_axis_off()

    contrast_axis = fig.add_subplot(132)
    contrast_axis.imshow(contrast_image, cmap="gray")
    contrast_axis.set_title(
        rf"Unsharp image $\text{{Radius}} = {UNSHARP_MASK_RADIUS}, \; \text{{Amount}} = {UNSHARP_MASK_AMOUNT}$"
    )
    contrast_axis.set_axis_off()

    space_range = np.arange(field.shape[0])
    x, y = np.meshgrid(space_range, -space_range)
    orientation_axis = fig.add_subplot(133)
    orientation_axis.quiver(x, y, field.x, field.y, color="tab:blue", headaxislength=0)
    orientation_axis.set_aspect("equal")
    orientation_axis.set_title("Orientation field")
    orientation_axis.set_axis_off()

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
