"""Finds spiral arcs in images of galaxies or anything with a spiral structure."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from numpy import float32
from PIL import Image

from pyarcfire import (
    GenerateClustersSettings,
    GenerateOrientationFieldSettings,
    GenerateSimilarityMatrixSettings,
    MergeClustersByFitSettings,
    UnsharpMaskSettings,
)

from .arc import fit_spiral_to_image
from .log_utils import setup_logging
from .spiral import detect_spirals_in_image

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

log: logging.Logger = logging.getLogger(__name__)


IMAGE_SIZE: int = 256


def main(raw_args: Sequence[str]) -> None:
    """Run the CLI.

    Parameters
    ----------
    raw_args : Sequence[str]
        The command line arguments.

    """
    args = _parse_args(raw_args)

    if not args.debug_flag:
        logging.getLogger("pyarcfire").setLevel("INFO")

    command: str = args.command
    if command == "image":
        process_from_image(args)
    else:
        log.critical("Command %s is unrecognised or not yet supported!", command)


def process_from_image(args: argparse.Namespace) -> None:
    """Preprocess an image and run it through the SpArcFiRe algorithm.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.

    """
    input_path: Path = Path(args.input_path)
    image: NDArray[float32] = _load_image(input_path)

    result = detect_spirals_in_image(
        image,
        preprocess=True,
    )

    if result is None:
        log.critical("Could not find any suitable clusters!")
        return

    unsharp_settings = result.unsharp_mask_settings
    cluster_arrays = result.get_cluster_arrays()

    image = result.get_image()
    contrast_image = result.get_unsharp_image()
    field = result.get_field()

    width: float = result.get_image_width() / 2 - 0.5
    height: float = result.get_image_height() / 2 - 0.5
    num_horizontal_pixels: int = result.get_image_width()
    num_vertical_pixels: int = result.get_image_height()
    log.debug("Dominant chirality %s", result.get_dominant_chirality())
    log.debug("Overall pitch angle %.2f degrees", np.rad2deg(result.get_overall_pitch_angle()))

    show_flag: bool = args.output_path is None

    fig = plt.figure(figsize=(14, 8))
    original_axis = fig.add_subplot(231)
    original_axis.imshow(
        image,
        extent=(-width, width, -height, height),
        cmap="gray",
    )
    original_axis.set_title("Original image")
    original_axis.set_axis_off()

    contrast_axis = fig.add_subplot(232)
    contrast_axis.imshow(
        contrast_image,
        extent=(-width, width, -height, height),
        cmap="gray",
    )
    contrast_axis.set_title(
        rf"Unsharp image $\mathrm{{Radius}} = {unsharp_settings.radius}, \; \mathrm{{Amount}} = {unsharp_settings.amount}$",
    )
    contrast_axis.set_axis_off()

    x_space_range = np.linspace(-width, width, num_horizontal_pixels)
    y_space_range = np.linspace(-height, height, num_vertical_pixels)
    x, y = np.meshgrid(x_space_range, -y_space_range)
    orientation_axis = fig.add_subplot(233)
    orientation_axis.quiver(x, y, field.x, field.y, color="tab:blue", headaxislength=0)
    orientation_axis.set_aspect("equal")
    orientation_axis.set_title("Orientation field")
    orientation_axis.set_axis_off()

    cluster_axis = fig.add_subplot(234)
    cluster_axis.set_title("Clusters")
    cluster_axis.set_xlim(-width, width)
    cluster_axis.set_ylim(-height, height)
    cluster_axis.set_axis_off()

    image_overlay_axis = fig.add_subplot(235)
    image_overlay_axis.imshow(
        image,
        extent=(-width, width, -height, height),
        cmap="gray",
    )
    image_overlay_axis.set_title("Original image overlaid with spirals")
    image_overlay_axis.set_xlim(-width, width)
    image_overlay_axis.set_ylim(-height, height)
    image_overlay_axis.set_axis_off()

    colored_image_overlay_axis = fig.add_subplot(236)
    colored_image_overlay_axis.set_title(
        "Original image colored with masks and overlaid with spirals",
    )
    colored_image_overlay_axis.set_xlim(-width, width)
    colored_image_overlay_axis.set_ylim(-height, height)
    colored_image_overlay_axis.set_axis_off()

    color_map = mpl.colormaps["hsv"]
    num_clusters: int = cluster_arrays.shape[2]
    colored_image = np.zeros((image.shape[0], image.shape[1], 4))
    colored_image[:, :, 0] = image / image.max()
    colored_image[:, :, 1] = image / image.max()
    colored_image[:, :, 2] = image / image.max()
    colored_image[:, :, 3] = 1.0
    for cluster_idx in range(num_clusters):
        current_array = cluster_arrays[:, :, cluster_idx]
        mask = current_array > 0
        cluster_mask = np.zeros((current_array.shape[0], current_array.shape[1], 4))
        cluster_color = color_map((cluster_idx + 0.5) / num_clusters)
        arc_color = color_map((num_clusters - cluster_idx + 0.5) / num_clusters)
        cluster_mask[mask, :] = cluster_color
        colored_image[mask, :] *= cluster_color
        cluster_axis.imshow(
            cluster_mask,
            extent=(-width, width, -height, height),
        )
        spiral_fit = fit_spiral_to_image(current_array)
        x, y = spiral_fit.calculate_cartesian_coordinates(100, pixel_to_distance=1, flip_y=False)
        cluster_axis.plot(
            x,
            y,
            color=arc_color,
            label=f"Cluster {cluster_idx}",
        )
        image_overlay_axis.plot(
            x,
            y,
            color=arc_color,
            label=f"Cluster {cluster_idx}",
        )
        colored_image_overlay_axis.plot(
            x,
            y,
            color=arc_color,
            label=f"Cluster {cluster_idx}",
        )
    colored_image_overlay_axis.imshow(
        colored_image,
        extent=(-width, width, -height, height),
    )

    fig.tight_layout()
    if show_flag:
        plt.show()
    else:
        fig.savefig(args.output_path)
        log.info(
            "[yellow]FILESYST[/yellow]: Saved plot to [yellow]%s[/yellow]",
            args.output_path,
        )
    plt.close()


def _load_image(input_path: Path) -> NDArray[float32]:
    """Load an image from a file.

    The returned image should in row-major storage form that is the first
    index indexes into rows while the second indexes into columns. The lowest
    indices also correspond to the most negative coordinates.

    Parameters
    ----------
    input_path : Path
        The path to the image.

    Returns
    -------
    image : NDArray[float32]
        The image in row-major form.

    """
    image: NDArray[float32]
    extension = Path(input_path).suffix.lstrip(".")
    # Numpy arrays from npy are already in row major form
    if extension == "npy":
        image = np.load(input_path).astype(float32)
    # Assume it is an image format like .png
    else:
        # Load image
        raw_image = Image.open(input_path).convert("L")
        image = np.asarray(raw_image).astype(float32) / 255
    return image


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyarcfire",
        description="Python port of SpArcFiRe, a program that finds and reports spiral features in images.",
    )

    base_subparser = argparse.ArgumentParser(add_help=False)
    base_subparser.add_argument(
        "-debug",
        "--debug",
        action="store_true",
        dest="debug_flag",
        help="Turns on debug statements.",
    )

    subparsers = parser.add_subparsers(dest="command")
    from_image_parser = subparsers.add_parser(
        "image",
        help="Process an image.",
        parents=(base_subparser,),
    )
    _configure_image_command_parser(from_image_parser)
    return parser.parse_args(args)


def _configure_image_command_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--i",
        type=str,
        dest="input_path",
        help="Path to the input image.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--o",
        type=str,
        dest="output_path",
        help="Path to save plot to. If this argument is not given, the plot will be shown in a GUI instead.",
        required=False,
    )


if __name__ == "__main__":
    import sys

    setup_logging()
    main(sys.argv[1:])
