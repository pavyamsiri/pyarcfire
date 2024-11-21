"""Finds spiral arcs in images of galaxies or anything with a spiral structure."""

from __future__ import annotations

import argparse
import colorsys
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from pyarcfire.finder import SpiralFinder, SpiralFinderResult

from .log_utils import setup_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

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
    output_path: Path | None = Path(args.output_path) if args.output_path is not None else None

    loaded_result = SpiralFinderResult.load(input_path)
    finder = SpiralFinder()

    result: SpiralFinderResult
    if loaded_result is None:
        image: npt.NDArray[np.float64] = _load_image(input_path)
        result = finder.extract(image)
    else:
        result = loaded_result

    if output_path is not None and output_path.suffix != ".png":
        result.dump(output_path)
        log.info(
            "[yellow]FILESYST[/yellow]: Saved result to [yellow]%s[/yellow]",
            output_path.absolute(),
        )
        return

    cluster_arrays = result.mask

    image = result.original_image
    contrast_image = result.processed_image
    field = result.orientation_field

    booster = finder.booster

    width: float = result.original_image_width / 2 - 0.5
    height: float = result.original_image_height / 2 - 0.5
    processed_width: float = result.processed_image_width / 2 - 0.5
    processed_height: float = result.processed_image_height / 2 - 0.5
    num_horizontal_pixels: int = result.processed_image_width
    num_vertical_pixels: int = result.processed_image_height
    log.debug("Dominant chirality %s", result.get_dominant_chirality())
    log.debug("Overall pitch angle %.2f degrees", np.rad2deg(result.get_overall_pitch_angle()))

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
        extent=(-processed_width, processed_width, -processed_height, processed_height),
        cmap="gray",
    )
    contrast_axis.set_title(f"Preprocessed image\nboosted with {booster}" if booster is not None else "Preprocessed Image")
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
    num_clusters: int = result.num_clusters
    colored_image = np.zeros((contrast_image.shape[0], contrast_image.shape[1], 4))
    colored_image[:, :, 0] = contrast_image
    colored_image[:, :, 1] = contrast_image
    colored_image[:, :, 2] = contrast_image
    colored_image[:, :, 3] = 1.0
    for cluster_index in range(num_clusters):
        current_array = np.where(cluster_arrays == (cluster_index + 1), contrast_image, 0)
        mask = current_array > 0
        cluster_mask = np.zeros((current_array.shape[0], current_array.shape[1], 4))
        cluster_color = color_map((cluster_index + 0.5) / num_clusters)
        arc_color = get_complementary_color(cluster_color)
        cluster_mask[mask, :] = cluster_color
        colored_image[mask, :] *= cluster_color
        cluster_axis.imshow(
            cluster_mask,
            extent=(-width, width, -height, height),
        )
        spiral_fit = result.get_fit(cluster_index)
        x, y = spiral_fit.calculate_cartesian_coordinates(100, pixel_to_distance=1, flip_y=False)
        cluster_axis.plot(
            x,
            y,
            color=arc_color,
            label=f"Cluster {cluster_index}",
        )
        image_overlay_axis.plot(
            x,
            y,
            color=arc_color,
            label=f"Cluster {cluster_index}",
        )
        colored_image_overlay_axis.plot(
            x,
            y,
            color=arc_color,
            label=f"Cluster {cluster_index}",
        )
    colored_image_overlay_axis.imshow(
        colored_image,
        extent=(-width, width, -height, height),
    )

    fig.tight_layout()
    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path)
        log.info("[yellow]FILESYST[/yellow]: Saved plot to [yellow]%s[/yellow]", output_path.absolute())
    plt.close()


def _load_image(input_path: Path) -> npt.NDArray[np.float64]:
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
    image : ArrayND[S, f64]
        The image in row-major form.

    """
    image: npt.NDArray[np.float64]
    extension = Path(input_path).suffix.lstrip(".")
    # Numpy arrays from npy are already in row major form
    if extension == "npy":
        image = np.load(input_path).astype(np.float64)
    # Assume it is an image format like .png
    else:
        # Load image
        raw_image = Image.open(input_path).convert("L")
        image = (np.asarray(raw_image) / 255).astype(np.float64)
    return image


def get_complementary_color(rgba: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Calculate the complementary color using hue rotation.

    Parameters
    ----------
    rgba : tuple[float, float, float, float]
        The base color as a tuple of RGBA values normalized to the range [0, 1].
        rgb_color (tuple): The base color as an RGB tuple, with values in [0, 1].

    Returns
    -------
    complement : tuple[float, float, float, float]
        The complementary color in normalized RGBA.

    """
    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(rgba[0], rgba[1], rgba[2])
    # Rotate the hue by 180 degrees (0.5 in normalized hue space)
    complementary_hue = (h + 0.5) % 1.0
    # Convert back to RGB
    complementary_rgba = colorsys.hsv_to_rgb(complementary_hue, s, v)
    return (complementary_rgba[0], complementary_rgba[1], complementary_rgba[2], rgba[3])


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
