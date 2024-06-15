"""Finds spiral arcs in images of galaxies or anything with a spiral structure."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from PIL import Image
from skimage import transform

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
    from typing import Any

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
    elif command == "cluster":
        process_cluster(args)
    else:
        log.critical("Command %s is unrecognised or not yet supported!", command)


def process_from_image(args: argparse.Namespace) -> None:
    """Preprocess an image and run it through the SpArcFiRe algorithm.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.

    """
    input_path: str = args.input_path
    image: NDArray[np.float32]
    extension = Path(input_path).suffix.lstrip(".")
    if extension == "npy":
        image = np.load(input_path, allow_pickle=True).astype(np.float32)
    else:
        # Load image
        image = np.asarray(Image.open(input_path).convert("L")).astype(np.float32) / 255
    # Rescale
    scaling_factor: float | None = args.scaling_factor
    if scaling_factor is not None:
        log.info("Rescaling image by factor %f...", scaling_factor)
        image = transform.rescale(image, scaling_factor)

    width: float = image.shape[0] / 2 - 0.5

    result = detect_spirals_in_image(
        image,
        UnsharpMaskSettings(),
        GenerateOrientationFieldSettings(),
        GenerateSimilarityMatrixSettings(),
        GenerateClustersSettings(),
        MergeClustersByFitSettings(),
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

    if args.cluster_path is not None:
        result.dump(args.cluster_path)

    show_flag: bool = args.output_path is None

    fig = plt.figure(figsize=(14, 8))
    original_axis = fig.add_subplot(231)
    original_axis.imshow(image, cmap="gray")
    original_axis.set_title("Original image")
    original_axis.set_axis_off()

    contrast_axis = fig.add_subplot(232)
    contrast_axis.imshow(contrast_image, cmap="gray")
    contrast_axis.set_title(
        rf"Unsharp image $\mathrm{{Radius}} = {unsharp_settings.radius}, \; \mathrm{{Amount}} = {unsharp_settings.amount}$",
    )
    contrast_axis.set_axis_off()

    x_space_range = np.arange(field.shape[0])
    y_space_range = np.arange(field.shape[1])
    x, y = np.meshgrid(x_space_range, -y_space_range)
    orientation_axis = fig.add_subplot(233)
    orientation_axis.quiver(x, y, field.x, field.y, color="tab:blue", headaxislength=0)
    orientation_axis.set_aspect("equal")
    orientation_axis.set_title("Orientation field")
    orientation_axis.set_axis_off()

    cluster_axis = fig.add_subplot(234)
    cluster_axis.set_title("Clusters")
    cluster_axis.set_xlim(-width, width)
    cluster_axis.set_ylim(-width, width)
    cluster_axis.set_axis_off()

    image_overlay_axis = fig.add_subplot(235)
    image_overlay_axis.imshow(image, extent=(-width, width, -width, width), cmap="gray")
    image_overlay_axis.set_title("Original image overlaid with spirals")
    image_overlay_axis.set_xlim(-width, width)
    image_overlay_axis.set_ylim(-width, width)
    image_overlay_axis.set_axis_off()

    colored_image_overlay_axis = fig.add_subplot(236)
    colored_image_overlay_axis.set_title(
        "Original image colored with masks and overlaid with spirals",
    )
    colored_image_overlay_axis.set_xlim(-width, width)
    colored_image_overlay_axis.set_ylim(-width, width)
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
        cluster_mask[mask, :] = color_map((cluster_idx + 0.5) / num_clusters)
        colored_image[mask, :] *= color_map((cluster_idx + 0.5) / num_clusters)
        cluster_axis.imshow(cluster_mask, extent=(-width, width, -width, width))
        spiral_fit = fit_spiral_to_image(current_array)
        x, y = spiral_fit.calculate_cartesian_coordinates(100, pixel_to_distance=1)
        cluster_axis.plot(
            x,
            y,
            color=color_map((num_clusters - cluster_idx + 0.5) / num_clusters),
            label=f"Cluster {cluster_idx}",
        )
        image_overlay_axis.plot(
            x,
            y,
            color=color_map((num_clusters - cluster_idx + 0.5) / num_clusters),
            label=f"Cluster {cluster_idx}",
        )
        colored_image_overlay_axis.plot(
            x,
            y,
            color=color_map((num_clusters - cluster_idx + 0.5) / num_clusters),
            label=f"Cluster {cluster_idx}",
        )
    colored_image_overlay_axis.imshow(
        colored_image,
        extent=(-width, width, -width, width),
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


def process_cluster(args: argparse.Namespace) -> None:
    """Load a cluster array and run it through the SpArcFiRe algorithm.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.

    """
    input_path: str = args.input_path
    extension = Path(input_path).suffix.lstrip(".")
    if extension == "npy":
        log.info("Loading npy...")
        arr = np.load(input_path)
    elif extension == "mat":
        log.info("Loading mat...")
        data: dict[str, Any] = scipy.io.loadmat(input_path)
        arr = data["image"]
        if len(arr.shape) == 2:
            arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
        assert len(arr.shape) == 3
    else:
        log.critical(
            "The %s data format is not valid or is not yet supported!",
            extension,
        )
        return
    num_clusters = arr.shape[2]
    log.debug("Loaded %d clusters", num_clusters)

    fit_spiral_to_image(arr.sum(axis=2))

    width = arr.shape[0] / 2 - 0.5

    for cluster_idx in range(num_clusters):
        log.debug("Cluster %d sums to = %f", cluster_idx, arr[:, :, cluster_idx].sum())

    if not args.plot_flag:
        return

    fig = plt.figure()
    axis = fig.add_subplot(111)
    color_map = mpl.colormaps["hsv"]
    for cluster_idx in range(num_clusters):
        current_array = arr[:, :, cluster_idx]
        mask = current_array > 0
        cluster_mask = np.zeros((current_array.shape[0], current_array.shape[1], 4))
        cluster_mask[mask, :] = color_map((cluster_idx + 0.5) / num_clusters)
        axis.imshow(cluster_mask, extent=(-width, width, -width, width))
        spiral_fit = fit_spiral_to_image(current_array)
        x, y = spiral_fit.calculate_cartesian_coordinates(100, pixel_to_distance=1)
        axis.plot(
            x,
            y,
            color=color_map((num_clusters - cluster_idx + 0.5) / num_clusters),
            label=f"Cluster {cluster_idx}",
        )
    axis.legend()
    axis.set_xlim(-width, width)
    axis.set_ylim(-width, width)

    plt.show()
    plt.close()


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
    from_cluster_parser = subparsers.add_parser(
        "cluster",
        help="Process a cluster stored in as a data array.",
        parents=(base_subparser,),
    )
    _configure_cluster_command_parser(from_cluster_parser)
    return parser.parse_args(args)


def _configure_image_command_parser(parser: argparse.ArgumentParser) -> None:
    __add_input_path_to_parser(parser)
    parser.add_argument(
        "-o",
        "--o",
        type=str,
        dest="output_path",
        help="Path to save plot to. If this argument is not given, the plot will be shown in a GUI instead.",
        required=False,
    )
    parser.add_argument(
        "-co",
        "--co",
        type=str,
        dest="cluster_path",
        help="Path to output data array of clusters.",
        required=False,
    )
    parser.add_argument(
        "-scale", "--scale", type=float, dest="scaling_factor", help="Give this flag a number to scale the image with."
    )


def _configure_cluster_command_parser(parser: argparse.ArgumentParser) -> None:
    __add_input_path_to_parser(parser)
    parser.add_argument(
        "-plot",
        "--plot",
        action="store_true",
        dest="plot_flag",
        help="Turn on plotting.",
        required=False,
    )


def __add_input_path_to_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--i",
        type=str,
        dest="input_path",
        help="Path to the input image.",
        required=True,
    )


if __name__ == "__main__":
    import sys

    setup_logging()
    main(sys.argv[1:])
