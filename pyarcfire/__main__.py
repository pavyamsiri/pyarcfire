# Standard libraries
import argparse
import logging
import os
from typing import Sequence

# External libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import scipy.io
from skimage import filters, transform

# Internal libraries
from .arc import fit_spiral_to_image, log_spiral
from .cluster import Cluster, generate_hac_tree
from .log_utils import setup_logging
from .similarity import generate_similarity_matrix
from .merge import calculate_arc_merge_error
from .merge_fit import merge_clusters_by_fit
from .orientation import generate_orientation_fields

log = logging.getLogger(__name__)


IMAGE_SIZE: int = 256


def main(raw_args: Sequence[str]) -> None:
    args = _parse_args(raw_args)

    match args.command:
        case "image":
            process_from_image(args)
        case "cluster":
            process_cluster(args)
        case _ as command:
            log.critical(f"Command {command} is unrecognised or not yet supported!")


def process_from_image(args: argparse.Namespace) -> None:
    # Load image
    image = np.asarray(Image.open(args.input_path).convert("L"))
    image = transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    width: float = image.shape[0] / 2 + 0.5

    UNSHARP_MASK_RADIUS: float = 25
    UNSHARP_MASK_AMOUNT: float = 6
    contrast_image = filters.unsharp_mask(
        image, radius=UNSHARP_MASK_RADIUS, amount=UNSHARP_MASK_AMOUNT
    )

    field = generate_orientation_fields(contrast_image)
    strengths = field.get_strengths()
    nonzero_cells = np.count_nonzero(strengths)
    total_cells = strengths.size
    log.info(
        f"Strengths has {nonzero_cells}/{total_cells} |{100*(nonzero_cells/total_cells):.2f}%| nonzero cells"
    )
    # Next steps
    # 1. Generate similarity matrix
    # 2. Generate HAC tree containing similar clusters/regions (Hierarchical Agglomerative Clustering)
    # 3. Fit spirals to clusters
    # 4. Merge clusters and arcs by considering compatible arcs
    # 5. Color arcs red for S-wise and cyan for Z-wise

    stop_threshold: float = 0.15
    matrix = generate_similarity_matrix(field, stop_threshold)
    clusters = generate_hac_tree(matrix.tocsr(), contrast_image, field, stop_threshold)  # type:ignore
    clusters = sorted(clusters, key=lambda x: x.size, reverse=True)
    cluster_sizes = np.array(
        [cluster.size for cluster in clusters if cluster.size > 150]
    )
    cluster_bins = np.logspace(0, np.log10(max(cluster_sizes)), 10)
    log.debug(f"Cluster sizes = {cluster_sizes}")

    cluster_arrays = Cluster.list_to_array(
        [cluster for cluster in clusters if cluster.size > 150], image
    )
    cluster_arrays = merge_clusters_by_fit(cluster_arrays)

    if args.cluster_path is not None:
        extension = os.path.splitext(args.cluster_path)[1].lstrip(".")
        match extension:
            case "npy":
                np.save(args.cluster_path, cluster_arrays)
                log.info(
                    f"Saved {cluster_arrays.shape[2]} clusters to [magenta]{args.cluster_path}[/magenta]"
                )
            case "mat":
                scipy.io.savemat(args.cluster_path, {"image": cluster_arrays})
                log.info(
                    f"Saved {cluster_arrays.shape[2]} clusters to [magenta]{args.cluster_path}[/magenta]"
                )
            case _:
                log.warn("Unsupported file extension for cluster data!")

    fig = plt.figure()
    original_axis = fig.add_subplot(231)
    original_axis.imshow(image, cmap="gray")
    original_axis.set_title("Original image")
    original_axis.set_axis_off()

    contrast_axis = fig.add_subplot(232)
    contrast_axis.imshow(contrast_image, cmap="gray")
    contrast_axis.set_title(
        rf"Unsharp image $\text{{Radius}} = {UNSHARP_MASK_RADIUS}, \; \text{{Amount}} = {UNSHARP_MASK_AMOUNT}$"
    )
    contrast_axis.set_axis_off()

    space_range = np.arange(field.shape[0])
    x, y = np.meshgrid(space_range, -space_range)
    orientation_axis = fig.add_subplot(233)
    orientation_axis.quiver(x, y, field.x, field.y, color="tab:blue", headaxislength=0)
    orientation_axis.set_aspect("equal")
    orientation_axis.set_title("Orientation field")
    orientation_axis.set_axis_off()

    cluster_axis = fig.add_subplot(234)
    cluster_axis.set_title("Clusters")
    color_map = mpl.colormaps["hsv"]
    num_clusters: int = cluster_arrays.shape[2]
    for cluster_idx in range(num_clusters):
        current_array = cluster_arrays[:, :, cluster_idx]
        mask = current_array > 0
        cluster_mask = np.zeros((current_array.shape[0], current_array.shape[1], 4))
        cluster_mask[mask, :] = color_map(cluster_idx / num_clusters)
        cluster_axis.imshow(cluster_mask, extent=(-width, width, -width, width))
        spiral_fit = fit_spiral_to_image(current_array)
        x, y = spiral_fit.calculate_cartesian_coordinates(100)
        cluster_axis.plot(x, y)

    cluster_size_axis = fig.add_subplot(235)
    cluster_size_axis.set_title("Cluster size")
    cluster_size_axis.set_yscale("log")
    cluster_size_axis.hist(cluster_sizes, bins=cluster_bins)

    fig.tight_layout()
    plt.show()
    plt.close()


def process_cluster(args: argparse.Namespace) -> None:
    input_path: str = args.input_path
    _, ext = os.path.splitext(input_path)
    match ext.lstrip("."):
        case "npy":
            log.info("Loading npy...")
            arr = np.load(input_path)
        case "mat":
            log.info("Loading mat...")
            data = scipy.io.loadmat(input_path)
            arr = data["image"]
            if len(arr.shape) == 2:
                arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
            assert len(arr.shape) == 3
        case _:
            log.critical(f"The {ext} data format is not valid or is not yet supported!")
            return
    num_clusters = arr.shape[2]
    log.debug(f"Loaded {num_clusters} clusters")

    width = arr.shape[0] / 2 + 0.5
    cluster_arrays = merge_clusters_by_fit(arr)

    fig = plt.figure()
    axis = fig.add_subplot(111)
    color_map = mpl.colormaps["hsv"]
    num_clusters: int = cluster_arrays.shape[2]
    for cluster_idx in range(num_clusters):
        current_array = arr[:, :, cluster_idx]
        mask = current_array > 0
        cluster_mask = np.zeros((current_array.shape[0], current_array.shape[1], 4))
        cluster_mask[mask, :] = color_map((cluster_idx + 0.5) / num_clusters)
        axis.imshow(cluster_mask, extent=(-width, width, -width, width))
        spiral_fit = fit_spiral_to_image(current_array)
        x, y = spiral_fit.calculate_cartesian_coordinates(100)
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
    subparsers = parser.add_subparsers(dest="command")
    from_image_parser = subparsers.add_parser("image", help="Process an image.")
    _configure_image_command_parser(from_image_parser)
    from_cluster_parser = subparsers.add_parser(
        "cluster", help="Process a cluster stored in as a data array."
    )
    __add_input_path_to_parser(from_cluster_parser)
    return parser.parse_args(args)


def _configure_image_command_parser(parser: argparse.ArgumentParser) -> None:
    __add_input_path_to_parser(parser)
    parser.add_argument(
        "-co",
        "--co",
        type=str,
        dest="cluster_path",
        help="Path to output data array of clusters.",
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
