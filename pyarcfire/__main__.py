# Standard libraries
import argparse
import logging
from typing import Sequence

# External libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters, transform

# Internal libraries
from .arc import fit_spiral_to_image, log_spiral
from .cluster import generate_hac_tree
from .log_utils import setup_logging
from .similarity import generate_similarity_matrix
from .orientation import generate_orientation_fields

log = logging.getLogger(__name__)


IMAGE_SIZE: int = 256


def main(raw_args: Sequence[str]) -> None:
    args = _parse_args(raw_args)
    # Load image
    image = np.asarray(Image.open(args.input_path).convert("L"))

    UNSHARP_MASK_RADIUS: float = 25
    UNSHARP_MASK_AMOUNT: float = 6
    contrast_image = filters.unsharp_mask(
        image, radius=UNSHARP_MASK_RADIUS, amount=UNSHARP_MASK_AMOUNT
    )

    contrast_image = transform.resize(contrast_image, (IMAGE_SIZE, IMAGE_SIZE))
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
    cluster_sizes = np.array([cluster.size for cluster in clusters])
    cluster_bins = np.logspace(0, np.log10(max(cluster_sizes)), 10)
    num_rows = field.num_rows
    num_columns = field.num_columns
    log.debug(f"Cluster sizes = {cluster_sizes[:5]}")

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
    clusters = [cluster for cluster in clusters if cluster.size >= 150]
    color_map = mpl.colormaps["hsv"]
    for idx, cluster in enumerate(clusters):
        mask = cluster.get_mask(num_rows, num_columns)
        cluster_mask = np.zeros((contrast_image.shape[0], contrast_image.shape[1], 4))
        cluster_mask[mask, :] = color_map(idx / len(clusters))
        cluster_axis.imshow(cluster_mask)

    cluster_size_axis = fig.add_subplot(235)
    cluster_size_axis.set_title("Cluster size")
    cluster_size_axis.set_yscale("log")
    cluster_size_axis.hist(cluster_sizes, bins=cluster_bins)

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
