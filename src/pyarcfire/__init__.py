"""Pyarcfire is a port of the SpArcFiRe software, a spiral arc finder based on image processing.

Please see [SpArcFiRe](https://github.com/waynebhayes/SpArcFiRe) for SpArcFiRe itself, and
the corresponding [paper](https://ui.adsabs.harvard.edu/abs/2014ApJ...790...87D/abstract).
"""

import importlib.metadata

from .arc import LogSpiralFitResult, fit_spiral_to_image, log_spiral
from .cluster import GenerateClustersSettings, generate_clusters
from .merge_fit import MergeClustersByFitSettings, merge_clusters_by_fit
from .orientation import GenerateOrientationFieldSettings, generate_orientation_fields
from .similarity import GenerateSimilarityMatrixSettings, generate_similarity_matrix
from .spiral import ClusterSpiralResult, UnsharpMaskSettings, detect_spirals_in_image

__version__ = importlib.metadata.version(__name__)


__all__ = [
    "ClusterSpiralResult",
    "GenerateClustersSettings",
    "GenerateOrientationFieldSettings",
    "GenerateSimilarityMatrixSettings",
    "LogSpiralFitResult",
    "MergeClustersByFitSettings",
    "UnsharpMaskSettings",
    "detect_spirals_in_image",
    "fit_spiral_to_image",
    "generate_clusters",
    "generate_orientation_fields",
    "generate_similarity_matrix",
    "log_spiral",
    "merge_clusters_by_fit",
]
