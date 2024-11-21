"""Pyarcfire is a port of the SpArcFiRe software, a spiral arc finder based on image processing.

Please see [SpArcFiRe](https://github.com/waynebhayes/SpArcFiRe) for SpArcFiRe itself, and
the corresponding [paper](https://ui.adsabs.harvard.edu/abs/2014ApJ...790...87D/abstract).
"""

import importlib.metadata

from . import preprocess
from .arc import LogSpiralFitResult, fit_spiral_to_image, log_spiral
from .cluster import GenerateClustersSettings, generate_clusters
from .finder import SpiralFinder, SpiralFinderResult
from .merge_fit import MergeClustersByFitSettings, merge_clusters_by_fit
from .orientation import GenerateOrientationFieldSettings, generate_orientation_fields
from .similarity import GenerateSimilarityMatrixSettings, generate_similarity_matrix

__version__ = importlib.metadata.version(__name__)


__all__ = [
    "fit_spiral_to_image",
    "generate_clusters",
    "GenerateClustersSettings",
    "generate_orientation_fields",
    "GenerateOrientationFieldSettings",
    "generate_similarity_matrix",
    "GenerateSimilarityMatrixSettings",
    "log_spiral",
    "LogSpiralFitResult",
    "merge_clusters_by_fit",
    "MergeClustersByFitSettings",
    "preprocess",
    "SpiralFinder",
    "SpiralFinderResult",
]
