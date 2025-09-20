"""
FUSION Spectrum Assignment Module.

This package contains various spectrum assignment algorithm implementations for
optical network resource allocation, including:

- First Fit spectrum assignment
- Best Fit spectrum assignment (minimizes fragmentation)
- Last Fit spectrum assignment
- Light path slicing management for segmented requests
- Multi-band spectrum assignment support

All algorithms implement the AbstractSpectrumAssigner interface and can be accessed
through the SpectrumRegistry for dynamic algorithm selection.
"""

from .best_fit import BestFitSpectrum
from .first_fit import FirstFitSpectrum
from .last_fit import LastFitSpectrum
from .light_path_slicing import LightPathSlicingManager
from .registry import (SPECTRUM_ALGORITHMS, SpectrumRegistry,
                       create_spectrum_algorithm,
                       get_multiband_spectrum_algorithms,
                       get_spectrum_algorithm, get_spectrum_algorithm_info,
                       list_spectrum_algorithms)
from .utils import SpectrumHelpers

__all__ = [
    # Registry functions
    "SpectrumRegistry",
    "create_spectrum_algorithm",
    "get_spectrum_algorithm",
    "list_spectrum_algorithms",
    "get_spectrum_algorithm_info",
    "get_multiband_spectrum_algorithms",
    "SPECTRUM_ALGORITHMS",
    # Algorithm classes
    "FirstFitSpectrum",
    "BestFitSpectrum",
    "LastFitSpectrum",
    # Utility classes
    "LightPathSlicingManager",
    "SpectrumHelpers",
]
