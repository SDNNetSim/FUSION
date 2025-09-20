"""
FUSION SNR Measurement Module.

This package contains various SNR (Signal-to-Noise Ratio) measurement algorithm
implementations for optical network quality assessment, including:

- Standard SNR measurement with linear and nonlinear noise modeling
- Multi-core fiber support with cross-talk calculations
- Amplified spontaneous emission (ASE) noise calculations
- Self-channel interference (SCI) and cross-channel interference (XCI) modeling

All algorithms implement the AbstractSNRMeasurer interface and can be accessed
through the SNRRegistry for dynamic algorithm selection.
"""

from . import utils
from .registry import (SNR_ALGORITHMS, SNRRegistry, create_snr_algorithm,
                       get_multicore_snr_algorithms, get_snr_algorithm,
                       get_snr_algorithm_info, list_snr_algorithms)
from .snr import StandardSNRMeasurer

__all__ = [
    # Registry functions
    "SNRRegistry",
    "create_snr_algorithm",
    "get_snr_algorithm",
    "list_snr_algorithms",
    "get_snr_algorithm_info",
    "get_multicore_snr_algorithms",
    "SNR_ALGORITHMS",
    # Algorithm classes
    "StandardSNRMeasurer",
    # Utility module
    "utils",
]
