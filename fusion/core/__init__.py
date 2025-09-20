"""
fusion.core: Core simulation components and data structures.

This package provides the fundamental building blocks for optical network simulation:
- Simulation engine and environment management
- Request generation and processing
- Network metrics and statistics collection
- Core data properties and structures
- Routing, spectrum assignment, and SNR measurement components
"""

from .properties import (SNAP_KEYS_LIST, RoutingProps, SDNProps, SNRProps,
                         SpectrumProps, StatsProps)

__all__ = [
    "RoutingProps",
    "SpectrumProps",
    "SNRProps",
    "SDNProps",
    "StatsProps",
    "SNAP_KEYS_LIST",
]
