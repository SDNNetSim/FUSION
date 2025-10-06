"""
fusion.core: Core simulation components and data structures.

This package provides the fundamental building blocks for optical network simulation:
- Simulation engine and environment management
- Request generation and processing
- Network metrics and statistics collection
- Core data properties and structures
- Routing, spectrum assignment, and SNR measurement components
"""

# Core simulation components
# Metrics and persistence
from .metrics import SimStats
from .ml_metrics import MLMetricsCollector
from .persistence import StatsPersistence

# Core data properties
from .properties import (
    SNAP_KEYS_LIST,
    RoutingProps,
    SDNProps,
    SNRProps,
    SpectrumProps,
    StatsProps,
)
from .routing import Routing

# Import these after other core components to avoid circular imports
from .sdn_controller import SDNController
from .simulation import SimulationEngine
from .snr_measurements import SnrMeasurements
from .spectrum_assignment import SpectrumAssignment

# Public API - explicitly define what's exported
__all__ = [
    # Core simulation components
    "SimulationEngine",
    "SDNController",
    "Routing",
    "SpectrumAssignment",
    "SnrMeasurements",
    # Metrics and persistence
    "SimStats",
    "MLMetricsCollector",
    "StatsPersistence",
    # Core data properties
    "RoutingProps",
    "SpectrumProps",
    "SNRProps",
    "SDNProps",
    "StatsProps",
    "SNAP_KEYS_LIST",
]
