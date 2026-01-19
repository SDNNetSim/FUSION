"""
fusion.core: Core simulation components and data structures.

This package provides the fundamental building blocks for optical network simulation:
- Simulation engine and environment management (legacy and orchestrator-based)
- Request generation and processing
- Network metrics and statistics collection
- Core data properties and structures
- Routing, spectrum assignment, and SNR measurement components
- Pipeline-based orchestration (v6.0+)
"""

from .grooming import Grooming
from .metrics import SimStats
from .ml_metrics import MLMetricsCollector

# Orchestrator components (v6.0+)
from .orchestrator import SDNOrchestrator
from .persistence import StatsPersistence
from .pipeline_factory import PipelineFactory, PipelineSet
from .properties import (
    SNAP_KEYS_LIST,
    RoutingProps,
    SDNProps,
    SNRProps,
    SpectrumProps,
    StatsProps,
)
from .routing import Routing
from .sdn_controller import SDNController
from .simulation import SimulationEngine
from .snr_measurements import SnrMeasurements
from .spectrum_assignment import SpectrumAssignment

# Public API - explicitly define what's exported
__all__ = [
    # Legacy simulation components
    "SimulationEngine",
    "SDNController",
    "Routing",
    "SpectrumAssignment",
    "SnrMeasurements",
    "Grooming",
    # Orchestrator components (v6.0+)
    "SDNOrchestrator",
    "PipelineFactory",
    "PipelineSet",
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
