"""
Legacy adapters for FUSION simulation.

This package provides adapter classes that wrap legacy implementations
to satisfy the new pipeline protocols, enabling gradual migration.

ADAPTERS: These classes are TEMPORARY MIGRATION LAYERS.
They will be replaced with clean implementations in Phase 4.

Adapters:
    RoutingAdapter - Wraps legacy Routing class
    SpectrumAdapter - Wraps legacy SpectrumAssignment class
    GroomingAdapter - Wraps legacy Grooming class
    SNRAdapter - Wraps legacy SnrMeasurements class

Phase: P2.4 - Legacy Adapters
"""

from fusion.core.adapters.grooming_adapter import GroomingAdapter
from fusion.core.adapters.routing_adapter import RoutingAdapter
from fusion.core.adapters.snr_adapter import SNRAdapter
from fusion.core.adapters.spectrum_adapter import SpectrumAdapter

__all__ = [
    "RoutingAdapter",
    "SpectrumAdapter",
    "GroomingAdapter",
    "SNRAdapter",
]
