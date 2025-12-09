"""
FUSION Domain Model Package.

This package contains the typed domain objects for FUSION v5:
- SimulationConfig: Immutable simulation configuration
- Request: Network service request with lifecycle tracking
- Lightpath: Allocated optical path with capacity management
- Result objects: Immutable pipeline stage outputs

All domain objects support legacy conversion via from_legacy_dict/to_legacy_dict.
"""

from fusion.domain.config import SimulationConfig

__all__ = [
    "SimulationConfig",
]
