"""Repository interfaces for data access abstraction."""

from fusion.visualization.domain.repositories.metadata_repository import (
    MetadataRepository,
)
from fusion.visualization.domain.repositories.simulation_repository import (
    SimulationRepository,
)

__all__ = [
    "SimulationRepository",
    "MetadataRepository",
]
