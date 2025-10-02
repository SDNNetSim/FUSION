"""Data adapters for handling different data format versions."""

from fusion.visualization.infrastructure.adapters.adapter_registry import (
    DataAdapterRegistry,
    get_adapter,
    get_default_registry,
    register_adapter,
)
from fusion.visualization.infrastructure.adapters.canonical_data import (
    CanonicalData,
    IterationData,
)
from fusion.visualization.infrastructure.adapters.data_adapter import DataAdapter
from fusion.visualization.infrastructure.adapters.v1_data_adapter import V1DataAdapter
from fusion.visualization.infrastructure.adapters.v2_data_adapter import V2DataAdapter

__all__ = [
    "DataAdapter",
    "CanonicalData",
    "IterationData",
    "V1DataAdapter",
    "V2DataAdapter",
    "DataAdapterRegistry",
    "get_default_registry",
    "register_adapter",
    "get_adapter",
]
