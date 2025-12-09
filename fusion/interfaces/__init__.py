"""
Interfaces module for FUSION simulator.

This module contains:
1. Abstract base classes that define contracts for legacy pluggable components
2. Protocol classes for v5 type-safe pipeline interfaces

Legacy Interfaces (Abstract Base Classes):
    - AbstractRoutingAlgorithm
    - AbstractSpectrumAssigner
    - AbstractSNRMeasurer
    - AgentInterface

V5 Pipeline Protocols (typing.Protocol):
    - RoutingPipeline
    - SpectrumPipeline
    - GroomingPipeline
    - SNRPipeline
    - SlicingPipeline
"""

from .agent import AgentInterface
from .factory import AlgorithmFactory, SimulationPipeline, create_simulation_pipeline
from .pipelines import (
    GroomingPipeline,
    RoutingPipeline,
    SlicingPipeline,
    SNRPipeline,
    SpectrumPipeline,
)
from .router import AbstractRoutingAlgorithm
from .snr import AbstractSNRMeasurer
from .spectrum import AbstractSpectrumAssigner

__all__ = [
    # Legacy abstract base classes
    "AbstractRoutingAlgorithm",
    "AbstractSpectrumAssigner",
    "AbstractSNRMeasurer",
    "AgentInterface",
    "AlgorithmFactory",
    "SimulationPipeline",
    "create_simulation_pipeline",
    # V5 pipeline protocols
    "RoutingPipeline",
    "SpectrumPipeline",
    "GroomingPipeline",
    "SNRPipeline",
    "SlicingPipeline",
]
