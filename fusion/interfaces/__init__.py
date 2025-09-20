"""
Interfaces module for FUSION simulator.

This module contains abstract base classes that define the contracts
for all pluggable components in the FUSION architecture.
"""

from .agent import AgentInterface
from .router import AbstractRoutingAlgorithm
from .snr import AbstractSNRMeasurer
from .spectrum import AbstractSpectrumAssigner

__all__ = [
    "AbstractRoutingAlgorithm",
    "AbstractSpectrumAssigner",
    "AbstractSNRMeasurer",
    "AgentInterface",
]
