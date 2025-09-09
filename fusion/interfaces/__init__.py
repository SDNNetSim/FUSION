"""
Interfaces module for FUSION simulator.

This module contains abstract base classes that define the contracts
for all pluggable components in the FUSION architecture.
"""

from .router import AbstractRoutingAlgorithm
from .spectrum import AbstractSpectrumAssigner
from .snr import AbstractSNRMeasurer
from .agent import AgentInterface

__all__ = [
    'AbstractRoutingAlgorithm',
    'AbstractSpectrumAssigner',
    'AbstractSNRMeasurer',
    'AgentInterface'
]
