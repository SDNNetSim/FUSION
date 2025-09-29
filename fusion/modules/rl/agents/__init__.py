"""
RL Agents Module.

Provides reinforcement learning agents for path selection, core assignment,
and spectrum allocation in the FUSION network simulation framework.
"""

# Public API exports
from .base_agent import BaseAgent
from .core_agent import CoreAgent
from .path_agent import PathAgent
from .spectrum_agent import SpectrumAgent

# Version info
__version__ = "1.0.0"

# Public API - explicitly define what's exported
__all__ = [
    "BaseAgent",
    "CoreAgent",
    "PathAgent",
    "SpectrumAgent",
]
