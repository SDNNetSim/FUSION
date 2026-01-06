"""RL environment implementations using V4 simulation stack.

This package provides Gymnasium-compatible environments that use the
V4 simulation stack (SDNOrchestrator, pipelines) through RLSimulationAdapter.

Key Components:
- UnifiedSimEnv: Main environment using unified simulation stack
- ActionMaskWrapper: Wrapper for SB3 MaskablePPO compatibility

Phase: P4.2 - UnifiedSimEnv Wiring
"""

from fusion.modules.rl.environments.unified_env import UnifiedSimEnv
from fusion.modules.rl.environments.wrappers import ActionMaskWrapper

__all__ = [
    "UnifiedSimEnv",
    "ActionMaskWrapper",
]
