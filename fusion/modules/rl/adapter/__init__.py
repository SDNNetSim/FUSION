"""RL Simulation Adapter package.

This package provides the adapter layer between RL environments and the
V4 simulation stack (SDNOrchestrator, pipelines).

Phase: P4.1 - RLSimulationAdapter Scaffolding
"""

from fusion.modules.rl.adapter.path_option import (
    ActionMask,
    PathOption,
    PathOptionList,
    compute_action_mask,
)

__all__ = [
    "PathOption",
    "PathOptionList",
    "ActionMask",
    "compute_action_mask",
]
