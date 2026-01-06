"""RL Simulation Adapter package.

This package provides the adapter layer between RL environments and the
V4 simulation stack (SDNOrchestrator, pipelines).

Phase: P4.1 - RLSimulationAdapter Scaffolding

Key Components:
- PathOption: Dataclass for candidate path representation
- RLSimulationAdapter: Main adapter between RL envs and simulation
- DisasterState: Dataclass for disaster/survivability scenarios
- OfflinePolicyAdapter: Bridge for BC/IQL offline policies
- RLConfig: Configuration for RL-specific settings
"""

from fusion.modules.rl.adapter.path_option import (
    ActionMask,
    PathOption,
    PathOptionList,
    compute_action_mask,
)
from fusion.modules.rl.adapter.rl_adapter import (
    DisasterState,
    OfflinePolicyAdapter,
    RLConfig,
    RLSimulationAdapter,
    create_disaster_state_from_engine,
)

__all__ = [
    # PathOption and helpers
    "PathOption",
    "PathOptionList",
    "ActionMask",
    "compute_action_mask",
    # Configuration
    "RLConfig",
    # Core adapter
    "RLSimulationAdapter",
    # Disaster/survivability support
    "DisasterState",
    "create_disaster_state_from_engine",
    # Offline RL support
    "OfflinePolicyAdapter",
]
