"""
Algorithm registry configuration for reinforcement learning implementations.

This module defines the registry mapping that associates algorithm names
with their setup functions and implementation classes.
"""
from collections.abc import Callable
from typing import Any

# Type alias for clarity
AlgorithmConfig = dict[str, Callable[..., Any] | None]

# Lazy algorithm registry to avoid circular imports
_algorithm_registry: dict[str, AlgorithmConfig] | None = None


def get_algorithm_registry() -> dict[str, AlgorithmConfig]:
    """
    Get the algorithm registry with lazy initialization to avoid circular imports.

    :return: Dictionary mapping algorithm names to their configurations
    :rtype: dict[str, AlgorithmConfig]
    """
    global _algorithm_registry
    if _algorithm_registry is None:
        # Import algorithm classes (lazy import to avoid circular dependencies)
        from fusion.modules.rl.algorithms.a2c import A2C
        from fusion.modules.rl.algorithms.dqn import DQN
        from fusion.modules.rl.algorithms.ppo import PPO
        from fusion.modules.rl.algorithms.qr_dqn import QrDQN

        # Import setup functions
        from fusion.modules.rl.utils.setup import (
            setup_a2c,
            setup_dqn,
            setup_ppo,
            setup_qr_dqn,
        )

        _algorithm_registry = {
            "a2c": {
                "class": A2C,
                "load": None,  # TODO: Implement model loading functionality
                "setup": setup_a2c,
            },
            "dqn": {
                "class": DQN,
                "load": None,  # TODO: Implement model loading functionality
                "setup": setup_dqn,
            },
            "ppo": {
                "class": PPO,
                "load": None,  # TODO: Implement model loading functionality
                "setup": setup_ppo,
            },
            "qr_dqn": {
                "class": QrDQN,
                "load": None,  # TODO: Implement model loading functionality
                "setup": setup_qr_dqn,
            },
        }
    return _algorithm_registry


# For backward compatibility, provide the ALGORITHM_REGISTRY as a property
def __getattr__(name: str) -> Any:
    """Handle module-level attribute access for backward compatibility."""
    if name == "ALGORITHM_REGISTRY":
        return get_algorithm_registry()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
