"""Unified RL environment using V4 simulation stack.

This module provides UnifiedSimEnv, a Gymnasium-compatible environment
that uses the V4 simulation stack through RLSimulationAdapter. Unlike
the legacy GeneralSimEnv, this environment uses the SAME pipelines
as non-RL simulation, eliminating duplicated logic.

Key Invariants:
- Uses RLSimulationAdapter for all simulation interactions
- Same pipelines as SDNOrchestrator (no forked simulator)
- Action mask in info["action_mask"] for SB3 MaskablePPO

Phase: P4.2 - UnifiedSimEnv Wiring
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fusion.modules.rl.adapter import RLConfig

if TYPE_CHECKING:
    from fusion.modules.rl.adapter import RLSimulationAdapter


class UnifiedSimEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium environment for optical network RSA using V4 simulation.

    This environment provides RL agents with access to the optical network
    routing and spectrum assignment problem. It uses the V4 simulation stack
    through RLSimulationAdapter, ensuring identical behavior between RL and
    non-RL simulations.

    Key Features:
    - Uses same pipelines as non-RL simulation (no code duplication)
    - Action masking via info["action_mask"] for SB3 MaskablePPO
    - Deterministic seeding for reproducibility
    - Gymnasium-compliant interface

    Observation Space:
        Dict with:
        - source: One-hot encoded source node (num_nodes,)
        - destination: One-hot encoded destination node (num_nodes,)
        - holding_time: Normalized request holding time (1,)
        - slots_needed: Spectrum slots needed per path (k_paths,)
        - path_lengths: Hop count per path (k_paths,)
        - congestion: Congestion metric per path (k_paths,)
        - available_slots: Available spectrum ratio per path (k_paths,)
        - is_feasible: Binary feasibility per path (k_paths,)

    Action Space:
        Discrete(k_paths) - select which candidate path to use

    Example:
        config = RLConfig(k_paths=3, num_nodes=14)
        env = UnifiedSimEnv(config)

        obs, info = env.reset(seed=42)
        mask = info["action_mask"]

        # Select valid action using mask
        valid_actions = np.where(mask)[0]
        action = valid_actions[0] if len(valid_actions) > 0 else 0

        obs, reward, terminated, truncated, info = env.step(action)

    Attributes:
        config: RLConfig with environment parameters
        observation_space: Gymnasium Dict space
        action_space: Gymnasium Discrete space
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: RLConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            config: RL configuration with environment parameters.
                Uses defaults if None.
            render_mode: Rendering mode (not used, for Gymnasium compatibility)
        """
        super().__init__()

        self._config = config or RLConfig()
        self.render_mode = render_mode

        # Initialize observation and action spaces
        self._setup_spaces()

        # These will be initialized during reset
        # Placeholder for adapter - will be set when wiring to simulation
        self._adapter: RLSimulationAdapter | None = None

    def _setup_spaces(self) -> None:
        """Initialize observation and action spaces based on config."""
        num_nodes = self._config.num_nodes
        k_paths = self._config.k_paths
        max_slots = self._config.total_slots

        # Observation space: Dict of Box spaces
        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "source": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(num_nodes,),
                    dtype=np.float32,
                ),
                "destination": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(num_nodes,),
                    dtype=np.float32,
                ),
                "holding_time": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "slots_needed": spaces.Box(
                    low=-1.0,
                    high=float(max_slots),
                    shape=(k_paths,),
                    dtype=np.float32,
                ),
                "path_lengths": spaces.Box(
                    low=0.0,
                    high=float(num_nodes),
                    shape=(k_paths,),
                    dtype=np.float32,
                ),
                "congestion": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(k_paths,),
                    dtype=np.float32,
                ),
                "available_slots": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(k_paths,),
                    dtype=np.float32,
                ),
                "is_feasible": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(k_paths,),
                    dtype=np.float32,
                ),
            }
        )

        # Action space: Discrete - select one of k paths
        self.action_space: spaces.Discrete = spaces.Discrete(k_paths)

    @property
    def config(self) -> RLConfig:
        """Access to RL configuration."""
        return self._config

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation dict
            info: Info dict with action_mask
        """
        super().reset(seed=seed)

        # TODO: Chunk 7 - Full reset implementation with simulation wiring
        # For now, return dummy observation and info

        obs = self._zero_observation()
        info = self._build_info()

        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute action and advance simulation.

        Args:
            action: Selected path index (0 to k_paths-1)

        Returns:
            observation: New observation dict
            reward: Scalar reward
            terminated: True if episode ended normally
            truncated: True if episode ended abnormally
            info: Info dict with action_mask
        """
        # TODO: Chunk 8 - Full step implementation with simulation wiring
        # For now, return dummy values

        obs = self._zero_observation()
        reward = 0.0
        terminated = True
        truncated = False
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def _build_info(self) -> dict[str, Any]:
        """Build info dict with action mask.

        Returns:
            Info dict with action_mask key
        """
        k = self._config.k_paths
        # All actions masked (none feasible) for skeleton
        mask = np.zeros(k, dtype=np.bool_)

        return {
            "action_mask": mask,
        }

    def _zero_observation(self) -> dict[str, np.ndarray]:
        """Return zero-filled observation for skeleton/terminal state."""
        k = self._config.k_paths
        n = self._config.num_nodes

        return {
            "source": np.zeros(n, dtype=np.float32),
            "destination": np.zeros(n, dtype=np.float32),
            "holding_time": np.zeros(1, dtype=np.float32),
            "slots_needed": np.full(k, -1.0, dtype=np.float32),
            "path_lengths": np.zeros(k, dtype=np.float32),
            "congestion": np.zeros(k, dtype=np.float32),
            "available_slots": np.zeros(k, dtype=np.float32),
            "is_feasible": np.zeros(k, dtype=np.float32),
        }

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass
