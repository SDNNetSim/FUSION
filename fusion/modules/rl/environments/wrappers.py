"""Environment wrappers for RL training compatibility.

This module provides wrappers that adapt UnifiedSimEnv for use with
various RL libraries, particularly Stable-Baselines3's MaskablePPO.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class ActionMaskWrapper(gym.Wrapper[dict[str, np.ndarray], int, dict[str, np.ndarray], int]):
    """Wrapper that exposes action_masks() method for SB3 MaskablePPO.

    Stable-Baselines3's MaskablePPO expects environments to have an
    action_masks() method that returns the current action mask. This
    wrapper extracts the action mask from info["action_mask"] and
    exposes it via the action_masks() method.

    Example:
        from stable_baselines3.common.maskable.policies import MaskableActorCriticPolicy
        from sb3_contrib import MaskablePPO

        env = UnifiedSimEnv(config)
        env = ActionMaskWrapper(env)

        model = MaskablePPO(MaskableActorCriticPolicy, env)
        model.learn(total_timesteps=10000)

    Note:
        The wrapped environment must return info["action_mask"] from
        both reset() and step() methods.
    """

    def __init__(self, env: gym.Env[dict[str, np.ndarray], int]) -> None:
        """Initialize the wrapper.

        Args:
            env: Environment to wrap. Must return info["action_mask"]
                from reset() and step().
        """
        super().__init__(env)
        self._current_mask: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment and update action mask.

        Args:
            seed: Random seed for reproducibility
            options: Additional options passed to wrapped env

        Returns:
            observation: Initial observation
            info: Info dict (includes action_mask)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_mask = info.get("action_mask")
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step and update action mask.

        Args:
            action: Action to take

        Returns:
            observation: New observation
            reward: Step reward
            terminated: Whether episode ended normally
            truncated: Whether episode was truncated
            info: Info dict (includes action_mask)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_mask = info.get("action_mask")
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the current action mask.

        This method is required by SB3's MaskablePPO. It returns a
        boolean array where True indicates valid actions.

        Returns:
            Boolean array of shape (n_actions,) indicating valid actions.

        Raises:
            RuntimeError: If called before reset() or if env doesn't
                provide action_mask in info.
        """
        if self._current_mask is None:
            raise RuntimeError(
                "No action mask available. Call reset() first, or ensure "
                "the wrapped environment returns info['action_mask']."
            )
        return self._current_mask
