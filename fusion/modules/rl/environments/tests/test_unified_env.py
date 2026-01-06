"""Tests for UnifiedSimEnv.

Phase: P4.2 - UnifiedSimEnv Wiring
Chunk 6: Environment skeleton tests

These tests verify that:
1. Environment can be instantiated
2. Observation and action spaces are valid Gymnasium spaces
3. Spaces have correct shapes based on configuration
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from fusion.modules.rl.adapter import RLConfig
from fusion.modules.rl.environments import UnifiedSimEnv


class TestUnifiedSimEnvInit:
    """Tests for UnifiedSimEnv initialization."""

    def test_init_with_default_config(self) -> None:
        """Environment initializes with default config."""
        env = UnifiedSimEnv()
        assert env is not None
        assert env.config is not None

    def test_init_with_custom_config(self) -> None:
        """Environment initializes with custom config."""
        config = RLConfig(k_paths=5, num_nodes=20, total_slots=400)
        env = UnifiedSimEnv(config=config)

        assert env.config.k_paths == 5
        assert env.config.num_nodes == 20
        assert env.config.total_slots == 400

    def test_init_with_render_mode(self) -> None:
        """Environment accepts render_mode parameter."""
        env = UnifiedSimEnv(render_mode=None)
        assert env.render_mode is None


class TestUnifiedSimEnvObservationSpace:
    """Tests for observation space validity."""

    def test_observation_space_is_dict(self) -> None:
        """Observation space is a Gymnasium Dict space."""
        env = UnifiedSimEnv()
        assert isinstance(env.observation_space, spaces.Dict)

    def test_observation_space_has_expected_keys(self) -> None:
        """Observation space contains all expected keys."""
        env = UnifiedSimEnv()
        expected_keys = {
            "source",
            "destination",
            "holding_time",
            "slots_needed",
            "path_lengths",
            "congestion",
            "available_slots",
            "is_feasible",
        }
        assert set(env.observation_space.spaces.keys()) == expected_keys

    def test_observation_space_shapes_match_config(self) -> None:
        """Observation space shapes match configuration."""
        config = RLConfig(k_paths=5, num_nodes=20)
        env = UnifiedSimEnv(config=config)

        # Node-related spaces should have shape (num_nodes,)
        assert env.observation_space["source"].shape == (20,)
        assert env.observation_space["destination"].shape == (20,)

        # Path-related spaces should have shape (k_paths,)
        assert env.observation_space["slots_needed"].shape == (5,)
        assert env.observation_space["path_lengths"].shape == (5,)
        assert env.observation_space["congestion"].shape == (5,)
        assert env.observation_space["available_slots"].shape == (5,)
        assert env.observation_space["is_feasible"].shape == (5,)

        # Scalar space
        assert env.observation_space["holding_time"].shape == (1,)

    def test_observation_space_dtypes_are_float32(self) -> None:
        """All observation space components are float32."""
        env = UnifiedSimEnv()
        for key, space in env.observation_space.spaces.items():
            assert isinstance(space, spaces.Box), f"{key} is not a Box space"
            assert space.dtype == np.float32, f"{key} has wrong dtype"

    def test_observation_space_bounds_are_valid(self) -> None:
        """Observation space bounds are finite and reasonable."""
        env = UnifiedSimEnv()

        for key, space in env.observation_space.spaces.items():
            assert isinstance(space, spaces.Box)
            assert np.all(np.isfinite(space.low)), f"{key} has infinite low bound"
            assert np.all(np.isfinite(space.high)), f"{key} has infinite high bound"
            assert np.all(space.low <= space.high), f"{key} has low > high"


class TestUnifiedSimEnvActionSpace:
    """Tests for action space validity."""

    def test_action_space_is_discrete(self) -> None:
        """Action space is a Gymnasium Discrete space."""
        env = UnifiedSimEnv()
        assert isinstance(env.action_space, spaces.Discrete)

    def test_action_space_matches_k_paths(self) -> None:
        """Action space size matches k_paths config."""
        config = RLConfig(k_paths=5)
        env = UnifiedSimEnv(config=config)
        assert env.action_space.n == 5

    def test_action_space_default_k_paths(self) -> None:
        """Default action space size is 3 (default k_paths)."""
        env = UnifiedSimEnv()
        assert env.action_space.n == 3


class TestUnifiedSimEnvGymnasiumCompliance:
    """Tests for Gymnasium interface compliance."""

    def test_inherits_from_gym_env(self) -> None:
        """UnifiedSimEnv inherits from gym.Env."""
        env = UnifiedSimEnv()
        assert isinstance(env, gym.Env)

    def test_has_metadata(self) -> None:
        """Environment has metadata dict."""
        env = UnifiedSimEnv()
        assert hasattr(env, "metadata")
        assert isinstance(env.metadata, dict)

    def test_has_render_method(self) -> None:
        """Environment has render method."""
        env = UnifiedSimEnv()
        assert hasattr(env, "render")
        assert callable(env.render)

    def test_has_close_method(self) -> None:
        """Environment has close method."""
        env = UnifiedSimEnv()
        assert hasattr(env, "close")
        assert callable(env.close)

    def test_reset_returns_tuple(self) -> None:
        """reset() returns (observation, info) tuple."""
        env = UnifiedSimEnv()
        result = env.reset()

        assert isinstance(result, tuple)
        assert len(result) == 2

        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_step_returns_tuple(self) -> None:
        """step() returns (obs, reward, terminated, truncated, info) tuple."""
        env = UnifiedSimEnv()
        env.reset()
        result = env.step(0)

        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_info_contains_action_mask(self) -> None:
        """Info dict contains action_mask."""
        env = UnifiedSimEnv()
        _, info = env.reset()

        assert "action_mask" in info
        assert isinstance(info["action_mask"], np.ndarray)

    def test_action_mask_shape_matches_action_space(self) -> None:
        """Action mask shape matches action space size."""
        config = RLConfig(k_paths=5)
        env = UnifiedSimEnv(config=config)
        _, info = env.reset()

        assert info["action_mask"].shape == (5,)

    def test_action_mask_dtype_is_bool(self) -> None:
        """Action mask dtype is boolean."""
        env = UnifiedSimEnv()
        _, info = env.reset()

        assert info["action_mask"].dtype == np.bool_


class TestUnifiedSimEnvObservationValidity:
    """Tests that observations are valid samples from observation space."""

    def test_reset_observation_in_space(self) -> None:
        """Observation from reset() is contained in observation_space."""
        env = UnifiedSimEnv()
        obs, _ = env.reset()

        assert env.observation_space.contains(obs), (
            "Reset observation not in observation space"
        )

    def test_step_observation_in_space(self) -> None:
        """Observation from step() is contained in observation_space."""
        env = UnifiedSimEnv()
        env.reset()
        obs, _, _, _, _ = env.step(0)

        assert env.observation_space.contains(obs), (
            "Step observation not in observation space"
        )


class TestUnifiedSimEnvZeroObservation:
    """Tests for _zero_observation helper method."""

    def test_zero_observation_matches_config(self) -> None:
        """Zero observation has correct shapes for config."""
        config = RLConfig(k_paths=4, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        obs = env._zero_observation()

        assert obs["source"].shape == (10,)
        assert obs["destination"].shape == (10,)
        assert obs["slots_needed"].shape == (4,)
        assert obs["is_feasible"].shape == (4,)

    def test_zero_observation_values_are_correct(self) -> None:
        """Zero observation has expected default values."""
        env = UnifiedSimEnv()
        obs = env._zero_observation()

        # Most values should be zero
        assert np.all(obs["source"] == 0)
        assert np.all(obs["destination"] == 0)
        assert np.all(obs["holding_time"] == 0)
        assert np.all(obs["path_lengths"] == 0)
        assert np.all(obs["congestion"] == 0)
        assert np.all(obs["available_slots"] == 0)
        assert np.all(obs["is_feasible"] == 0)

        # slots_needed should be -1 (indicates no path)
        assert np.all(obs["slots_needed"] == -1.0)
