"""Tests for UnifiedSimEnv.

Phase: P4.2 - UnifiedSimEnv Wiring
Chunks 6-8: Environment skeleton, reset(), and step() tests

These tests verify that:
1. Environment can be instantiated
2. Observation and action spaces are valid Gymnasium spaces
3. Spaces have correct shapes based on configuration
4. reset() generates requests and returns valid observations
5. Seeding produces deterministic episodes
6. step() advances episode and returns correct rewards
7. Episode terminates after all requests processed
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from fusion.modules.rl.adapter import RLConfig
from fusion.modules.rl.environments import ActionMaskWrapper, UnifiedSimEnv


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
        """Observation space contains all expected keys for obs_8 (default)."""
        env = UnifiedSimEnv()
        # Default obs_8 includes all features
        expected_keys = {
            "source",
            "destination",
            "request_bandwidth",
            "holding_time",
            "slots_needed",
            "path_lengths",
            "paths_cong",
            "available_slots",
            "is_feasible",
        }
        assert set(env.observation_space.spaces.keys()) == expected_keys

    def test_observation_space_obs_1(self) -> None:
        """obs_1 only includes source and destination."""
        config = RLConfig(obs_space="obs_1")
        env = UnifiedSimEnv(config=config)
        expected_keys = {"source", "destination"}
        assert set(env.observation_space.spaces.keys()) == expected_keys

    def test_observation_space_obs_4(self) -> None:
        """obs_4 includes source, destination, request_bandwidth, holding_time."""
        config = RLConfig(obs_space="obs_4")
        env = UnifiedSimEnv(config=config)
        expected_keys = {"source", "destination", "request_bandwidth", "holding_time"}
        assert set(env.observation_space.spaces.keys()) == expected_keys

    def test_observation_space_shapes_match_config(self) -> None:
        """Observation space shapes match configuration."""
        config = RLConfig(k_paths=5, num_nodes=20, num_bandwidth_classes=4)
        env = UnifiedSimEnv(config=config)

        # Node-related spaces should have shape (num_nodes,)
        assert env.observation_space["source"].shape == (20,)
        assert env.observation_space["destination"].shape == (20,)

        # Bandwidth space should have shape (num_bandwidth_classes,)
        assert env.observation_space["request_bandwidth"].shape == (4,)

        # Path-related spaces should have shape (k_paths,)
        assert env.observation_space["slots_needed"].shape == (5,)
        assert env.observation_space["path_lengths"].shape == (5,)
        assert env.observation_space["paths_cong"].shape == (5,)
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
        assert np.all(obs["paths_cong"] == 0)
        assert np.all(obs["available_slots"] == 0)
        assert np.all(obs["is_feasible"] == 0)

        # slots_needed should be -1 (indicates no path)
        assert np.all(obs["slots_needed"] == -1.0)


class TestUnifiedSimEnvReset:
    """Tests for reset() method (Chunk 7)."""

    def test_reset_generates_requests(self) -> None:
        """reset() generates requests for the episode."""
        env = UnifiedSimEnv(num_requests=50)
        env.reset(seed=42)

        assert env.num_requests == 50
        assert env.current_request is not None

    def test_reset_initializes_request_index(self) -> None:
        """reset() initializes request index to 0."""
        env = UnifiedSimEnv()
        env.reset(seed=42)

        assert env.request_index == 0

    def test_reset_with_options_num_requests(self) -> None:
        """reset() accepts num_requests in options."""
        env = UnifiedSimEnv(num_requests=100)
        env.reset(seed=42, options={"num_requests": 25})

        assert env.num_requests == 25

    def test_reset_returns_observation_with_request_data(self) -> None:
        """reset() returns observation based on first request."""
        env = UnifiedSimEnv()
        obs, _ = env.reset(seed=42)

        # Source and destination should have exactly one 1.0 value (one-hot)
        assert np.sum(obs["source"]) == 1.0
        assert np.sum(obs["destination"]) == 1.0

        # Holding time should be in [0, 1]
        assert 0.0 <= obs["holding_time"][0] <= 1.0

    def test_reset_info_contains_metadata(self) -> None:
        """reset() info contains episode metadata."""
        env = UnifiedSimEnv(num_requests=50)
        _, info = env.reset(seed=42)

        assert "action_mask" in info
        assert "request_index" in info
        assert "total_requests" in info
        assert info["request_index"] == 0
        assert info["total_requests"] == 50

    def test_reset_action_mask_has_feasible_paths(self) -> None:
        """reset() action mask indicates some feasible paths."""
        env = UnifiedSimEnv()
        # Run multiple times to ensure at least one has feasible paths
        has_feasible = False
        for seed in range(10):
            _, info = env.reset(seed=seed)
            if np.any(info["action_mask"]):
                has_feasible = True
                break

        assert has_feasible, "No feasible paths found in 10 resets"


class TestUnifiedSimEnvSeeding:
    """Tests for deterministic seeding."""

    def test_same_seed_same_requests(self) -> None:
        """Same seed produces identical request sequences."""
        env1 = UnifiedSimEnv(num_requests=20)
        env2 = UnifiedSimEnv(num_requests=20)

        env1.reset(seed=123)
        env2.reset(seed=123)

        # Compare first request
        req1 = env1.current_request
        req2 = env2.current_request

        assert req1 is not None
        assert req2 is not None
        assert req1.source == req2.source
        assert req1.destination == req2.destination
        assert req1.bandwidth_gbps == req2.bandwidth_gbps

    def test_same_seed_same_observations(self) -> None:
        """Same seed produces identical observations."""
        env1 = UnifiedSimEnv()
        env2 = UnifiedSimEnv()

        obs1, _ = env1.reset(seed=456)
        obs2, _ = env2.reset(seed=456)

        for key in obs1:
            assert np.allclose(obs1[key], obs2[key]), f"Mismatch in {key}"

    def test_different_seeds_different_requests(self) -> None:
        """Different seeds produce different request sequences."""
        env = UnifiedSimEnv(num_requests=20)

        env.reset(seed=100)
        req1 = env.current_request

        env.reset(seed=200)
        req2 = env.current_request

        assert req1 is not None
        assert req2 is not None
        # Very unlikely to have same src, dst, and bandwidth with different seeds
        different = (
            req1.source != req2.source
            or req1.destination != req2.destination
            or req1.bandwidth_gbps != req2.bandwidth_gbps
        )
        assert different, "Different seeds should produce different requests"

    def test_reset_without_seed_uses_random(self) -> None:
        """reset() without seed still works (uses random)."""
        env = UnifiedSimEnv()
        obs, info = env.reset()

        assert env.observation_space.contains(obs)
        assert "action_mask" in info


class TestUnifiedSimEnvRequestGeneration:
    """Tests for internal request generation."""

    def test_requests_have_valid_source_destination(self) -> None:
        """Generated requests have valid source/destination nodes."""
        config = RLConfig(num_nodes=14)
        env = UnifiedSimEnv(config=config, num_requests=50)
        env.reset(seed=42)

        req = env.current_request
        assert req is not None
        # Source/destination are strings in SimpleRequest
        src = int(req.source)
        dst = int(req.destination)
        assert 0 <= src < 14
        assert 0 <= dst < 14
        assert src != dst

    def test_requests_have_valid_bandwidth(self) -> None:
        """Generated requests have valid bandwidth values."""
        env = UnifiedSimEnv(num_requests=100)
        env.reset(seed=42)

        req = env.current_request
        assert req is not None
        assert req.bandwidth_gbps in [10.0, 40.0, 100.0, 400.0]

    def test_requests_have_positive_holding_time(self) -> None:
        """Generated requests have positive holding time."""
        env = UnifiedSimEnv()
        env.reset(seed=42)

        req = env.current_request
        assert req is not None
        assert req.holding_time > 0

    def test_requests_have_increasing_arrival_times(self) -> None:
        """Generated requests have non-decreasing arrival times."""
        env = UnifiedSimEnv(num_requests=50)
        env.reset(seed=42)

        # Access internal request list
        prev_time = 0.0
        for req in env._requests:
            assert req.arrive_time >= prev_time
            prev_time = req.arrive_time


class TestUnifiedSimEnvObservationBuilding:
    """Tests for observation building from requests."""

    def test_source_one_hot_matches_request(self) -> None:
        """Source one-hot encoding matches request source."""
        config = RLConfig(num_nodes=14)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        req = env.current_request
        assert req is not None

        # Find the 1.0 in source array
        source_idx = np.argmax(obs["source"])
        # Source is stored as string in SimpleRequest
        assert source_idx == int(req.source)

    def test_destination_one_hot_matches_request(self) -> None:
        """Destination one-hot encoding matches request destination."""
        config = RLConfig(num_nodes=14)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        req = env.current_request
        assert req is not None

        # Find the 1.0 in destination array
        dest_idx = np.argmax(obs["destination"])
        # Destination is stored as string in SimpleRequest
        assert dest_idx == int(req.destination)

    def test_holding_time_is_normalized(self) -> None:
        """Holding time is normalized to [0, 1]."""
        config = RLConfig(max_holding_time=100.0)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert 0.0 <= obs["holding_time"][0] <= 1.0

    def test_path_features_have_correct_shape(self) -> None:
        """Path features have shape (k_paths,)."""
        config = RLConfig(k_paths=5)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert obs["slots_needed"].shape == (5,)
        assert obs["path_lengths"].shape == (5,)
        assert obs["paths_cong"].shape == (5,)
        assert obs["available_slots"].shape == (5,)
        assert obs["is_feasible"].shape == (5,)

    def test_slots_needed_are_positive(self) -> None:
        """Slots needed are positive values."""
        env = UnifiedSimEnv()
        obs, _ = env.reset(seed=42)

        # All slots_needed should be >= 1 (at least 1 slot needed)
        assert np.all(obs["slots_needed"] >= 1.0)

    def test_congestion_in_valid_range(self) -> None:
        """Congestion values (paths_cong) are in [0, 1]."""
        env = UnifiedSimEnv()
        obs, _ = env.reset(seed=42)

        assert np.all(obs["paths_cong"] >= 0.0)
        assert np.all(obs["paths_cong"] <= 1.0)

    def test_available_slots_in_valid_range(self) -> None:
        """Available slots ratios are in [0, 1]."""
        env = UnifiedSimEnv()
        obs, _ = env.reset(seed=42)

        assert np.all(obs["available_slots"] >= 0.0)
        assert np.all(obs["available_slots"] <= 1.0)


class TestUnifiedSimEnvStep:
    """Tests for step() method (Chunk 8)."""

    def test_step_returns_five_tuple(self) -> None:
        """step() returns (obs, reward, terminated, truncated, info)."""
        env = UnifiedSimEnv()
        env.reset(seed=42)
        result = env.step(0)

        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_returns_valid_observation(self) -> None:
        """step() returns observation in observation_space."""
        env = UnifiedSimEnv()
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(0)

        assert env.observation_space.contains(obs)

    def test_step_returns_float_reward(self) -> None:
        """step() returns numeric reward."""
        env = UnifiedSimEnv()
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(0)

        assert isinstance(reward, (int, float))

    def test_step_returns_bool_terminated(self) -> None:
        """step() returns boolean terminated flag."""
        env = UnifiedSimEnv()
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(0)

        assert isinstance(terminated, bool)

    def test_step_returns_bool_truncated(self) -> None:
        """step() returns boolean truncated flag (always False)."""
        env = UnifiedSimEnv()
        env.reset(seed=42)
        _, _, _, truncated, _ = env.step(0)

        assert isinstance(truncated, bool)
        assert truncated is False

    def test_step_returns_info_with_action_mask(self) -> None:
        """step() returns info dict with action_mask."""
        env = UnifiedSimEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)

        assert "action_mask" in info
        assert isinstance(info["action_mask"], np.ndarray)

    def test_step_advances_request_index(self) -> None:
        """step() advances request_index by 1."""
        env = UnifiedSimEnv()
        env.reset(seed=42)

        assert env.request_index == 0
        env.step(0)
        assert env.request_index == 1
        env.step(0)
        assert env.request_index == 2

    def test_step_updates_current_request(self) -> None:
        """step() updates current_request to next request."""
        env = UnifiedSimEnv(num_requests=10)
        env.reset(seed=42)

        first_request = env.current_request
        env.step(0)
        second_request = env.current_request

        assert first_request is not None
        assert second_request is not None
        assert first_request.request_id == 0
        assert second_request.request_id == 1

    def test_step_without_reset_raises_error(self) -> None:
        """step() before reset() raises RuntimeError."""
        env = UnifiedSimEnv()

        with pytest.raises(RuntimeError, match="Must call reset"):
            env.step(0)


class TestUnifiedSimEnvEpisodeTermination:
    """Tests for episode termination logic."""

    def test_terminated_after_all_requests(self) -> None:
        """Episode terminates after processing all requests."""
        env = UnifiedSimEnv(num_requests=5)
        env.reset(seed=42)

        for i in range(4):
            _, _, terminated, _, _ = env.step(0)
            assert not terminated, f"Terminated too early at step {i}"

        # Last step should terminate
        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_is_episode_done_property(self) -> None:
        """is_episode_done property reflects episode state."""
        env = UnifiedSimEnv(num_requests=3)
        env.reset(seed=42)

        assert not env.is_episode_done
        env.step(0)
        assert not env.is_episode_done
        env.step(0)
        assert not env.is_episode_done
        env.step(0)
        assert env.is_episode_done

    def test_current_request_none_after_termination(self) -> None:
        """current_request is None after episode terminates."""
        env = UnifiedSimEnv(num_requests=2)
        env.reset(seed=42)

        env.step(0)
        assert env.current_request is not None

        env.step(0)  # This terminates the episode
        assert env.current_request is None

    def test_step_after_termination_raises_error(self) -> None:
        """step() after episode termination raises RuntimeError."""
        env = UnifiedSimEnv(num_requests=2)
        env.reset(seed=42)

        env.step(0)
        env.step(0)  # Episode terminates

        with pytest.raises(RuntimeError, match="Episode has ended"):
            env.step(0)

    def test_reset_after_termination_starts_new_episode(self) -> None:
        """reset() after termination starts a fresh episode."""
        env = UnifiedSimEnv(num_requests=2)
        env.reset(seed=42)

        env.step(0)
        env.step(0)  # Episode terminates
        assert env.is_episode_done

        env.reset(seed=42)
        assert not env.is_episode_done
        assert env.request_index == 0
        assert env.current_request is not None


class TestUnifiedSimEnvReward:
    """Tests for reward computation."""

    def test_feasible_action_gives_positive_reward(self) -> None:
        """Selecting a feasible path gives positive reward."""
        config = RLConfig(rl_success_reward=1.0, rl_block_penalty=-1.0)
        env = UnifiedSimEnv(config=config)

        # Try multiple seeds to find one with a feasible action
        for seed in range(20):
            env.reset(seed=seed)
            mask = env._current_feasibility
            if mask is not None and np.any(mask):
                feasible_action = int(np.argmax(mask))
                _, reward, _, _, _ = env.step(feasible_action)
                assert reward == 1.0
                return

        pytest.skip("Could not find seed with feasible action")

    def test_infeasible_action_gives_negative_reward(self) -> None:
        """Selecting an infeasible path gives negative reward."""
        config = RLConfig(rl_success_reward=1.0, rl_block_penalty=-1.0)
        env = UnifiedSimEnv(config=config)

        # Try multiple seeds to find one with an infeasible action
        for seed in range(20):
            env.reset(seed=seed)
            mask = env._current_feasibility
            if mask is not None and not np.all(mask):
                infeasible_action = int(np.argmin(mask))
                if not mask[infeasible_action]:
                    _, reward, _, _, _ = env.step(infeasible_action)
                    assert reward == -1.0
                    return

        pytest.skip("Could not find seed with infeasible action")

    def test_reward_uses_config_values(self) -> None:
        """Reward uses configured success/penalty values."""
        config = RLConfig(rl_success_reward=5.0, rl_block_penalty=-2.5)
        env = UnifiedSimEnv(config=config)
        env.reset(seed=42)

        # Get the computed reward (either success or penalty)
        _, reward, _, _, _ = env.step(0)
        assert reward in [5.0, -2.5]

    def test_invalid_action_gives_penalty(self) -> None:
        """Invalid action index gives penalty reward."""
        config = RLConfig(k_paths=3, rl_block_penalty=-1.0)
        env = UnifiedSimEnv(config=config)
        env.reset(seed=42)

        # Action outside valid range
        _, reward, _, _, _ = env.step(10)
        assert reward == -1.0


class TestUnifiedSimEnvFullEpisode:
    """Tests for running complete episodes."""

    def test_can_run_full_episode(self) -> None:
        """Can run a complete episode from reset to termination."""
        env = UnifiedSimEnv(num_requests=20)
        obs, info = env.reset(seed=42)

        total_reward = 0.0
        steps = 0

        while True:
            # Select first feasible action, or action 0 if none
            mask = info["action_mask"]
            if np.any(mask):
                action = int(np.argmax(mask))
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps == 20
        assert terminated
        assert not truncated

    def test_multiple_episodes_independent(self) -> None:
        """Multiple episodes are independent with different seeds."""
        env = UnifiedSimEnv(num_requests=10)

        # Run episode 1
        env.reset(seed=100)
        rewards_1 = []
        for _ in range(10):
            _, reward, _, _, _ = env.step(0)
            rewards_1.append(reward)

        # Run episode 2 with different seed
        env.reset(seed=200)
        rewards_2 = []
        for _ in range(10):
            _, reward, _, _, _ = env.step(0)
            rewards_2.append(reward)

        # Rewards should differ (different random feasibility)
        assert rewards_1 != rewards_2

    def test_same_seed_same_episode(self) -> None:
        """Same seed produces identical episode trajectories."""
        env = UnifiedSimEnv(num_requests=10)

        # Run episode 1
        env.reset(seed=42)
        rewards_1 = []
        for _ in range(10):
            _, reward, _, _, _ = env.step(0)
            rewards_1.append(reward)

        # Run episode 2 with same seed
        env.reset(seed=42)
        rewards_2 = []
        for _ in range(10):
            _, reward, _, _, _ = env.step(0)
            rewards_2.append(reward)

        assert rewards_1 == rewards_2


class TestUnifiedSimEnvGymnasiumChecker:
    """Tests using gymnasium.utils.env_checker (Chunk 9).

    This is Integration Checkpoint 2 - verifies full Gymnasium compliance.
    """

    def test_passes_env_checker(self) -> None:
        """Environment passes gymnasium.utils.env_checker."""
        from gymnasium.utils.env_checker import check_env

        env = UnifiedSimEnv(num_requests=10)
        # check_env raises an exception if there are issues
        # It also prints warnings for non-critical issues
        check_env(env, skip_render_check=True)

    def test_passes_env_checker_with_custom_config(self) -> None:
        """Environment with custom config passes env_checker."""
        from gymnasium.utils.env_checker import check_env

        config = RLConfig(k_paths=5, num_nodes=20, total_slots=400)
        env = UnifiedSimEnv(config=config, num_requests=10)
        check_env(env, skip_render_check=True)


class TestActionMaskWrapper:
    """Tests for ActionMaskWrapper (Chunk 10)."""

    def test_wrapper_wraps_environment(self) -> None:
        """ActionMaskWrapper wraps UnifiedSimEnv."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        assert wrapped.env is env

    def test_wrapper_has_action_masks_method(self) -> None:
        """Wrapped environment has action_masks() method."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        assert hasattr(wrapped, "action_masks")
        assert callable(wrapped.action_masks)

    def test_action_masks_after_reset(self) -> None:
        """action_masks() returns valid mask after reset()."""
        config = RLConfig(k_paths=3)
        env = UnifiedSimEnv(config=config, num_requests=10)
        wrapped = ActionMaskWrapper(env)

        wrapped.reset(seed=42)
        mask = wrapped.action_masks()

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (3,)
        assert mask.dtype == np.bool_

    def test_action_masks_after_step(self) -> None:
        """action_masks() returns updated mask after step()."""
        config = RLConfig(k_paths=3)
        env = UnifiedSimEnv(config=config, num_requests=10)
        wrapped = ActionMaskWrapper(env)

        wrapped.reset(seed=42)
        wrapped.step(0)
        mask = wrapped.action_masks()

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (3,)

    def test_action_masks_before_reset_raises(self) -> None:
        """action_masks() before reset() raises RuntimeError."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        with pytest.raises(RuntimeError, match="No action mask available"):
            wrapped.action_masks()

    def test_wrapper_preserves_observation_space(self) -> None:
        """Wrapper preserves observation_space from wrapped env."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        assert wrapped.observation_space == env.observation_space

    def test_wrapper_preserves_action_space(self) -> None:
        """Wrapper preserves action_space from wrapped env."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        assert wrapped.action_space == env.action_space

    def test_wrapper_reset_returns_same_as_env(self) -> None:
        """Wrapper reset() returns same observation as wrapped env."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        obs, info = wrapped.reset(seed=42)

        assert env.observation_space.contains(obs)
        assert "action_mask" in info

    def test_wrapper_step_returns_same_as_env(self) -> None:
        """Wrapper step() returns same tuple structure as wrapped env."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        wrapped.reset(seed=42)
        obs, reward, terminated, truncated, info = wrapped.step(0)

        assert env.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "action_mask" in info

    def test_wrapper_mask_matches_info_mask(self) -> None:
        """action_masks() returns same mask as info['action_mask']."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        _, info = wrapped.reset(seed=42)
        mask_from_method = wrapped.action_masks()
        mask_from_info = info["action_mask"]

        assert np.array_equal(mask_from_method, mask_from_info)

    def test_wrapper_can_run_full_episode(self) -> None:
        """Wrapped environment can run a full episode."""
        env = UnifiedSimEnv(num_requests=5)
        wrapped = ActionMaskWrapper(env)

        obs, info = wrapped.reset(seed=42)
        steps = 0

        while True:
            mask = wrapped.action_masks()
            # Select first valid action or 0
            if np.any(mask):
                action = int(np.argmax(mask))
            else:
                action = 0

            obs, reward, terminated, truncated, info = wrapped.step(action)
            steps += 1

            if terminated or truncated:
                break

        assert steps == 5
        assert terminated


# Check if sb3-contrib is available for integration tests
try:
    from sb3_contrib import MaskablePPO

    HAS_SB3_CONTRIB = True
except ImportError:
    HAS_SB3_CONTRIB = False


@pytest.mark.skipif(not HAS_SB3_CONTRIB, reason="sb3-contrib not installed")
class TestSB3Integration:
    """Integration tests with Stable-Baselines3 MaskablePPO (Chunk 11).

    This is Integration Checkpoint 3 - verifies RL training runs.
    These tests require sb3-contrib to be installed.

    Note: We use "MultiInputPolicy" for Dict observation spaces.
    """

    def test_can_create_maskable_ppo(self) -> None:
        """Can create MaskablePPO with wrapped environment."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        # Use MultiInputPolicy for Dict observation spaces
        model = MaskablePPO(
            "MultiInputPolicy",
            wrapped,
            verbose=0,
        )

        assert model is not None

    def test_can_call_predict(self) -> None:
        """Can call predict() on MaskablePPO model."""
        env = UnifiedSimEnv(num_requests=10)
        wrapped = ActionMaskWrapper(env)

        model = MaskablePPO(
            "MultiInputPolicy",
            wrapped,
            verbose=0,
        )

        obs, _ = wrapped.reset(seed=42)
        # MultiInputPolicy accepts dict obs; mypy types are incomplete
        action, _ = model.predict(obs, action_masks=wrapped.action_masks())  # type: ignore[arg-type]

        # Action can be int, np.integer, or np.ndarray with single element
        if isinstance(action, np.ndarray):
            action_val = int(action.item())
        else:
            action_val = int(action)

        assert 0 <= action_val < env.action_space.n

    def test_can_train_1000_steps(self) -> None:
        """Can train MaskablePPO for 1000 timesteps without crashing."""
        env = UnifiedSimEnv(num_requests=50)
        wrapped = ActionMaskWrapper(env)

        model = MaskablePPO(
            "MultiInputPolicy",
            wrapped,
            verbose=0,
            n_steps=64,  # Smaller batch for faster test
            batch_size=32,
        )

        # This should not raise any exceptions
        model.learn(total_timesteps=1000)

    def test_can_train_and_evaluate(self) -> None:
        """Can train and then evaluate the model."""
        env = UnifiedSimEnv(num_requests=20)
        wrapped = ActionMaskWrapper(env)

        model = MaskablePPO(
            "MultiInputPolicy",
            wrapped,
            verbose=0,
            n_steps=64,
            batch_size=32,
        )

        # Train briefly
        model.learn(total_timesteps=500)

        # Evaluate for one episode
        obs, info = wrapped.reset(seed=99)
        total_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, action_masks=wrapped.action_masks())  # type: ignore[arg-type]
            obs, reward, terminated, truncated, info = wrapped.step(int(action))
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        assert steps == 20  # Should complete all requests
        # Reward can be positive or negative depending on learning


class TestUnifiedSimEnvGNNObservations:
    """Tests for GNN observation mode (Chunk 12)."""

    def test_gnn_mode_disabled_by_default(self) -> None:
        """GNN mode is disabled by default."""
        config = RLConfig()
        assert not config.use_gnn_obs

    def test_gnn_mode_adds_adjacency_to_obs_space(self) -> None:
        """GNN mode adds adjacency matrix to observation space."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        assert "adjacency" in env.observation_space.spaces
        assert env.observation_space["adjacency"].shape == (10, 10)

    def test_gnn_mode_adds_node_features_to_obs_space(self) -> None:
        """GNN mode adds node features to observation space."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10, num_node_features=4)
        env = UnifiedSimEnv(config=config)

        assert "node_features" in env.observation_space.spaces
        assert env.observation_space["node_features"].shape == (10, 4)

    def test_gnn_mode_custom_num_node_features(self) -> None:
        """GNN mode respects custom num_node_features."""
        config = RLConfig(use_gnn_obs=True, num_nodes=8, num_node_features=6)
        env = UnifiedSimEnv(config=config)

        assert env.observation_space["node_features"].shape == (8, 6)

    def test_non_gnn_mode_no_adjacency(self) -> None:
        """Non-GNN mode does not include adjacency."""
        config = RLConfig(use_gnn_obs=False)
        env = UnifiedSimEnv(config=config)

        assert "adjacency" not in env.observation_space.spaces
        assert "node_features" not in env.observation_space.spaces

    def test_gnn_reset_returns_adjacency(self) -> None:
        """reset() returns adjacency matrix in GNN mode."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        obs, _ = env.reset(seed=42)

        assert "adjacency" in obs
        assert obs["adjacency"].shape == (10, 10)
        assert obs["adjacency"].dtype == np.float32

    def test_gnn_reset_returns_node_features(self) -> None:
        """reset() returns node features in GNN mode."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10, num_node_features=4)
        env = UnifiedSimEnv(config=config)

        obs, _ = env.reset(seed=42)

        assert "node_features" in obs
        assert obs["node_features"].shape == (10, 4)
        assert obs["node_features"].dtype == np.float32

    def test_gnn_step_returns_adjacency(self) -> None:
        """step() returns adjacency matrix in GNN mode."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config, num_requests=5)

        env.reset(seed=42)
        obs, _, _, _, _ = env.step(0)

        assert "adjacency" in obs
        assert obs["adjacency"].shape == (10, 10)

    def test_gnn_adjacency_is_symmetric(self) -> None:
        """Adjacency matrix is symmetric."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        obs, _ = env.reset(seed=42)
        adj = obs["adjacency"]

        assert np.allclose(adj, adj.T), "Adjacency matrix should be symmetric"

    def test_gnn_adjacency_values_in_range(self) -> None:
        """Adjacency matrix values are in [0, 1]."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        obs, _ = env.reset(seed=42)
        adj = obs["adjacency"]

        assert np.all(adj >= 0.0)
        assert np.all(adj <= 1.0)

    def test_gnn_node_features_in_range(self) -> None:
        """Node features are in [0, 1]."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        obs, _ = env.reset(seed=42)
        features = obs["node_features"]

        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_gnn_node_features_source_marked(self) -> None:
        """Source node is marked in node features."""
        config = RLConfig(use_gnn_obs=True, num_nodes=14, num_node_features=4)
        env = UnifiedSimEnv(config=config)

        env.reset(seed=42)
        req = env.current_request
        assert req is not None

        obs, _ = env.reset(seed=42)  # Reset again to get same request
        features = obs["node_features"]

        # Feature 3 (index 3) is the source/destination indicator
        # Source should be 1.0
        src_idx = int(req.source)
        assert features[src_idx, 3] == 1.0

    def test_gnn_node_features_destination_marked(self) -> None:
        """Destination node is marked in node features."""
        config = RLConfig(use_gnn_obs=True, num_nodes=14, num_node_features=4)
        env = UnifiedSimEnv(config=config)

        env.reset(seed=42)
        req = env.current_request
        assert req is not None

        obs, _ = env.reset(seed=42)  # Reset again to get same request
        features = obs["node_features"]

        # Feature 3 (index 3) is the source/destination indicator
        # Destination should be 0.5
        dst_idx = int(req.destination)
        assert features[dst_idx, 3] == 0.5

    def test_gnn_observation_in_space(self) -> None:
        """GNN observation is contained in observation space."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        obs, _ = env.reset(seed=42)

        assert env.observation_space.contains(obs)

    def test_gnn_zero_observation_has_correct_shape(self) -> None:
        """Zero observation includes GNN features with correct shapes."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10, num_node_features=4)
        env = UnifiedSimEnv(config=config)

        obs = env._zero_observation()

        assert "adjacency" in obs
        assert "node_features" in obs
        assert obs["adjacency"].shape == (10, 10)
        assert obs["node_features"].shape == (10, 4)
        assert np.all(obs["adjacency"] == 0)
        assert np.all(obs["node_features"] == 0)

    def test_gnn_passes_env_checker(self) -> None:
        """GNN mode environment passes gymnasium env_checker."""
        from gymnasium.utils.env_checker import check_env

        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config, num_requests=10)
        check_env(env, skip_render_check=True)


class TestUnifiedSimEnvPyGGraphObservations:
    """Tests for PyG-format graph observations (edge_index, edge_attr, path_masks)."""

    def test_gnn_mode_adds_edge_index(self) -> None:
        """GNN mode adds edge_index in PyG format [2, num_edges]."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        assert "edge_index" in env.observation_space.spaces
        # Shape is [2, num_edges] where num_edges is estimated initially
        edge_index_shape = env.observation_space["edge_index"].shape
        assert edge_index_shape is not None
        assert edge_index_shape[0] == 2

    def test_gnn_mode_adds_edge_attr(self) -> None:
        """GNN mode adds edge_attr."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)

        assert "edge_attr" in env.observation_space.spaces
        # Shape is [num_edges, edge_dim]
        edge_attr_shape = env.observation_space["edge_attr"].shape
        assert edge_attr_shape is not None
        assert edge_attr_shape[1] == 2  # [utilization, length]

    def test_gnn_mode_adds_path_masks(self) -> None:
        """GNN mode adds path_masks."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10, k_paths=5)
        env = UnifiedSimEnv(config=config)

        assert "path_masks" in env.observation_space.spaces
        # Shape is [k_paths, num_edges]
        path_masks_shape = env.observation_space["path_masks"].shape
        assert path_masks_shape is not None
        assert path_masks_shape[0] == 5

    def test_edge_index_contains_valid_indices(self) -> None:
        """edge_index values are valid node indices."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        edge_index = obs["edge_index"]
        # All values should be valid node indices [0, num_nodes-1]
        assert np.all(edge_index >= 0)
        assert np.all(edge_index < 10)

    def test_edge_attr_is_float(self) -> None:
        """edge_attr values are float32."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert obs["edge_attr"].dtype == np.float32

    def test_path_masks_are_binary(self) -> None:
        """path_masks values are in [0, 1]."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10, k_paths=3)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert np.all(obs["path_masks"] >= 0.0)
        assert np.all(obs["path_masks"] <= 1.0)

    def test_edge_index_dtype_is_int64(self) -> None:
        """edge_index dtype is int64."""
        config = RLConfig(use_gnn_obs=True, num_nodes=10)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert obs["edge_index"].dtype == np.int64


class TestPathEncoder:
    """Tests for PathEncoder class."""

    def test_path_encoder_init(self) -> None:
        """PathEncoder initializes correctly."""
        from fusion.modules.rl.environments import PathEncoder

        # Create simple edge_index
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        encoder = PathEncoder(edge_index=edge_index, num_nodes=3)

        assert encoder.num_edges == 4

    def test_path_encoder_encode_simple_path(self) -> None:
        """PathEncoder encodes a simple path correctly."""
        from fusion.modules.rl.environments import PathEncoder

        # Create simple edge_index: 0<->1<->2
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        encoder = PathEncoder(edge_index=edge_index, num_nodes=3)

        # Encode path 0->1->2
        path = ("0", "1", "2")
        mask = encoder.encode_path(path)

        assert mask.shape == (4,)
        # Should mark edges 0->1 and 1->2
        assert mask.sum() == 2

    def test_path_encoder_empty_path(self) -> None:
        """PathEncoder handles single-node paths."""
        from fusion.modules.rl.environments import PathEncoder

        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
        encoder = PathEncoder(edge_index=edge_index, num_nodes=2)

        # Single node path (no edges)
        path = ("0",)
        mask = encoder.encode_path(path)

        assert mask.sum() == 0

    def test_path_encoder_reverse_direction(self) -> None:
        """PathEncoder handles reverse direction edges."""
        from fusion.modules.rl.environments import PathEncoder

        # Only forward edges
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
        encoder = PathEncoder(edge_index=edge_index, num_nodes=3)

        # Path in reverse direction: 2->1->0
        path = ("2", "1", "0")
        mask = encoder.encode_path(path)

        # Should find edges due to reverse lookup
        assert mask.dtype == np.float32


class TestConfigurableObsSpace:
    """Tests for configurable observation space (obs_1 through obs_8)."""

    def test_obs_space_default_is_obs_8(self) -> None:
        """Default obs_space is obs_8."""
        config = RLConfig()
        assert config.obs_space == "obs_8"

    def test_obs_1_minimal_features(self) -> None:
        """obs_1 includes only source and destination."""
        config = RLConfig(obs_space="obs_1")
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert "source" in obs
        assert "destination" in obs
        assert "holding_time" not in obs
        assert "request_bandwidth" not in obs

    def test_obs_2_adds_bandwidth(self) -> None:
        """obs_2 adds request_bandwidth."""
        config = RLConfig(obs_space="obs_2")
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert "source" in obs
        assert "destination" in obs
        assert "request_bandwidth" in obs
        assert "holding_time" not in obs

    def test_obs_5_adds_path_features(self) -> None:
        """obs_5 adds slots_needed and path_lengths."""
        config = RLConfig(obs_space="obs_5")
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert "slots_needed" in obs
        assert "path_lengths" in obs
        assert "paths_cong" not in obs  # Added in obs_6

    def test_obs_6_adds_congestion(self) -> None:
        """obs_6 adds paths_cong."""
        config = RLConfig(obs_space="obs_6")
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert "paths_cong" in obs
        assert "available_slots" not in obs  # Added in obs_7

    def test_obs_7_adds_available_slots(self) -> None:
        """obs_7 adds available_slots."""
        config = RLConfig(obs_space="obs_7")
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert "available_slots" in obs
        assert "is_feasible" not in obs  # Added in obs_8

    def test_obs_8_complete_features(self) -> None:
        """obs_8 includes all features including is_feasible."""
        config = RLConfig(obs_space="obs_8")
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        assert "is_feasible" in obs

    def test_request_bandwidth_one_hot(self) -> None:
        """request_bandwidth is one-hot encoded."""
        config = RLConfig(obs_space="obs_2", num_bandwidth_classes=4)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        # Should be one-hot (exactly one 1.0)
        assert obs["request_bandwidth"].shape == (4,)
        assert np.sum(obs["request_bandwidth"]) == 1.0

    def test_obs_graph_suffix_stripped(self) -> None:
        """obs_space with _graph suffix is handled correctly."""
        config = RLConfig(obs_space="obs_7_graph", use_gnn_obs=True)
        env = UnifiedSimEnv(config=config)
        obs, _ = env.reset(seed=42)

        # Should use obs_7 features
        assert "available_slots" in obs
        assert "is_feasible" not in obs
        # Plus GNN features
        assert "adjacency" in obs
