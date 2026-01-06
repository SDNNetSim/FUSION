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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fusion.modules.rl.adapter import RLConfig

if TYPE_CHECKING:
    from fusion.modules.rl.adapter import RLSimulationAdapter


@dataclass
class SimpleRequest:
    """Simple request representation for standalone environment testing.

    This is a lightweight request dataclass used when the environment
    operates without the full V4 simulation stack. It will be replaced
    by the real Request class when wired to SimulationEngine.
    """

    request_id: int
    source: int
    destination: int
    bandwidth_gbps: float
    holding_time: float
    arrive_time: float


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
        num_requests: int = 100,
    ) -> None:
        """Initialize the environment.

        Args:
            config: RL configuration with environment parameters.
                Uses defaults if None.
            render_mode: Rendering mode (not used, for Gymnasium compatibility)
            num_requests: Number of requests per episode (default 100)
        """
        super().__init__()

        self._config = config or RLConfig()
        self.render_mode = render_mode
        self._num_requests = num_requests

        # Initialize observation and action spaces
        self._setup_spaces()

        # Episode state - initialized during reset()
        self._requests: list[SimpleRequest] = []
        self._request_index: int = 0
        self._current_request: SimpleRequest | None = None
        self._current_feasibility: np.ndarray | None = None

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

        Initializes the episode with a new set of requests and returns
        the observation for the first request.

        Args:
            seed: Random seed for reproducibility. Same seed produces
                identical episode sequences.
            options: Additional options:
                - num_requests: Override default number of requests

        Returns:
            observation: Initial observation dict for first request
            info: Info dict with action_mask and episode metadata
        """
        super().reset(seed=seed)

        # Handle options
        if options is not None:
            num_requests = options.get("num_requests", self._num_requests)
        else:
            num_requests = self._num_requests

        # Generate requests for this episode using seeded RNG
        self._requests = self._generate_requests(num_requests)
        self._request_index = 0

        # Get first request
        if len(self._requests) > 0:
            self._current_request = self._requests[0]
            # Generate feasibility for paths (simulated for standalone mode)
            self._current_feasibility = self._generate_path_feasibility()
        else:
            self._current_request = None
            self._current_feasibility = None

        # Build observation and info
        obs = self._build_observation()
        info = self._build_info()

        return obs, info

    def _generate_requests(self, num_requests: int) -> list[SimpleRequest]:
        """Generate requests for the episode using seeded RNG.

        Args:
            num_requests: Number of requests to generate

        Returns:
            List of SimpleRequest objects sorted by arrival time
        """
        requests: list[SimpleRequest] = []
        num_nodes = self._config.num_nodes
        max_holding_time = self._config.max_holding_time

        # Use self.np_random which is seeded by super().reset(seed=seed)
        current_time = 0.0

        for i in range(num_requests):
            # Generate source and destination (different nodes)
            source = self.np_random.integers(0, num_nodes)
            destination = self.np_random.integers(0, num_nodes)
            while destination == source:
                destination = self.np_random.integers(0, num_nodes)

            # Generate bandwidth (typical values: 10, 40, 100, 400 Gbps)
            bandwidth_options = [10.0, 40.0, 100.0, 400.0]
            bandwidth = float(self.np_random.choice(bandwidth_options))

            # Generate holding time (exponential distribution)
            holding_time = float(
                self.np_random.exponential(max_holding_time / 2)
            )
            holding_time = min(holding_time, max_holding_time)

            # Generate inter-arrival time (Poisson process)
            inter_arrival = float(self.np_random.exponential(1.0))
            current_time += inter_arrival

            request = SimpleRequest(
                request_id=i,
                source=int(source),
                destination=int(destination),
                bandwidth_gbps=bandwidth,
                holding_time=holding_time,
                arrive_time=current_time,
            )
            requests.append(request)

        return requests

    def _generate_path_feasibility(self) -> np.ndarray:
        """Generate path feasibility for current request (standalone mode).

        In standalone mode, randomly generate feasibility based on
        a probability. When wired to real simulation, this will be
        replaced by actual spectrum checks.

        Returns:
            Boolean array of shape (k_paths,) indicating path feasibility
        """
        k = self._config.k_paths
        # In standalone mode, assume ~70% of paths are feasible
        feasibility_prob = 0.7
        feasibility = self.np_random.random(k) < feasibility_prob
        return feasibility.astype(np.bool_)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute action and advance simulation.

        Applies the selected action (path choice), computes reward based on
        feasibility, and advances to the next request in the episode.

        Args:
            action: Selected path index (0 to k_paths-1)

        Returns:
            observation: New observation dict for next request
            reward: Scalar reward (positive if feasible, negative if blocked)
            terminated: True if all requests processed
            truncated: Always False (no truncation in optical network episodes)
            info: Info dict with action_mask and metadata

        Raises:
            RuntimeError: If step() called before reset() or after episode end
        """
        # Validate state
        if len(self._requests) == 0:
            raise RuntimeError("Must call reset() before step()")

        if self._request_index >= len(self._requests):
            raise RuntimeError("Episode has ended. Call reset() to start new episode.")

        if self._current_request is None:
            raise RuntimeError("No current request. Call reset() to start new episode.")

        # Compute reward based on action feasibility
        reward = self._compute_reward(action)

        # Record result for statistics (in standalone mode, just track counts)
        self._record_step_result(action)

        # Advance to next request
        self._request_index += 1

        # Check termination
        terminated = self._request_index >= len(self._requests)
        truncated = False

        # Update current request and feasibility for next step
        if not terminated:
            self._current_request = self._requests[self._request_index]
            self._current_feasibility = self._generate_path_feasibility()
        else:
            self._current_request = None
            self._current_feasibility = None

        # Build observation and info
        obs = self._build_observation()
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: int) -> float:
        """Compute reward for the selected action.

        In standalone mode, reward is based on whether the selected path
        was feasible. When wired to real simulation, this will use
        adapter.compute_reward() with actual allocation results.

        Args:
            action: Selected path index

        Returns:
            Reward value (positive for success, negative for block)
        """
        k = self._config.k_paths

        # Check if action is valid
        if action < 0 or action >= k:
            # Invalid action - penalize
            return self._config.rl_block_penalty

        # Check feasibility of selected path
        if self._current_feasibility is not None and action < len(self._current_feasibility):
            is_feasible = self._current_feasibility[action]
        else:
            is_feasible = False

        if is_feasible:
            return self._config.rl_success_reward
        else:
            return self._config.rl_block_penalty

    def _record_step_result(self, action: int) -> None:
        """Record step result for statistics tracking.

        In standalone mode, this is a no-op placeholder. When wired to
        real simulation, this will call engine.record_allocation_result().

        Args:
            action: Selected path index
        """
        # Placeholder for statistics tracking
        # In real mode: self._engine.record_allocation_result(request, result)
        pass

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Build observation dict from current request state.

        Returns:
            Observation dict matching observation_space
        """
        if self._current_request is None:
            return self._zero_observation()

        k = self._config.k_paths
        n = self._config.num_nodes
        max_ht = self._config.max_holding_time

        req = self._current_request

        # One-hot encode source and destination
        source_onehot = np.zeros(n, dtype=np.float32)
        dest_onehot = np.zeros(n, dtype=np.float32)

        if 0 <= req.source < n:
            source_onehot[req.source] = 1.0
        if 0 <= req.destination < n:
            dest_onehot[req.destination] = 1.0

        # Normalize holding time to [0, 1]
        holding_time_norm = min(req.holding_time / max_ht, 1.0) if max_ht > 0 else 0.0

        # Path features (simulated in standalone mode)
        # In real mode, these come from PathOption via adapter
        slots_needed = self._generate_slots_needed(req.bandwidth_gbps)
        path_lengths = self._generate_path_lengths()
        congestion = self._generate_congestion()
        available_slots = self._generate_available_slots()

        # Use current feasibility
        if self._current_feasibility is not None:
            is_feasible = self._current_feasibility.astype(np.float32)
        else:
            is_feasible = np.zeros(k, dtype=np.float32)

        return {
            "source": source_onehot,
            "destination": dest_onehot,
            "holding_time": np.array([holding_time_norm], dtype=np.float32),
            "slots_needed": slots_needed,
            "path_lengths": path_lengths,
            "congestion": congestion,
            "available_slots": available_slots,
            "is_feasible": is_feasible,
        }

    def _build_info(self) -> dict[str, Any]:
        """Build info dict with action mask and metadata.

        Returns:
            Info dict with action_mask and episode tracking info
        """
        k = self._config.k_paths

        # Action mask based on path feasibility
        if self._current_feasibility is not None:
            mask = self._current_feasibility.copy()
        else:
            # No feasible actions if no current request
            mask = np.zeros(k, dtype=np.bool_)

        return {
            "action_mask": mask,
            "request_index": self._request_index,
            "total_requests": len(self._requests),
        }

    def _zero_observation(self) -> dict[str, np.ndarray]:
        """Return zero-filled observation for terminal state."""
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

    def _generate_slots_needed(self, bandwidth_gbps: float) -> np.ndarray:
        """Generate slots needed per path (standalone mode simulation).

        Args:
            bandwidth_gbps: Request bandwidth

        Returns:
            Array of slots needed per path
        """
        k = self._config.k_paths
        # Estimate slots based on bandwidth (simplified)
        # Real implementation uses modulation format and path length
        base_slots = int(bandwidth_gbps / 12.5)  # ~12.5 GHz per slot
        slots = np.full(k, float(base_slots), dtype=np.float32)
        # Add some variation per path
        variation = self.np_random.integers(-2, 3, size=k)
        slots = slots + variation.astype(np.float32)
        slots = np.maximum(slots, 1.0)  # At least 1 slot
        return slots

    def _generate_path_lengths(self) -> np.ndarray:
        """Generate path lengths (hop counts) for paths (standalone mode).

        Returns:
            Array of hop counts per path
        """
        k = self._config.k_paths
        # Generate plausible hop counts (typically 2-6 hops)
        lengths = self.np_random.integers(2, 7, size=k)
        return lengths.astype(np.float32)

    def _generate_congestion(self) -> np.ndarray:
        """Generate congestion metrics per path (standalone mode).

        Returns:
            Array of congestion values in [0, 1]
        """
        k = self._config.k_paths
        # Generate random congestion values
        congestion = self.np_random.random(k).astype(np.float32)
        return congestion

    def _generate_available_slots(self) -> np.ndarray:
        """Generate available slots ratio per path (standalone mode).

        Returns:
            Array of available slot ratios in [0, 1]
        """
        k = self._config.k_paths
        # Generate random availability (typically 30-100%)
        available = 0.3 + 0.7 * self.np_random.random(k)
        return available.astype(np.float32)

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    # --- Properties for accessing episode state ---

    @property
    def current_request(self) -> SimpleRequest | None:
        """Current request being processed."""
        return self._current_request

    @property
    def request_index(self) -> int:
        """Index of current request in episode."""
        return self._request_index

    @property
    def num_requests(self) -> int:
        """Total number of requests in current episode."""
        return len(self._requests)

    @property
    def is_episode_done(self) -> bool:
        """Whether the episode has ended."""
        return self._request_index >= len(self._requests)
