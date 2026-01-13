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

from fusion.modules.rl.adapter import RLConfig, RLSimulationAdapter
from fusion.modules.rl.adapter.path_option import PathOption, PathOptionList
from fusion.modules.rl.args.observation_args import OBS_DICT

if TYPE_CHECKING:
    from fusion.core.orchestrator import SDNOrchestrator
    from fusion.core.simulation import SimulationEngine
    from fusion.domain.config import SimulationConfig
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request


@dataclass
class SimpleRequest:
    """Simple request representation for standalone environment testing.

    This is a lightweight request dataclass used when the environment
    operates without the full V4 simulation stack. It provides a minimal
    interface compatible with the Request protocol for standalone testing.
    """

    request_id: int
    source: str
    destination: str
    bandwidth_gbps: int
    holding_time: float
    arrive_time: float
    depart_time: float = 0.0

    def __post_init__(self) -> None:
        """Calculate depart_time if not set."""
        if self.depart_time == 0.0:
            self.depart_time = self.arrive_time + self.holding_time


class UnifiedSimEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium environment for optical network RSA using V4 simulation.

    This environment provides RL agents with access to the optical network
    routing and spectrum assignment problem. It supports two modes:

    1. **Wired Mode** (production): Uses SimulationEngine, SDNOrchestrator,
       and RLSimulationAdapter with real spectrum checks and allocations.

    2. **Standalone Mode** (testing): Uses synthetic requests and random
       feasibility for unit testing without full simulation dependencies.

    Key Features:
    - Uses same pipelines as non-RL simulation (no code duplication)
    - Action masking via info["action_mask"] for SB3 MaskablePPO
    - Deterministic seeding for reproducibility
    - Gymnasium-compliant interface
    - Graph observations for GNN policies (optional)

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

        Optional graph observations (when use_gnn_obs=True):
        - adjacency: Adjacency matrix (num_nodes, num_nodes)
        - node_features: Node feature matrix (num_nodes, num_node_features)
        - edge_index: Edge connectivity [2, num_edges] (PyG format)
        - edge_attr: Edge attributes (num_edges, edge_dim)
        - path_masks: Path-to-edge masks (k_paths, num_edges)

    Action Space:
        Discrete(k_paths) - select which candidate path to use

    Example (Standalone Mode):
        config = RLConfig(k_paths=3, num_nodes=14)
        env = UnifiedSimEnv(config)

        obs, info = env.reset(seed=42)
        mask = info["action_mask"]

        valid_actions = np.where(mask)[0]
        action = valid_actions[0] if len(valid_actions) > 0 else 0

        obs, reward, terminated, truncated, info = env.step(action)

    Example (Wired Mode):
        from fusion.core.simulation import SimulationEngine
        from fusion.core.orchestrator import SDNOrchestrator
        from fusion.core.pipeline_factory import PipelineFactory

        engine = SimulationEngine(engine_props)
        orchestrator = PipelineFactory.create_orchestrator(sim_config)
        adapter = RLSimulationAdapter(orchestrator, rl_config)

        env = UnifiedSimEnv(
            config=rl_config,
            engine=engine,
            orchestrator=orchestrator,
            adapter=adapter,
        )
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: RLConfig | None = None,
        render_mode: str | None = None,
        num_requests: int = 100,
        *,
        engine: SimulationEngine | None = None,
        orchestrator: SDNOrchestrator | None = None,
        adapter: RLSimulationAdapter | None = None,
        path_agent: Any | None = None,
        rl_props: Any | None = None,
        path_algorithm: str = "",
        is_training: bool = False,
    ) -> None:
        """Initialize the environment.

        Args:
            config: RL configuration with environment parameters.
                Uses defaults if None.
            render_mode: Rendering mode (not used, for Gymnasium compatibility)
            num_requests: Number of requests per episode (default 100)
            engine: Optional SimulationEngine for wired mode
            orchestrator: Optional SDNOrchestrator for wired mode
            adapter: Optional RLSimulationAdapter for wired mode
            path_agent: Optional PathAgent for non-DRL algorithms (bandits, Q-learning)
            rl_props: Optional RLProps for non-DRL algorithms
            path_algorithm: Name of the path algorithm (e.g., "epsilon_greedy_bandit")
            is_training: Whether in training mode

        Note:
            If engine, orchestrator, and adapter are all provided, the
            environment operates in wired mode using real simulation.
            Otherwise, it operates in standalone mode with synthetic data.
        """
        super().__init__()

        self._config = config or RLConfig()
        self.render_mode = render_mode
        self._num_requests = num_requests

        # Wired mode components (optional)
        self._engine = engine
        self._orchestrator = orchestrator
        self._adapter = adapter

        # RL agent infrastructure for non-DRL algorithms
        self._path_agent = path_agent
        self._rl_props = rl_props
        self._path_algorithm = path_algorithm
        self._is_training = is_training

        # Determine operating mode
        self._wired_mode = (
            engine is not None and orchestrator is not None and adapter is not None
        )

        # Determine if using non-DRL path selection (bandits, Q-learning)
        self._use_rl_agent = path_agent is not None and (
            "bandit" in path_algorithm or path_algorithm == "q_learning"
        )

        # Initialize observation and action spaces
        self._setup_spaces()

        # Episode state - initialized during reset()
        self._requests: list[SimpleRequest] = []
        self._request_index: int = 0
        self._current_request: SimpleRequest | Request | None = None
        self._current_options: PathOptionList = []
        self._current_feasibility: np.ndarray | None = None

        # Graph structures for PyG observations
        self._edge_index: np.ndarray | None = None
        self._path_encoder: PathEncoder | None = None

        # Legacy compatibility attributes for workflow_runner
        # These are expected by the existing RL training infrastructure
        self.trial: int = 0
        self.iteration: int = 0

        # Expose path_agent for legacy compatibility
        self.path_agent = path_agent

    @property
    def is_wired(self) -> bool:
        """Whether environment is wired to real simulation."""
        return self._wired_mode

    @property
    def engine_obj(self) -> SimulationEngine | None:
        """Legacy compatibility: expose engine as engine_obj.

        This property allows the existing RL infrastructure (workflow_runner,
        run_comparison.py) to access the simulation engine for statistics.
        """
        return self._engine

    def _setup_spaces(self) -> None:
        """Initialize observation and action spaces based on config.

        Uses OBS_DICT to determine which features to include based on
        the obs_space configuration (obs_1 through obs_8).
        """
        num_nodes = self._config.num_nodes
        k_paths = self._config.k_paths
        max_slots = self._config.total_slots
        num_bw_classes = self._config.num_bandwidth_classes

        # Get features for configured observation space
        obs_key = self._config.obs_space
        if obs_key.endswith("_graph"):
            obs_key = obs_key.replace("_graph", "")
        self._obs_features = OBS_DICT.get(obs_key, OBS_DICT["obs_8"])

        # Define all possible feature spaces
        all_feature_spaces: dict[str, spaces.Box] = {
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
            "request_bandwidth": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_bw_classes,),
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
            # paths_cong is an alias for congestion (backward compatibility)
            "paths_cong": spaces.Box(
                low=0.0,
                high=1.0,
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

        # Build observation space with only configured features
        obs_spaces: dict[str, spaces.Box] = {}
        for feature in self._obs_features:
            if feature in all_feature_spaces:
                obs_spaces[feature] = all_feature_spaces[feature]

        # Add GNN features if enabled
        if self._config.use_gnn_obs:
            num_node_features = self._config.num_node_features
            # Estimate number of edges (will be updated on reset if wired)
            num_edges = num_nodes * 2  # Rough estimate for sparse graphs

            obs_spaces["adjacency"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_nodes, num_nodes),
                dtype=np.float32,
            )
            obs_spaces["node_features"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_nodes, num_node_features),
                dtype=np.float32,
            )
            # PyG-format edge_index: [2, num_edges]
            obs_spaces["edge_index"] = spaces.Box(
                low=0,
                high=num_nodes - 1,
                shape=(2, num_edges),
                dtype=np.int64,
            )
            # Edge attributes
            obs_spaces["edge_attr"] = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(num_edges, 2),  # [utilization, normalized_length]
                dtype=np.float32,
            )
            # Path masks for k paths
            obs_spaces["path_masks"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(k_paths, num_edges),
                dtype=np.float32,
            )

        self.observation_space: spaces.Dict = spaces.Dict(obs_spaces)

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

        # Update legacy compatibility attributes
        if seed is not None:
            self.trial = seed
        # Note: iteration is incremented at episode END (in step), not at reset
        # This matches legacy SimEnv behavior where first episode has iteration=0

        # Handle options
        if options is not None:
            num_requests = options.get("num_requests", self._num_requests)
        else:
            num_requests = self._num_requests

        if self._wired_mode:
            return self._reset_wired(seed, num_requests)
        else:
            return self._reset_standalone(num_requests)

    def _reset_wired(
        self,
        seed: int | None,
        num_requests: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset in wired mode using real simulation.

        Args:
            seed: Random seed
            num_requests: Number of requests (may be ignored if engine has its own)

        Returns:
            Initial observation and info
        """
        assert self._engine is not None
        assert self._adapter is not None

        # Reset simulation engine for new episode
        episode_seed = seed if seed is not None else 0
        self._engine.reset_rl_state()

        # Initialize iteration (generates requests, creates topology)
        # Must use self.iteration (not hardcoded 0) to match legacy behavior
        # stats_obj.iteration is set inside init_iter and used for saving iter_stats
        self._engine.init_iter(iteration=self.iteration, seed=episode_seed)

        # Sync reward/penalty with engine_props to match legacy behavior
        engine_props = self._engine.engine_props
        if "penalty" in engine_props:
            self._adapter._config.rl_block_penalty = float(engine_props["penalty"])
        if "reward" in engine_props:
            self._adapter._config.rl_success_reward = float(engine_props["reward"])

        # Initialize graph structures if using GNN observations
        if self._config.use_gnn_obs and self._engine.network_state is not None:
            self._init_graph_structures_from_state(self._engine.network_state)

        self._request_index = 0

        # Get first request from engine
        self._current_request = self._engine.get_next_request()
        if self._current_request is None:
            raise RuntimeError("No requests generated by SimulationEngine")

        # Get path options using adapter (real spectrum checks)
        if self._engine.network_state is not None:
            self._current_options = self._adapter.get_path_options(
                self._current_request,
                self._engine.network_state,
            )
        else:
            self._current_options = []

        # Extract feasibility from options
        self._current_feasibility = self._extract_feasibility_from_options()

        # Build observation and info
        obs = self._build_observation()
        info = self._build_info()

        return obs, info

    def _reset_standalone(
        self,
        num_requests: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset in standalone mode with synthetic data.

        Args:
            num_requests: Number of requests to generate

        Returns:
            Initial observation and info
        """
        # Generate requests for this episode using seeded RNG
        self._requests = self._generate_requests(num_requests)
        self._request_index = 0

        # Get first request
        if len(self._requests) > 0:
            self._current_request = self._requests[0]
            # Generate feasibility for paths (simulated for standalone mode)
            self._current_feasibility = self._generate_path_feasibility()
            self._current_options = self._generate_synthetic_options()
        else:
            self._current_request = None
            self._current_feasibility = None
            self._current_options = []

        # Initialize synthetic graph structures for GNN observations
        if self._config.use_gnn_obs:
            self._init_synthetic_graph_structures()

        # Build observation and info
        obs = self._build_observation()
        info = self._build_info()

        return obs, info

    def _init_graph_structures_from_state(self, network_state: NetworkState) -> None:
        """Initialize graph structures from real network state.

        Creates edge_index and PathEncoder from the actual topology.

        Args:
            network_state: NetworkState with topology
        """
        topology = network_state.topology
        num_nodes = topology.number_of_nodes()

        # Build edge list in PyG format: [2, num_edges]
        edges_src: list[int] = []
        edges_dst: list[int] = []

        # Map node IDs to indices
        node_to_idx: dict[str, int] = {}
        for idx, node in enumerate(topology.nodes()):
            node_to_idx[str(node)] = idx

        for u, v in topology.edges():
            u_idx = node_to_idx[str(u)]
            v_idx = node_to_idx[str(v)]
            # Add both directions for undirected graph
            edges_src.extend([u_idx, v_idx])
            edges_dst.extend([v_idx, u_idx])

        self._edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        self._node_to_idx = node_to_idx

        # Create path encoder
        self._path_encoder = PathEncoder(
            edge_index=self._edge_index,
            num_nodes=num_nodes,
        )

        # Update observation space if edge count changed
        num_edges = self._edge_index.shape[1]
        if self._config.use_gnn_obs:
            k_paths = self._config.k_paths
            self.observation_space.spaces["edge_index"] = spaces.Box(
                low=0,
                high=num_nodes - 1,
                shape=(2, num_edges),
                dtype=np.int64,
            )
            self.observation_space.spaces["edge_attr"] = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(num_edges, 2),
                dtype=np.float32,
            )
            self.observation_space.spaces["path_masks"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(k_paths, num_edges),
                dtype=np.float32,
            )

    def _init_synthetic_graph_structures(self) -> None:
        """Initialize synthetic graph structures for standalone mode."""
        n = self._config.num_nodes
        k_paths = self._config.k_paths

        # Create a ring topology (fixed size: 2*n edges for bidirectional)
        edges_src: list[int] = []
        edges_dst: list[int] = []

        # Ring topology - creates exactly 2*n edges
        for i in range(n):
            j = (i + 1) % n
            edges_src.extend([i, j])
            edges_dst.extend([j, i])

        self._edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        self._node_to_idx = {str(i): i for i in range(n)}

        self._path_encoder = PathEncoder(
            edge_index=self._edge_index,
            num_nodes=n,
        )

        # Update observation space to match actual edge count
        num_edges = self._edge_index.shape[1]
        if self._config.use_gnn_obs:
            self.observation_space.spaces["edge_index"] = spaces.Box(
                low=0,
                high=n - 1,
                shape=(2, num_edges),
                dtype=np.int64,
            )
            self.observation_space.spaces["edge_attr"] = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(num_edges, 2),
                dtype=np.float32,
            )
            self.observation_space.spaces["path_masks"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(k_paths, num_edges),
                dtype=np.float32,
            )

    def _extract_feasibility_from_options(self) -> np.ndarray:
        """Extract feasibility array from current path options.

        Returns:
            Boolean array of shape (k_paths,) indicating path feasibility
        """
        k = self._config.k_paths
        feasibility = np.zeros(k, dtype=np.bool_)

        for opt in self._current_options:
            if opt.path_index < k:
                feasibility[opt.path_index] = opt.is_feasible

        return feasibility

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
            source = int(self.np_random.integers(0, num_nodes))
            destination = int(self.np_random.integers(0, num_nodes))
            while destination == source:
                destination = int(self.np_random.integers(0, num_nodes))

            # Generate bandwidth (typical values: 10, 40, 100, 400 Gbps)
            bandwidth_options = [10, 40, 100, 400]
            bandwidth = int(self.np_random.choice(bandwidth_options))

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
                source=str(source),
                destination=str(destination),
                bandwidth_gbps=bandwidth,
                holding_time=holding_time,
                arrive_time=current_time,
            )
            requests.append(request)

        return requests

    def _generate_path_feasibility(self) -> np.ndarray:
        """Generate path feasibility for current request (standalone mode).

        Returns:
            Boolean array of shape (k_paths,) indicating path feasibility
        """
        k = self._config.k_paths
        # In standalone mode, assume ~70% of paths are feasible
        feasibility_prob = 0.7
        feasibility = self.np_random.random(k) < feasibility_prob
        return feasibility.astype(np.bool_)

    def _generate_synthetic_options(self) -> PathOptionList:
        """Generate synthetic path options for standalone mode.

        Returns:
            List of PathOption objects with synthetic data
        """
        if self._current_request is None or self._current_feasibility is None:
            return []

        k = self._config.k_paths
        options: list[PathOption] = []

        req = self._current_request
        src = req.source if isinstance(req.source, str) else str(req.source)
        dst = req.destination if isinstance(req.destination, str) else str(req.destination)

        for i in range(k):
            # Generate synthetic path data
            num_hops = int(self.np_random.integers(2, 7))
            weight_km = float(self.np_random.uniform(100.0, 1000.0))
            slots_needed = max(1, req.bandwidth_gbps // 12)

            # Create simple path (just src -> dst for synthetic)
            path = (src,) + tuple(f"node_{j}" for j in range(num_hops - 1)) + (dst,)

            option = PathOption(
                path_index=i,
                path=path,
                weight_km=weight_km,
                num_hops=num_hops,
                modulation="QPSK",
                slots_needed=slots_needed,
                is_feasible=bool(self._current_feasibility[i]),
                congestion=float(self.np_random.random()),
                available_slots=float(0.3 + 0.7 * self.np_random.random()),
            )
            options.append(option)

        return options

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute action and advance simulation.

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
        if self._wired_mode:
            return self._step_wired(action)
        else:
            return self._step_standalone(action)

    def _step_wired(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute step in wired mode using real simulation.

        Args:
            action: Selected path index (ignored for bandit/Q-learning algorithms)

        Returns:
            Step result tuple
        """
        assert self._engine is not None
        assert self._adapter is not None
        assert self._current_request is not None

        # For non-DRL algorithms (bandits, Q-learning), use the RL agent to select path
        # The action parameter is ignored - the algorithm decides based on Q-values
        actual_action = action
        if self._use_rl_agent and self._path_agent is not None:
            algorithm_obj = getattr(self._path_agent, "algorithm_obj", None)
            if algorithm_obj is not None and hasattr(algorithm_obj, "select_path_arm"):
                # Get source/dest from current request
                source = int(self._current_request.source)
                dest = int(self._current_request.destination)

                # Set epsilon for bandit (needed for exploration)
                if hasattr(algorithm_obj, "epsilon"):
                    engine_props = self._engine.engine_props
                    algorithm_obj.epsilon = engine_props.get("epsilon_start", 1.0)

                # Use bandit algorithm to select path based on Q-values
                actual_action = algorithm_obj.select_path_arm(source=source, dest=dest)

                # Update rl_props for tracking (if available)
                if self._rl_props is not None:
                    self._rl_props.chosen_path_index = actual_action

        # Apply action via adapter (routes through orchestrator)
        result = self._adapter.apply_action(
            action=actual_action,
            request=self._current_request,
            network_state=self._engine.network_state,
            options=self._current_options,
        )

        # Compute reward
        reward = self._adapter.compute_reward(result, self._current_request)

        # Update bandit algorithm with reward (for non-DRL algorithms)
        if self._use_rl_agent and self._path_agent is not None and self._is_training:
            algorithm_obj = getattr(self._path_agent, "algorithm_obj", None)
            if algorithm_obj is not None and hasattr(algorithm_obj, "update"):
                # Sync bandit's iteration counter with env's iteration
                if hasattr(algorithm_obj, "iteration"):
                    algorithm_obj.iteration = self.iteration
                # Update the bandit Q-values with the reward
                algorithm_obj.update(
                    arm=actual_action,
                    reward=reward,
                    iteration=self.iteration,
                    trial=self.trial,
                )

        # Record result (updates stats, schedules release if success)
        self._engine.record_allocation_result(self._current_request, result)

        # Advance to next request
        self._request_index += 1

        # Process releases and get next request
        next_request = self._engine.get_next_request()
        if next_request is not None:
            # Process releases due before next arrival
            self._engine.process_releases_until(next_request.arrive_time)
            self._current_request = next_request

            # Get fresh path options
            if self._engine.network_state is not None:
                self._current_options = self._adapter.get_path_options(
                    self._current_request,
                    self._engine.network_state,
                )
            else:
                self._current_options = []

            self._current_feasibility = self._extract_feasibility_from_options()
        else:
            self._current_request = None
            self._current_options = []
            self._current_feasibility = None

        # Check termination
        terminated = self._current_request is None

        # Call end_iter for path_agent at episode end (updates hyperparams like epsilon)
        if terminated:
            if self._use_rl_agent and self._path_agent is not None:
                if hasattr(self._path_agent, "end_iter"):
                    try:
                        self._path_agent.end_iter()
                    except (ValueError, AttributeError):
                        # hyperparam_obj may not be initialized for some algorithms
                        pass
            # Call engine.end_iter() to calculate and record blocking statistics
            # This matches what legacy SimEnv does in sim_env.py:check_terminated()
            if self._engine is not None:
                self._engine.end_iter(
                    iteration=self.iteration,
                    print_flag=False,
                    base_file_path="data",
                )
            # Increment iteration counter at episode end (matches legacy behavior)
            self.iteration += 1

        # Build observation and info
        obs = self._build_observation()
        info = self._build_info()

        return obs, float(reward), terminated, False, info

    def _step_standalone(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute step in standalone mode with synthetic data.

        Args:
            action: Selected path index

        Returns:
            Step result tuple
        """
        # Validate state
        if len(self._requests) == 0:
            raise RuntimeError("Must call reset() before step()")

        if self._request_index >= len(self._requests):
            raise RuntimeError("Episode has ended. Call reset() to start new episode.")

        if self._current_request is None:
            raise RuntimeError("No current request. Call reset() to start new episode.")

        # Compute reward based on action feasibility
        reward = self._compute_standalone_reward(action)

        # Advance to next request
        self._request_index += 1

        # Check termination
        terminated = self._request_index >= len(self._requests)
        truncated = False

        # Update current request and feasibility for next step
        if not terminated:
            self._current_request = self._requests[self._request_index]
            self._current_feasibility = self._generate_path_feasibility()
            self._current_options = self._generate_synthetic_options()
        else:
            self._current_request = None
            self._current_feasibility = None
            self._current_options = []

        # Build observation and info
        obs = self._build_observation()
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def _compute_standalone_reward(self, action: int) -> float:
        """Compute reward in standalone mode based on feasibility.

        Args:
            action: Selected path index

        Returns:
            Reward value
        """
        k = self._config.k_paths

        # Check if action is valid
        if action < 0 or action >= k:
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

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Build observation dict from current request state.

        Only includes features configured in obs_space (obs_1 through obs_8).

        Returns:
            Observation dict matching observation_space
        """
        if self._current_request is None:
            return self._zero_observation()

        k = self._config.k_paths
        n = self._config.num_nodes
        max_ht = self._config.max_holding_time
        num_bw_classes = self._config.num_bandwidth_classes

        req = self._current_request

        # Handle both SimpleRequest and Request types
        src = req.source
        dst = req.destination
        src_idx = int(src) if isinstance(src, (int, str)) and str(src).isdigit() else 0
        dst_idx = int(dst) if isinstance(dst, (int, str)) and str(dst).isdigit() else 0

        # Prepare all possible features
        all_features: dict[str, np.ndarray] = {}

        # Source: One-hot encoded
        if "source" in self._obs_features:
            source_onehot = np.zeros(n, dtype=np.float32)
            if 0 <= src_idx < n:
                source_onehot[src_idx] = 1.0
            all_features["source"] = source_onehot

        # Destination: One-hot encoded
        if "destination" in self._obs_features:
            dest_onehot = np.zeros(n, dtype=np.float32)
            if 0 <= dst_idx < n:
                dest_onehot[dst_idx] = 1.0
            all_features["destination"] = dest_onehot

        # Request bandwidth: One-hot encoded by class
        if "request_bandwidth" in self._obs_features:
            bandwidth = getattr(req, "bandwidth_gbps", 100)
            # Map bandwidth to class index (10->0, 40->1, 100->2, 400->3)
            bandwidth_classes = [10, 40, 100, 400]
            bw_onehot = np.zeros(num_bw_classes, dtype=np.float32)
            if bandwidth in bandwidth_classes:
                bw_idx = bandwidth_classes.index(bandwidth)
                if bw_idx < num_bw_classes:
                    bw_onehot[bw_idx] = 1.0
            else:
                # Default to highest class for unknown bandwidths
                bw_onehot[min(num_bw_classes - 1, 3)] = 1.0
            all_features["request_bandwidth"] = bw_onehot

        # Holding time: Normalized to [0, 1]
        if "holding_time" in self._obs_features:
            holding_time = getattr(req, "holding_time", max_ht / 2)
            holding_time_norm = min(holding_time / max_ht, 1.0) if max_ht > 0 else 0.0
            all_features["holding_time"] = np.array([holding_time_norm], dtype=np.float32)

        # Path features from options
        slots_needed = np.full(k, -1.0, dtype=np.float32)
        path_lengths = np.zeros(k, dtype=np.float32)
        congestion = np.zeros(k, dtype=np.float32)
        available_slots = np.zeros(k, dtype=np.float32)

        for opt in self._current_options:
            i = opt.path_index
            if i < k:
                slots_needed[i] = float(opt.slots_needed)
                path_lengths[i] = float(opt.num_hops)
                congestion[i] = opt.congestion
                available_slots[i] = opt.available_slots

        if "slots_needed" in self._obs_features:
            all_features["slots_needed"] = slots_needed
        if "path_lengths" in self._obs_features:
            all_features["path_lengths"] = path_lengths
        if "paths_cong" in self._obs_features:
            all_features["paths_cong"] = congestion
        if "congestion" in self._obs_features:
            all_features["congestion"] = congestion
        if "available_slots" in self._obs_features:
            all_features["available_slots"] = available_slots

        # Feasibility
        if "is_feasible" in self._obs_features:
            if self._current_feasibility is not None:
                all_features["is_feasible"] = self._current_feasibility.astype(np.float32)
            else:
                all_features["is_feasible"] = np.zeros(k, dtype=np.float32)

        # Build observation with only configured features
        obs: dict[str, np.ndarray] = {}
        for feature in self._obs_features:
            if feature in all_features:
                obs[feature] = all_features[feature]

        # Add GNN features if enabled
        if self._config.use_gnn_obs:
            obs.update(self._build_graph_observation())

        return obs

    def _build_graph_observation(self) -> dict[str, np.ndarray]:
        """Build graph-structured observation components.

        Returns:
            Dict with graph observation components
        """
        n = self._config.num_nodes
        k = self._config.k_paths

        # Adjacency matrix from edge_index
        if self._edge_index is not None:
            num_edges = self._edge_index.shape[1]
            adjacency = np.zeros((n, n), dtype=np.float32)
            for e in range(num_edges):
                src = int(self._edge_index[0, e])
                dst = int(self._edge_index[1, e])
                if src < n and dst < n:
                    adjacency[src, dst] = 1.0
        else:
            adjacency = np.zeros((n, n), dtype=np.float32)
            num_edges = n * 2

        # Node features
        node_features = self._compute_node_features()

        # Edge attributes
        edge_attr = self._compute_edge_features()

        # Path masks
        path_masks = self._compute_path_masks()

        # Return edge_index or zeros
        if self._edge_index is not None:
            edge_index = self._edge_index.copy()
        else:
            edge_index = np.zeros((2, num_edges), dtype=np.int64)

        return {
            "adjacency": adjacency,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "path_masks": path_masks,
        }

    def _compute_node_features(self) -> np.ndarray:
        """Compute node feature matrix.

        Features per node:
        - utilization: Link utilization around the node [0, 1]
        - degree: Normalized node degree [0, 1]
        - centrality: Betweenness centrality approximation [0, 1]
        - is_src_dst: 1.0 if source, 0.5 if destination, 0.0 otherwise

        Returns:
            Node features of shape (num_nodes, num_node_features)
        """
        n = self._config.num_nodes
        num_features = self._config.num_node_features
        features = np.zeros((n, num_features), dtype=np.float32)

        # Get current request source/destination
        src_idx = -1
        dst_idx = -1
        if self._current_request is not None:
            src = self._current_request.source
            dst = self._current_request.destination
            src_idx = int(src) if isinstance(src, (int, str)) and str(src).isdigit() else -1
            dst_idx = int(dst) if isinstance(dst, (int, str)) and str(dst).isdigit() else -1

        # In wired mode, use real network state
        if self._wired_mode and self._engine is not None and self._engine.network_state is not None:
            topology = self._engine.network_state.topology
            degrees = dict(topology.degree())
            max_degree = max(degrees.values()) if degrees else 1

            for i in range(n):
                node_id = str(i)
                if node_id in degrees:
                    features[i, 0] = float(self.np_random.random())  # Utilization placeholder
                    features[i, 1] = degrees[node_id] / max_degree
                    features[i, 2] = float(self.np_random.random())  # Centrality placeholder
                if i == src_idx:
                    features[i, 3] = 1.0
                elif i == dst_idx:
                    features[i, 3] = 0.5
        else:
            # Standalone mode - synthetic features
            for i in range(n):
                features[i, 0] = float(self.np_random.random())
                features[i, 1] = float(self.np_random.integers(2, 7)) / 6.0
                features[i, 2] = float(self.np_random.random())
                if i == src_idx:
                    features[i, 3] = 1.0
                elif i == dst_idx:
                    features[i, 3] = 0.5

        return features

    def _compute_edge_features(self) -> np.ndarray:
        """Compute edge feature matrix.

        Features per edge:
        - Link utilization [0, 1]
        - Normalized link length [0, 1]

        Returns:
            Edge features of shape (num_edges, 2)
        """
        if self._edge_index is None:
            return np.zeros((self._config.num_nodes * 2, 2), dtype=np.float32)

        num_edges = self._edge_index.shape[1]
        edge_attr = np.zeros((num_edges, 2), dtype=np.float32)

        # In wired mode, try to get real utilization
        if self._wired_mode and self._engine is not None and self._engine.network_state is not None:
            state = self._engine.network_state
            for e in range(num_edges):
                src_idx = int(self._edge_index[0, e])
                dst_idx = int(self._edge_index[1, e])
                # Utilization placeholder (real would query spectrum)
                edge_attr[e, 0] = state.get_spectrum_utilization() if hasattr(state, "get_spectrum_utilization") else 0.0
                # Normalized length placeholder
                edge_attr[e, 1] = 0.5
        else:
            # Standalone mode - random features
            for e in range(num_edges):
                edge_attr[e, 0] = float(self.np_random.random())
                edge_attr[e, 1] = float(self.np_random.random())

        return edge_attr

    def _compute_path_masks(self) -> np.ndarray:
        """Compute path-to-edge masks.

        Returns:
            Binary matrix [k_paths, num_edges] where 1 indicates
            the edge is used by the path.
        """
        k = self._config.k_paths

        if self._edge_index is None:
            num_edges = self._config.num_nodes * 2
            return np.zeros((k, num_edges), dtype=np.float32)

        num_edges = self._edge_index.shape[1]
        path_masks = np.zeros((k, num_edges), dtype=np.float32)

        if self._path_encoder is not None:
            for opt in self._current_options:
                i = opt.path_index
                if i < k:
                    path_masks[i] = self._path_encoder.encode_path(opt.path)

        return path_masks

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

        total = self._engine.num_requests if self._wired_mode and self._engine else len(self._requests)

        return {
            "action_mask": mask,
            "request_index": self._request_index,
            "total_requests": total,
        }

    def _zero_observation(self) -> dict[str, np.ndarray]:
        """Return zero-filled observation for terminal state.

        Only includes features configured in obs_space (obs_1 through obs_8).
        """
        k = self._config.k_paths
        n = self._config.num_nodes
        num_bw_classes = self._config.num_bandwidth_classes

        # Define zero values for all possible features
        all_zeros: dict[str, np.ndarray] = {
            "source": np.zeros(n, dtype=np.float32),
            "destination": np.zeros(n, dtype=np.float32),
            "request_bandwidth": np.zeros(num_bw_classes, dtype=np.float32),
            "holding_time": np.zeros(1, dtype=np.float32),
            "slots_needed": np.full(k, -1.0, dtype=np.float32),
            "path_lengths": np.zeros(k, dtype=np.float32),
            "paths_cong": np.zeros(k, dtype=np.float32),
            "congestion": np.zeros(k, dtype=np.float32),
            "available_slots": np.zeros(k, dtype=np.float32),
            "is_feasible": np.zeros(k, dtype=np.float32),
        }

        # Build observation with only configured features
        obs: dict[str, np.ndarray] = {}
        for feature in self._obs_features:
            if feature in all_zeros:
                obs[feature] = all_zeros[feature]

        # Add zero GNN features if enabled
        if self._config.use_gnn_obs:
            num_node_features = self._config.num_node_features
            num_edges = self._edge_index.shape[1] if self._edge_index is not None else n * 2
            obs["adjacency"] = np.zeros((n, n), dtype=np.float32)
            obs["node_features"] = np.zeros((n, num_node_features), dtype=np.float32)
            obs["edge_index"] = np.zeros((2, num_edges), dtype=np.int64)
            obs["edge_attr"] = np.zeros((num_edges, 2), dtype=np.float32)
            obs["path_masks"] = np.zeros((k, num_edges), dtype=np.float32)

        return obs

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    # --- Properties for accessing episode state ---

    @property
    def current_request(self) -> SimpleRequest | Request | None:
        """Current request being processed."""
        return self._current_request

    @property
    def current_options(self) -> PathOptionList:
        """Current path options for the request."""
        return self._current_options

    @property
    def request_index(self) -> int:
        """Index of current request in episode."""
        return self._request_index

    @property
    def num_requests(self) -> int:
        """Total number of requests in current episode."""
        if self._wired_mode and self._engine:
            return self._engine.num_requests
        return len(self._requests)

    @property
    def is_episode_done(self) -> bool:
        """Whether the episode has ended."""
        if self._wired_mode:
            return self._current_request is None
        return self._request_index >= len(self._requests)

    @property
    def adapter(self) -> RLSimulationAdapter | None:
        """Access to the RL adapter (wired mode only)."""
        return self._adapter

    @property
    def network_state(self) -> NetworkState | None:
        """Access to network state (wired mode only)."""
        if self._wired_mode and self._engine:
            return self._engine.network_state
        return None


class PathEncoder:
    """Encodes paths as binary edge masks for GNN observations.

    Converts path node sequences to binary edge masks indicating
    which edges in the graph are used by the path.
    """

    def __init__(self, edge_index: np.ndarray, num_nodes: int) -> None:
        """Initialize path encoder.

        Args:
            edge_index: [2, num_edges] edge index array in PyG format
            num_nodes: Number of nodes in graph
        """
        self._edge_index = edge_index
        self._num_edges = edge_index.shape[1]
        self._num_nodes = num_nodes

        # Build edge lookup: (src, dst) -> edge_index
        self._edge_lookup: dict[tuple[int, int], int] = {}
        for e in range(self._num_edges):
            src = int(edge_index[0, e])
            dst = int(edge_index[1, e])
            self._edge_lookup[(src, dst)] = e

    def encode_path(self, path: tuple[str, ...]) -> np.ndarray:
        """Encode path as binary edge mask.

        Args:
            path: Tuple of node IDs as strings

        Returns:
            Binary array [num_edges] with 1s for edges used by the path
        """
        mask = np.zeros(self._num_edges, dtype=np.float32)

        for i in range(len(path) - 1):
            # Convert node IDs to indices
            src_str = path[i]
            dst_str = path[i + 1]

            # Try numeric conversion
            try:
                src = int(src_str)
                dst = int(dst_str)
            except ValueError:
                # Non-numeric node IDs - skip
                continue

            # Check both directions (for undirected interpretation)
            if (src, dst) in self._edge_lookup:
                mask[self._edge_lookup[(src, dst)]] = 1.0
            elif (dst, src) in self._edge_lookup:
                mask[self._edge_lookup[(dst, src)]] = 1.0

        return mask

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self._num_edges
