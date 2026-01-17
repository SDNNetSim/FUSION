"""RLSimulationAdapter - coordination layer between RL and V4 simulation.

This module provides the adapter that connects RL environments to the v6.0
simulation stack (SDNOrchestrator, pipelines). The adapter ensures RL agents
use the SAME pipeline instances as the orchestrator, eliminating duplicated
simulation logic.

Key Invariants:
- Pipeline identity: adapter.routing IS orchestrator.routing (same object)
- Stateless: adapter never stores NetworkState
- Read-only queries: get_path_options() doesn't allocate
- Write-through: apply_action() routes through orchestrator

RLSimulationAdapter Scaffolding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fusion.modules.rl.adapter.path_option import (
    ActionMask,
    PathOption,
    PathOptionList,
    compute_action_mask,
)

if TYPE_CHECKING:
    from fusion.core.orchestrator import SDNOrchestrator
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.domain.results import AllocationResult
    from fusion.interfaces.pipelines import RoutingPipeline, SpectrumPipeline


@dataclass
class RLConfig:
    """
    RL-specific configuration.

    :ivar k_paths: Number of candidate paths to consider
    :vartype k_paths: int
    :ivar rl_success_reward: Reward for successful allocation
    :vartype rl_success_reward: float
    :ivar rl_block_penalty: Penalty for blocked request
    :vartype rl_block_penalty: float
    :ivar rl_grooming_bonus: Bonus for groomed allocation
    :vartype rl_grooming_bonus: float
    :ivar rl_slicing_penalty: Penalty for sliced allocation
    :vartype rl_slicing_penalty: float
    :ivar rl_bandwidth_weighted: Whether to weight reward by bandwidth
    :vartype rl_bandwidth_weighted: bool
    :ivar max_holding_time: Maximum holding time for normalization
    :vartype max_holding_time: float
    :ivar num_nodes: Number of nodes in network (for observation space)
    :vartype num_nodes: int
    :ivar total_slots: Total spectrum slots per link
    :vartype total_slots: int
    :ivar use_gnn_obs: Whether to include GNN features (adjacency, node features)
    :vartype use_gnn_obs: bool
    :ivar num_node_features: Number of features per node for GNN mode
    :vartype num_node_features: int
    :ivar obs_space: Observation space configuration (obs_1 through obs_8)
    :vartype obs_space: str
    :ivar num_bandwidth_classes: Number of bandwidth classes for one-hot encoding
    :vartype num_bandwidth_classes: int
    """

    k_paths: int = 3
    rl_success_reward: float = 1.0
    rl_block_penalty: float = -1.0
    rl_grooming_bonus: float = 0.1
    rl_slicing_penalty: float = -0.05
    rl_bandwidth_weighted: bool = False
    max_holding_time: float = 100.0
    num_nodes: int = 14
    total_slots: int = 320
    use_gnn_obs: bool = False
    num_node_features: int = 4  # utilization, degree, centrality, is_src_dst
    obs_space: str = "obs_8"  # obs_1 through obs_8, configurable observation space
    num_bandwidth_classes: int = 4  # Number of bandwidth classes (e.g., 10, 40, 100, 400 Gbps)


@dataclass(frozen=True)
class DisasterState:
    """
    Immutable disaster state information for survivability scenarios.

    This dataclass captures disaster information needed by offline RL
    policies (BC, IQL) that were trained on survivability experiments.

    :ivar active: Whether a disaster is currently active
    :vartype active: bool
    :ivar centroid: Geographic centroid (x, y) of disaster area
    :vartype centroid: tuple[float, float] | None
    :ivar radius: Affected radius from centroid in km
    :vartype radius: float
    :ivar failed_links: Set of failed link tuples (frozen for hashability)
    :vartype failed_links: frozenset[tuple[str, str]]
    :ivar network_diameter: Max distance across network for normalization
    :vartype network_diameter: float
    """

    active: bool
    centroid: tuple[float, float] | None = None
    radius: float = 0.0
    failed_links: frozenset[tuple[str, str]] = frozenset()
    network_diameter: float = 1.0


class RLSimulationAdapter:
    """
    Adapter layer between RL environments and simulation stack.

    This adapter provides RL agents with access to routing and spectrum
    pipelines WITHOUT duplicating any simulation logic. It uses the SAME
    pipeline instances as the SDNOrchestrator.

    Key Invariants:

    - Adapter shares pipeline references with orchestrator (identity, not copy)
    - Adapter never stores NetworkState (receives per-call)
    - Adapter never directly mutates spectrum (goes through orchestrator)

    Example::

        orchestrator = SDNOrchestrator(config, pipelines)
        adapter = RLSimulationAdapter(orchestrator)

        # Verify pipeline identity
        assert adapter.routing is orchestrator.routing
        assert adapter.spectrum is orchestrator.spectrum

    :ivar routing: Reference to shared routing pipeline
    :ivar spectrum: Reference to shared spectrum pipeline
    """

    def __init__(
        self,
        orchestrator: SDNOrchestrator,
        config: RLConfig | None = None,
    ) -> None:
        """
        Initialize adapter with orchestrator reference.

        The adapter stores references to the orchestrator's pipelines,
        ensuring RL code uses the exact same instances as non-RL simulation.

        :param orchestrator: SDNOrchestrator instance (shares pipelines)
        :type orchestrator: SDNOrchestrator
        :param config: RL configuration settings (uses defaults if None)
        :type config: RLConfig | None
        :raises ValueError: If orchestrator is None
        """
        if orchestrator is None:
            raise ValueError("orchestrator cannot be None")

        self._orchestrator = orchestrator
        self._config = config or RLConfig()

        # Store pipeline references - these are the SAME instances used by orchestrator
        # This is critical: adapter.routing IS orchestrator.routing (identity)
        self._routing = orchestrator.routing
        self._spectrum = orchestrator.spectrum

    @property
    def routing(self) -> RoutingPipeline:
        """
        Access to shared routing pipeline (same instance as orchestrator).

        :return: Routing pipeline reference
        :rtype: RoutingPipeline
        """
        return self._routing

    @property
    def spectrum(self) -> SpectrumPipeline:
        """
        Access to shared spectrum pipeline (same instance as orchestrator).

        :return: Spectrum pipeline reference
        :rtype: SpectrumPipeline
        """
        return self._spectrum

    @property
    def orchestrator(self) -> SDNOrchestrator:
        """
        Access to the underlying orchestrator.

        :return: SDN orchestrator reference
        :rtype: SDNOrchestrator
        """
        return self._orchestrator

    @property
    def config(self) -> RLConfig:
        """
        Access to RL configuration.

        :return: RL configuration object
        :rtype: RLConfig
        """
        return self._config

    def get_path_options(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> PathOptionList:
        """
        Get candidate paths with feasibility information.

        This method is READ-ONLY. It queries routing and spectrum pipelines
        but does NOT allocate any spectrum or modify network state.

        :param request: Current request to route
        :type request: Request
        :param network_state: Current network state (not stored by adapter)
        :type network_state: NetworkState
        :return: List of PathOption, one per candidate path, with path geometry
            (nodes, length, hops), modulation and slots needed, feasibility from
            real spectrum check, and congestion/availability metrics
        :rtype: PathOptionList

        .. note::

            The number of options returned may be less than k_paths
            if routing cannot find enough paths.
        """
        # 1. Get candidate paths from routing pipeline
        route_result = self._routing.find_routes(
            source=request.source,
            destination=request.destination,
            bandwidth_gbps=request.bandwidth_gbps,
            network_state=network_state,
        )

        # If no paths found, return empty list
        if route_result.is_empty:
            return []

        options: list[PathOption] = []

        # 2. For each candidate path, check spectrum feasibility
        for path_idx in range(len(route_result.paths)):
            path = route_result.paths[path_idx]
            weight_km = route_result.weights_km[path_idx]
            modulations = route_result.modulations[path_idx]

            # Filter out False/None values (modulations that don't reach path distance)
            valid_mods = [m for m in modulations if m and m is not False]

            # Like legacy RL, use get_path_modulation to select ONE modulation
            # based on path length. Legacy does:
            #   modulation_format = get_path_modulation(mod_per_bw[bandwidth], path_length)
            # This ensures we pick the appropriate modulation for the path distance.
            if valid_mods:
                from fusion.utils.network import get_path_modulation
                # Get mod_per_bw from orchestrator's config
                mod_per_bw = self._orchestrator.config.mod_per_bw
                bw_key = str(request.bandwidth_gbps)
                if bw_key in mod_per_bw:
                    single_mod = get_path_modulation(
                        modulation_formats=mod_per_bw[bw_key],
                        path_length=weight_km,
                    )
                    # get_path_modulation returns False if path too long
                    if single_mod and single_mod is not False:
                        valid_mods = [str(single_mod)]
                    else:
                        valid_mods = []  # Path too long for any modulation

            # Default values for spectrum-related fields
            is_feasible = False
            slots_needed = 0
            spectrum_start = None
            spectrum_end = None
            core_index = None
            band = None
            modulation = None  # Will be set from spectrum result

            # 3. Check spectrum feasibility if we have valid modulations
            if valid_mods:
                spectrum_result = self._spectrum.find_spectrum(
                    path=list(path),
                    modulation=valid_mods,  # Pass all valid mods like orchestrator does
                    bandwidth_gbps=request.bandwidth_gbps,
                    network_state=network_state,
                )
                # Get the modulation that was actually selected
                modulation = spectrum_result.modulation if spectrum_result.is_free else valid_mods[0]

                is_feasible = spectrum_result.is_free
                slots_needed = spectrum_result.slots_needed

                if is_feasible:
                    spectrum_start = spectrum_result.start_slot
                    spectrum_end = spectrum_result.end_slot
                    core_index = spectrum_result.core
                    band = spectrum_result.band

            # 4. Compute congestion metrics
            # TODO: These require NetworkState methods that may not exist yet
            # Using placeholder values for now
            congestion = self._compute_path_congestion(path, network_state)
            available_slots = self._compute_available_slots(path, network_state)

            # 5. Create PathOption
            option = PathOption(
                path_index=path_idx,
                path=path,  # Already a tuple from RouteResult
                weight_km=weight_km,
                num_hops=len(path) - 1,
                modulation=modulation,
                slots_needed=slots_needed,
                is_feasible=is_feasible,
                congestion=congestion,
                available_slots=available_slots,
                spectrum_start=spectrum_start,
                spectrum_end=spectrum_end,
                core_index=core_index,
                band=band,
            )
            options.append(option)

        return options

    def _compute_path_congestion(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> float:
        """
        Compute congestion metric for a path.

        Congestion = max link utilization along the path.

        :param path: Tuple of node IDs representing the path
        :type path: tuple[str, ...]
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: Congestion value in [0, 1]
        :rtype: float
        """
        if len(path) < 2:
            return 0.0

        max_util = 0.0
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            # Try to get utilization from network state
            if hasattr(network_state, "get_link_utilization"):
                util = network_state.get_link_utilization(src, dst)
                max_util = max(max_util, util)
            # Fallback: compute from available slots
            elif hasattr(network_state, "get_available_slots"):
                total = self._config.total_slots
                if total > 0:
                    avail = network_state.get_available_slots(src, dst)
                    util = 1.0 - (avail / total)
                    max_util = max(max_util, util)

        return max_util

    def _compute_available_slots(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> float:
        """
        Compute available slots ratio for a path.

        Returns min(available/total) across all links.

        :param path: Tuple of node IDs representing the path
        :type path: tuple[str, ...]
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: Available slots ratio in [0, 1] where 1 = fully available
        :rtype: float
        """
        if len(path) < 2:
            return 1.0

        total_slots = self._config.total_slots
        if total_slots <= 0:
            return 1.0

        min_ratio = 1.0
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            if hasattr(network_state, "get_available_slots"):
                avail = network_state.get_available_slots(src, dst)
                ratio = avail / total_slots
                min_ratio = min(min_ratio, ratio)

        return min_ratio

    def apply_action(
        self,
        action: int,
        request: Request,
        network_state: NetworkState,
        options: PathOptionList,
    ) -> AllocationResult:
        """
        Apply the selected action via orchestrator.

        This method routes through the SDNOrchestrator with a forced path,
        ensuring all allocation logic (spectrum assignment, SNR validation,
        grooming, slicing) uses the same code paths as non-RL simulation.

        :param action: Index of selected path (corresponds to PathOption.path_index)
        :type action: int
        :param request: Current request to allocate
        :type request: Request
        :param network_state: Current network state
        :type network_state: NetworkState
        :param options: PathOption list from get_path_options()
        :type options: PathOptionList
        :return: AllocationResult from orchestrator indicating success/failure
        :rtype: AllocationResult
        :raises ValueError: If action is negative

        .. note::

            If action doesn't match any PathOption.path_index, returns
            a failed AllocationResult (no paths available for that action).
        """
        from fusion.domain.request import BlockReason
        from fusion.domain.results import AllocationResult

        if action < 0:
            raise ValueError(f"Action must be non-negative, got {action}")

        # Find the corresponding PathOption
        selected_option: PathOption | None = None
        for opt in options:
            if opt.path_index == action:
                selected_option = opt
                break

        if selected_option is None:
            # Action refers to a path that wasn't returned
            # (e.g., fewer paths found than k_paths, or invalid action)
            return AllocationResult(
                success=False,
                block_reason=BlockReason.NO_PATH,
            )

        # Apply via orchestrator with forced path AND forced modulation
        # NOTE: When modulation is None (path too long for routing's mod selection),
        # legacy passes [False] which causes standard allocation to FAIL, then
        # slicing kicks in with its own modulation. We mimic this by passing
        # "INVALID" as forced_modulation when None, which will fail standard
        # allocation and trigger slicing fallback.
        forced_mod = selected_option.modulation
        if forced_mod is None:
            # Pass an invalid modulation to fail standard allocation like legacy's [False]
            forced_mod = "INVALID_MOD_TRIGGER_SLICING"

        result = self._orchestrator.handle_arrival(
            request=request,
            network_state=network_state,
            forced_path=list(selected_option.path),
            forced_modulation=forced_mod,
        )

        return result

    def compute_reward(
        self,
        result: AllocationResult,
        request: Request | None = None,
    ) -> float:
        """
        Compute reward signal from allocation result.

        Reward structure (configurable via RLConfig):

        - Success: +config.rl_success_reward (default: 1.0)
        - Failure: +config.rl_block_penalty (default: -1.0)
        - Grooming bonus: +config.rl_grooming_bonus (default: 0.1)
        - Slicing penalty: +config.rl_slicing_penalty (default: -0.05)

        Optional bandwidth weighting scales reward by request size.

        :param result: AllocationResult from apply_action()
        :type result: AllocationResult
        :param request: Original request (for bandwidth weighting if enabled)
        :type request: Request | None
        :return: Scalar reward value
        :rtype: float
        """
        if not result.success:
            return self._config.rl_block_penalty

        # Legacy bandit algorithms use raw reward/penalty without modifiers
        # Slicing/grooming bonuses and bandwidth weighting are for DRL only
        return self._config.rl_success_reward

    def get_action_mask(self, options: PathOptionList) -> ActionMask:
        """
        Generate action mask from path options.

        Creates a boolean mask where True indicates a valid (feasible) action.
        This is used by action-masked RL algorithms like MaskablePPO.

        :param options: List of PathOption from get_path_options()
        :type options: PathOptionList
        :return: Boolean array of shape (k_paths,) where True = feasible
        :rtype: ActionMask
        """
        return compute_action_mask(options, self._config.k_paths)

    def build_observation(
        self,
        request: Request,
        options: PathOptionList,
        network_state: NetworkState,
    ) -> dict[str, Any]:
        """
        Build observation dictionary for RL agent.

        Constructs observation from domain objects (Request, PathOption,
        NetworkState) without accessing raw numpy arrays or engine_props.

        :param request: Current request
        :type request: Request
        :param options: PathOption list from get_path_options()
        :type options: PathOptionList
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: Dict matching the standard observation space specification
        :rtype: dict[str, Any]
        """
        k = self._config.k_paths
        num_nodes = self._config.num_nodes

        # Request features - one-hot encoded source/destination
        source_onehot = np.zeros(num_nodes, dtype=np.float32)
        dest_onehot = np.zeros(num_nodes, dtype=np.float32)

        # Handle both string and int node identifiers
        src_idx = int(request.source) if isinstance(request.source, str) else request.source
        dst_idx = int(request.destination) if isinstance(request.destination, str) else request.destination

        if 0 <= src_idx < num_nodes:
            source_onehot[src_idx] = 1.0
        if 0 <= dst_idx < num_nodes:
            dest_onehot[dst_idx] = 1.0

        # Normalize holding time to [0, 1]
        max_ht = self._config.max_holding_time
        holding_time = min(request.holding_time / max_ht, 1.0) if max_ht > 0 else 0.0

        # Path features (padded to k_paths)
        slots_needed = np.full(k, -1.0, dtype=np.float32)
        path_lengths = np.zeros(k, dtype=np.float32)
        congestion = np.zeros(k, dtype=np.float32)
        available = np.zeros(k, dtype=np.float32)
        is_feasible = np.zeros(k, dtype=np.float32)

        for opt in options:
            i = opt.path_index
            if i < k:
                slots_needed[i] = float(opt.slots_needed)
                path_lengths[i] = float(opt.num_hops)
                congestion[i] = opt.congestion
                available[i] = opt.available_slots
                is_feasible[i] = 1.0 if opt.is_feasible else 0.0

        return {
            "source": source_onehot,
            "destination": dest_onehot,
            "holding_time": np.array([holding_time], dtype=np.float32),
            "slots_needed": slots_needed,
            "path_lengths": path_lengths,
            "congestion": congestion,
            "available_slots": available,
            "is_feasible": is_feasible,
        }

    def build_offline_state(
        self,
        request: Request,
        options: PathOptionList,
        network_state: NetworkState,
        disaster_state: DisasterState | None = None,
    ) -> dict[str, Any]:
        """
        Build state dict for offline RL policies (BC, IQL).

        This method creates the flattened state dictionary expected by
        BC and IQL policies trained on heuristic behavior logs.

        :param request: Current request
        :type request: Request
        :param options: Available path options
        :type options: PathOptionList
        :param network_state: Current network state
        :type network_state: NetworkState
        :param disaster_state: Optional disaster information
        :type disaster_state: DisasterState | None
        :return: State dict compatible with BCPolicy/IQLPolicy
        :rtype: dict[str, Any]
        """
        state: dict[str, Any] = {
            "src": request.source,
            "dst": request.destination,
            "slots_needed": self._get_min_slots_needed(options),
            "est_remaining_time": self._normalize_holding_time(request.holding_time),
            "is_disaster": 1.0 if disaster_state and disaster_state.active else 0.0,
            "paths": [],
        }

        for opt in options:
            path_features = {
                "path_hops": opt.num_hops,
                "min_residual_slots": self._compute_min_residual(opt.path, network_state),
                "frag_indicator": self._compute_fragmentation(opt.path, network_state),
                "failure_mask": self._compute_failure_mask(opt.path, disaster_state),
                "dist_to_disaster_centroid": self._compute_disaster_distance(
                    opt.path, network_state, disaster_state
                ),
            }
            state["paths"].append(path_features)

        # Pad to k_paths if needed
        while len(state["paths"]) < self._config.k_paths:
            state["paths"].append(self._dummy_path_features())

        return state

    # --- Private helper methods ---

    def _get_min_slots_needed(self, options: PathOptionList) -> float:
        """
        Get minimum slots needed across all feasible paths.

        :param options: List of path options
        :type options: PathOptionList
        :return: Minimum slots needed, or -1.0 if no feasible paths
        :rtype: float
        """
        feasible = [o for o in options if o.is_feasible]
        if not feasible:
            return -1.0
        return float(min(o.slots_needed for o in feasible))

    def _normalize_holding_time(self, holding_time: float) -> float:
        """
        Normalize holding time to [0, 1].

        :param holding_time: Raw holding time value
        :type holding_time: float
        :return: Normalized holding time in [0, 1]
        :rtype: float
        """
        max_ht = self._config.max_holding_time
        if max_ht <= 0:
            return 0.0
        return min(holding_time / max_ht, 1.0)

    def _compute_min_residual(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> float:
        """
        Compute minimum residual slots along path, normalized to [0, 1].

        :param path: Tuple of node IDs representing the path
        :type path: tuple[str, ...]
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: Minimum residual slots normalized to [0, 1]
        :rtype: float
        """
        if len(path) < 2:
            return 0.0

        total_slots = self._config.total_slots
        if total_slots <= 0:
            return 0.0

        min_available = float("inf")
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            # Try to get available slots from network state
            if hasattr(network_state, "get_available_slots"):
                available = network_state.get_available_slots(src, dst)
            else:
                available = total_slots  # Assume full availability
            min_available = min(min_available, available)

        if min_available == float("inf"):
            return 1.0

        return min_available / total_slots

    def _compute_fragmentation(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> float:
        """
        Compute path fragmentation indicator, normalized to [0, 1].

        :param path: Tuple of node IDs representing the path
        :type path: tuple[str, ...]
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: Fragmentation indicator in [0, 1]
        :rtype: float
        """
        # Check if network_state has fragmentation tracker
        if hasattr(network_state, "fragmentation_tracker"):
            tracker = network_state.fragmentation_tracker
            if hasattr(tracker, "get_fragmentation"):
                path_int = [int(n) if isinstance(n, str) else n for n in path]
                frag_dict = tracker.get_fragmentation(path_int, core_index=0)
                return float(frag_dict.get("path_frag", [0.0])[0])

        # Default: no fragmentation tracking available
        return 0.0

    def _compute_failure_mask(
        self,
        path: tuple[str, ...],
        disaster_state: DisasterState | None,
    ) -> float:
        """
        Compute whether path passes through failed links.

        :param path: Tuple of node IDs representing the path
        :type path: tuple[str, ...]
        :param disaster_state: Optional disaster state information
        :type disaster_state: DisasterState | None
        :return: 1.0 if any link is failed, 0.0 otherwise
        :rtype: float
        """
        if disaster_state is None or not disaster_state.active:
            return 0.0

        failed_links = disaster_state.failed_links
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            link_rev = (path[i + 1], path[i])
            if link in failed_links or link_rev in failed_links:
                return 1.0
        return 0.0

    def _compute_disaster_distance(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
        disaster_state: DisasterState | None,
    ) -> float:
        """
        Compute normalized distance from path to disaster centroid.

        :param path: Tuple of node IDs representing the path
        :type path: tuple[str, ...]
        :param network_state: Current network state
        :type network_state: NetworkState
        :param disaster_state: Optional disaster state information
        :type disaster_state: DisasterState | None
        :return: Value in [0, 1] where 0 = at disaster, 1 = max distance
        :rtype: float
        """
        if disaster_state is None or not disaster_state.active:
            return 1.0  # No disaster = max safety

        if disaster_state.centroid is None:
            return 1.0

        # Compute path centroid
        path_centroid = self._compute_path_centroid(path, network_state)
        if path_centroid is None:
            return 1.0

        # Compute distance to disaster centroid
        dx = path_centroid[0] - disaster_state.centroid[0]
        dy = path_centroid[1] - disaster_state.centroid[1]
        distance = (dx**2 + dy**2) ** 0.5

        # Normalize by network diameter
        max_distance = disaster_state.network_diameter
        if max_distance <= 0:
            return 1.0

        return float(min(distance / max_distance, 1.0))

    def _compute_path_centroid(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> tuple[float, float] | None:
        """
        Compute geographic centroid of path nodes.

        :param path: Tuple of node IDs representing the path
        :type path: tuple[str, ...]
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: Centroid coordinates (x, y) or None if unavailable
        :rtype: tuple[float, float] | None
        """
        # Check if network_state has node coordinates
        if not hasattr(network_state, "get_node_coords"):
            return None

        coords = []
        for node in path:
            coord = network_state.get_node_coords(node)
            if coord is not None:
                coords.append(coord)

        if not coords:
            return None

        x = sum(c[0] for c in coords) / len(coords)
        y = sum(c[1] for c in coords) / len(coords)
        return (x, y)

    def _dummy_path_features(self) -> dict[str, float]:
        """
        Return dummy features for padding.

        :return: Dict with dummy path features (masked/infeasible)
        :rtype: dict[str, float]
        """
        return {
            "path_hops": 0,
            "min_residual_slots": 0.0,
            "frag_indicator": 0.0,
            "failure_mask": 1.0,  # Masked = infeasible
            "dist_to_disaster_centroid": 0.0,
        }


class OfflinePolicyAdapter:
    """
    Adapter to use offline RL policies (BC, IQL) with UnifiedSimEnv.

    This adapter bridges the gap between the Gymnasium observation space
    used by online RL and the flattened state format expected by offline
    RL policies (BC, IQL) trained on heuristic behavior logs.

    The offline policies expect a specific state dict format with features
    like src, dst, slots_needed, is_disaster, and per-path features.
    This adapter converts the environment state to that format before
    calling the policy.

    Example::

        # Load offline policy
        bc_policy = BCPolicy("models/bc_model.pt", device="cpu")

        # Create unified environment
        config = SimulationConfig.from_file("config.ini")
        env = UnifiedSimEnv(config)

        # Create offline adapter
        offline_adapter = OfflinePolicyAdapter(
            policy=bc_policy,
            rl_adapter=env.adapter,
        )

        # Run evaluation
        obs, info = env.reset(seed=42)
        while True:
            action = offline_adapter.select_action(
                request=env.current_request,
                options=env.current_options,
                network_state=env.network_state,
                action_mask=info.get("action_mask"),
            )
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    :ivar policy: The offline RL policy (BC or IQL)
    :ivar rl_adapter: RLSimulationAdapter for building offline state
    """

    def __init__(
        self,
        policy: Any,
        rl_adapter: RLSimulationAdapter,
    ) -> None:
        """
        Initialize offline policy adapter.

        :param policy: Offline RL policy with select_path method
        :type policy: Any
        :param rl_adapter: RLSimulationAdapter for state building
        :type rl_adapter: RLSimulationAdapter
        """
        self._policy = policy
        self._rl_adapter = rl_adapter

    @property
    def policy(self) -> Any:
        """
        Access to underlying offline policy.

        :return: The offline RL policy
        :rtype: Any
        """
        return self._policy

    @property
    def rl_adapter(self) -> RLSimulationAdapter:
        """
        Access to RL simulation adapter.

        :return: The RL simulation adapter
        :rtype: RLSimulationAdapter
        """
        return self._rl_adapter

    def select_action(
        self,
        request: Request,
        options: PathOptionList,
        network_state: NetworkState,
        action_mask: ActionMask | list[bool] | None = None,
        disaster_state: DisasterState | None = None,
    ) -> int:
        """
        Select action using offline policy.

        Converts the current state to the offline format expected by
        BC/IQL policies and returns the selected action.

        :param request: Current request
        :type request: Request
        :param options: Path options from adapter
        :type options: PathOptionList
        :param network_state: Network state
        :type network_state: NetworkState
        :param action_mask: Optional action mask (uses feasibility if None)
        :type action_mask: ActionMask | list[bool] | None
        :param disaster_state: Optional disaster state for survivability
        :type disaster_state: DisasterState | None
        :return: Selected action index (path index)
        :rtype: int
        """
        # Build offline state format
        offline_state = self._rl_adapter.build_offline_state(
            request=request,
            options=options,
            network_state=network_state,
            disaster_state=disaster_state,
        )

        # Build action mask if not provided
        if action_mask is None:
            mask_list = [opt.is_feasible for opt in options]
            # Pad to k_paths
            while len(mask_list) < self._rl_adapter.config.k_paths:
                mask_list.append(False)
        elif isinstance(action_mask, np.ndarray):
            mask_list = action_mask.tolist()
        else:
            mask_list = list(action_mask)

        # Call offline policy
        # Most offline policies have a select_path method
        if hasattr(self._policy, "select_path"):
            return int(self._policy.select_path(offline_state, mask_list))
        elif hasattr(self._policy, "predict"):
            # Alternative: some policies use predict
            action, _ = self._policy.predict(offline_state, action_masks=mask_list)
            return int(action)
        elif hasattr(self._policy, "__call__"):
            # Fallback: callable policy
            return int(self._policy(offline_state, mask_list))
        else:
            raise TypeError(
                f"Policy {type(self._policy)} does not have a recognized "
                "action selection method (select_path, predict, or __call__)"
            )


def create_disaster_state_from_engine(
    engine_props: dict[str, Any],
) -> DisasterState | None:
    """
    Create DisasterState from legacy engine_props dict.

    Factory function to convert legacy engine_props dictionary format
    to the new DisasterState dataclass.

    :param engine_props: Legacy engine properties dict containing disaster info
    :type engine_props: dict[str, Any]
    :return: DisasterState if disaster is active, None otherwise
    :rtype: DisasterState | None
    """
    if not engine_props.get("is_disaster", False):
        return None

    # Extract failed links, handling various formats
    failed_links_raw = engine_props.get("failed_links", [])
    failed_links: set[tuple[str, str]] = set()
    for link in failed_links_raw:
        if isinstance(link, (list, tuple)) and len(link) >= 2:
            failed_links.add((str(link[0]), str(link[1])))

    return DisasterState(
        active=True,
        centroid=engine_props.get("disaster_centroid"),
        radius=engine_props.get("disaster_radius", 0.0),
        failed_links=frozenset(failed_links),
        network_diameter=engine_props.get("network_diameter", 1.0),
    )
