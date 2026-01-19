"""
Routing strategy pattern for FUSION simulation.

This module defines the RoutingStrategy protocol and concrete implementations
for pluggable routing algorithms. Strategies encapsulate route selection logic
and can be swapped at runtime based on configuration.

Design Pattern:
    Strategy pattern allows different routing algorithms to be used
    interchangeably through a common interface.

Strategies:

- KShortestPathStrategy: Basic k-shortest paths
- LoadBalancedStrategy: Consider link utilization
- ProtectionAwareStrategy: Find disjoint path pairs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fusion.domain.network_state import NetworkState

logger = logging.getLogger(__name__)


# =============================================================================
# Route Constraints
# =============================================================================


@dataclass(frozen=True)
class RouteConstraints:
    """
    Constraints for route selection.

    Used to specify requirements and exclusions when finding routes,
    particularly for protection path computation where the backup
    must avoid links/nodes used by the primary path.

    :ivar max_hops: Maximum number of hops allowed (None = no limit).
    :vartype max_hops: int | None
    :ivar max_length_km: Maximum path length in km (None = no limit).
    :vartype max_length_km: float | None
    :ivar min_bandwidth_gbps: Minimum available bandwidth required.
    :vartype min_bandwidth_gbps: int | None
    :ivar exclude_links: Links to avoid (for disjoint paths).
    :vartype exclude_links: frozenset[tuple[str, str]]
    :ivar exclude_nodes: Nodes to avoid (for node-disjoint paths).
    :vartype exclude_nodes: frozenset[str]
    :ivar required_modulation: Force specific modulation format.
    :vartype required_modulation: str | None
    :ivar protection_mode: True if finding protection/backup path.
    :vartype protection_mode: bool

    Example:
        >>> # Find backup path avoiding primary path links
        >>> constraints = RouteConstraints(
        ...     exclude_links={("A", "B"), ("B", "C")},
        ...     protection_mode=True,
        ... )
    """

    max_hops: int | None = None
    max_length_km: float | None = None
    min_bandwidth_gbps: int | None = None
    exclude_links: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    exclude_nodes: frozenset[str] = field(default_factory=frozenset)
    required_modulation: str | None = None
    protection_mode: bool = False

    def with_exclusions(
        self,
        links: set[tuple[str, str]] | None = None,
        nodes: set[str] | None = None,
    ) -> RouteConstraints:
        """
        Create new constraints with additional exclusions.

        :param links: Additional links to exclude.
        :type links: set[tuple[str, str]] | None
        :param nodes: Additional nodes to exclude.
        :type nodes: set[str] | None
        :return: New RouteConstraints with merged exclusions.
        :rtype: RouteConstraints
        """
        new_links = self.exclude_links
        new_nodes = self.exclude_nodes

        if links:
            new_links = self.exclude_links | frozenset(links)
        if nodes:
            new_nodes = self.exclude_nodes | frozenset(nodes)

        return RouteConstraints(
            max_hops=self.max_hops,
            max_length_km=self.max_length_km,
            min_bandwidth_gbps=self.min_bandwidth_gbps,
            exclude_links=new_links,
            exclude_nodes=new_nodes,
            required_modulation=self.required_modulation,
            protection_mode=self.protection_mode,
        )


# =============================================================================
# Routing Strategy Protocol
# =============================================================================


@runtime_checkable
class RoutingStrategy(Protocol):
    """
    Protocol for routing strategies.

    Defines the interface that all routing strategy implementations
    must follow. Strategies are responsible for selecting candidate
    routes between source and destination nodes.

    Design Notes:
        - Strategies are stateless (configuration passed to __init__)
        - select_routes() is a pure function (no side effects)
        - Returns RouteResult from fusion.domain.results

    Example:
        >>> strategy = KShortestPathStrategy(k=3)
        >>> result = strategy.select_routes(
        ...     "A", "Z", 100, network_state
        ... )
        >>> for path in result.paths:
        ...     print(path)
    """

    @property
    def name(self) -> str:
        """Strategy name for logging and configuration."""
        ...

    def select_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        constraints: RouteConstraints | None = None,
    ) -> RouteResult:
        """
        Select candidate routes for the request.

        :param source: Source node identifier.
        :type source: str
        :param destination: Destination node identifier.
        :type destination: str
        :param bandwidth_gbps: Required bandwidth (for modulation selection).
        :type bandwidth_gbps: int
        :param network_state: Current network state.
        :type network_state: NetworkState
        :param constraints: Optional routing constraints.
        :type constraints: RouteConstraints | None
        :return: RouteResult with ordered candidate routes (best first).
        :rtype: RouteResult
        """
        ...


# =============================================================================
# K-Shortest Path Strategy
# =============================================================================


class KShortestPathStrategy:
    """
    K-shortest paths routing strategy.

    Finds the k shortest paths between source and destination
    using path weight (typically distance in km). This is the
    default routing strategy.

    :ivar k: Number of candidate paths to find.
    :vartype k: int
    :ivar _name: Strategy identifier.
    :vartype _name: str

    Example:
        >>> strategy = KShortestPathStrategy(k=3)
        >>> result = strategy.select_routes("A", "Z", 100, network_state)
    """

    def __init__(self, k: int = 3) -> None:
        """
        Initialize k-shortest path strategy.

        :param k: Number of candidate paths to compute.
        :type k: int
        """
        self.k = k
        self._name = f"k_shortest_{k}"

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    def select_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        constraints: RouteConstraints | None = None,
    ) -> RouteResult:
        """
        Select k-shortest paths between source and destination.

        :param source: Source node identifier.
        :type source: str
        :param destination: Destination node identifier.
        :type destination: str
        :param bandwidth_gbps: Required bandwidth.
        :type bandwidth_gbps: int
        :param network_state: Current network state.
        :type network_state: NetworkState
        :param constraints: Optional routing constraints.
        :type constraints: RouteConstraints | None
        :return: RouteResult with up to k candidate paths.
        :rtype: RouteResult
        """
        # Import here to avoid circular imports
        from fusion.domain.results import RouteResult

        try:
            import networkx as nx
        except ImportError:
            logger.error("NetworkX not available for routing")
            return RouteResult.empty(self._name)

        topology = network_state.topology

        # Apply constraints if provided
        if constraints:
            topology = self._apply_constraints(topology, constraints)

        # Find k-shortest simple paths
        try:
            paths_gen = nx.shortest_simple_paths(topology, source, destination, weight="weight")
            paths: list[tuple[str, ...]] = []
            for path in paths_gen:
                if len(paths) >= self.k:
                    break
                # Apply hop constraint if specified
                if constraints and constraints.max_hops:
                    if len(path) - 1 > constraints.max_hops:
                        continue
                paths.append(tuple(str(n) for n in path))

        except nx.NetworkXNoPath:
            logger.debug(f"No path found from {source} to {destination}")
            return RouteResult.empty(self._name)
        except nx.NodeNotFound as e:
            logger.warning(f"Node not found in topology: {e}")
            return RouteResult.empty(self._name)

        if not paths:
            return RouteResult.empty(self._name)

        # Calculate weights and select modulations
        weights_km: list[float] = []
        modulations: list[tuple[str, ...]] = []

        for path in paths:
            weight = self._calculate_path_weight(path, topology)

            # Apply length constraint
            if constraints and constraints.max_length_km:
                if weight > constraints.max_length_km:
                    continue

            weights_km.append(weight)
            mods = self._select_modulations(weight, bandwidth_gbps, network_state)
            modulations.append(mods)

        # Filter out paths that exceeded constraints
        valid_paths = []
        valid_weights = []
        valid_mods = []
        for i, path in enumerate(paths):
            if i < len(weights_km):
                valid_paths.append(path)
                valid_weights.append(weights_km[i])
                valid_mods.append(modulations[i])

        if not valid_paths:
            return RouteResult.empty(self._name)

        return RouteResult(
            paths=tuple(valid_paths),
            weights_km=tuple(valid_weights),
            modulations=tuple(valid_mods),
            strategy_name=self._name,
        )

    def _apply_constraints(
        self,
        topology: nx.Graph,
        constraints: RouteConstraints,
    ) -> nx.Graph:
        """Apply constraints by filtering topology."""

        # Create a copy to avoid modifying original
        filtered = topology.copy()

        # Remove excluded links
        for u, v in constraints.exclude_links:
            if filtered.has_edge(u, v):
                filtered.remove_edge(u, v)
            if filtered.has_edge(v, u):
                filtered.remove_edge(v, u)

        # Remove excluded nodes
        for node in constraints.exclude_nodes:
            if filtered.has_node(node):
                filtered.remove_node(node)

        return filtered

    def _calculate_path_weight(
        self,
        path: tuple[str, ...],
        topology: nx.Graph,
    ) -> float:
        """Calculate total path weight (distance in km)."""
        total = 0.0
        for i in range(len(path) - 1):
            edge_data = topology.get_edge_data(path[i], path[i + 1])
            if edge_data:
                total += edge_data.get("weight", 1.0)
            else:
                total += 1.0
        return total

    def _select_modulations(
        self,
        weight_km: float,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> tuple[str, ...]:
        """Select valid modulation formats for path distance."""
        # Default modulation formats with reach (km)
        # Higher order formats have shorter reach
        default_reach = {
            "BPSK": 9600.0,
            "QPSK": 4800.0,
            "8-QAM": 2400.0,
            "16-QAM": 1200.0,
            "32-QAM": 600.0,
            "64-QAM": 300.0,
        }

        # Get reach from config if available
        reach_map = default_reach
        if hasattr(network_state, "config"):
            config = network_state.config
            if hasattr(config, "modulation_formats"):
                # Use config modulation reach if available
                for mod_name, mod_info in config.modulation_formats.items():
                    if isinstance(mod_info, dict) and "reach" in mod_info:
                        reach_map[mod_name] = mod_info["reach"]

        # Filter modulations by reach
        valid_mods = []
        for mod, reach in reach_map.items():
            if weight_km <= reach:
                valid_mods.append(mod)

        # Sort by efficiency (higher bandwidth capacity first)
        efficiency_order = [
            "64-QAM",
            "32-QAM",
            "16-QAM",
            "8-QAM",
            "QPSK",
            "BPSK",
        ]
        valid_mods.sort(key=lambda m: efficiency_order.index(m) if m in efficiency_order else 99)

        return tuple(valid_mods) if valid_mods else ("BPSK",)


# =============================================================================
# Load Balanced Strategy
# =============================================================================


class LoadBalancedStrategy:
    """
    Load-balanced routing considering link utilization.

    Extends k-shortest paths by scoring routes based on both
    path length and link utilization, preferring less congested paths.

    :ivar k: Number of candidate paths to consider.
    :vartype k: int
    :ivar utilization_weight: Weight for utilization in scoring (0-1).
    :vartype utilization_weight: float
    :ivar _name: Strategy identifier.
    :vartype _name: str

    Scoring::

        score = (1 - util_weight) * (1/length) + util_weight * (1 - utilization)
        Higher score = better path

    Example:
        >>> strategy = LoadBalancedStrategy(k=5, utilization_weight=0.5)
        >>> result = strategy.select_routes("A", "Z", 100, network_state)
    """

    def __init__(self, k: int = 3, utilization_weight: float = 0.5) -> None:
        """
        Initialize load-balanced strategy.

        :param k: Number of candidate paths to consider.
        :type k: int
        :param utilization_weight: Weight for utilization (0=length only, 1=util only).
        :type utilization_weight: float
        """
        self.k = k
        self.utilization_weight = max(0.0, min(1.0, utilization_weight))
        self._name = "load_balanced"

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    def select_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        constraints: RouteConstraints | None = None,
    ) -> RouteResult:
        """
        Select routes balancing path length and utilization.

        :param source: Source node identifier.
        :type source: str
        :param destination: Destination node identifier.
        :type destination: str
        :param bandwidth_gbps: Required bandwidth.
        :type bandwidth_gbps: int
        :param network_state: Current network state.
        :type network_state: NetworkState
        :param constraints: Optional routing constraints.
        :type constraints: RouteConstraints | None
        :return: RouteResult with routes sorted by combined score.
        :rtype: RouteResult
        """
        from fusion.domain.results import RouteResult

        # First get k-shortest paths
        ksp = KShortestPathStrategy(k=self.k * 2)  # Get more candidates
        base_result = ksp.select_routes(source, destination, bandwidth_gbps, network_state, constraints)

        if base_result.is_empty:
            return RouteResult.empty(self._name)

        # Score each path by utilization
        scored_paths: list[tuple[float, int]] = []
        for i, path in enumerate(base_result.paths):
            utilization = self._calculate_path_utilization(path, network_state)
            weight_km = base_result.weights_km[i]

            # Combined score (higher is better)
            length_score = 1.0 / max(weight_km, 1.0)
            util_score = 1.0 - utilization

            score = (1.0 - self.utilization_weight) * length_score + self.utilization_weight * util_score
            scored_paths.append((score, i))

        # Sort by score (descending) and take top k
        scored_paths.sort(reverse=True)
        top_indices = [idx for _, idx in scored_paths[: self.k]]

        # Build result with reordered paths
        paths = tuple(base_result.paths[i] for i in top_indices)
        weights = tuple(base_result.weights_km[i] for i in top_indices)
        mods = tuple(base_result.modulations[i] for i in top_indices)

        return RouteResult(
            paths=paths,
            weights_km=weights,
            modulations=mods,
            strategy_name=self._name,
        )

    def _calculate_path_utilization(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> float:
        """Calculate average link utilization along path."""
        if len(path) < 2:
            return 0.0

        total_util = 0.0
        link_count = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            util = network_state.get_link_utilization(link)
            total_util += util
            link_count += 1

        return total_util / link_count if link_count > 0 else 0.0


# =============================================================================
# Protection Aware Strategy
# =============================================================================


class ProtectionAwareStrategy:
    """
    Strategy for finding disjoint working/protection paths.

    Used for 1+1 protection scenarios where the backup path must
    be disjoint from the primary path (no shared links or nodes).

    :ivar node_disjoint: If True, paths must be node-disjoint (stricter).
    :vartype node_disjoint: bool
    :ivar _name: Strategy identifier.
    :vartype _name: str

    Usage:

    1. First call with protection_mode=False to get working path
    2. Second call with protection_mode=True and exclude_links set
       to working path's links

    Example:
        >>> strategy = ProtectionAwareStrategy(node_disjoint=True)
        >>> # Get working path
        >>> working = strategy.select_routes("A", "Z", 100, state)
        >>> # Get protection path avoiding working path
        >>> constraints = RouteConstraints(
        ...     exclude_links=get_path_links(working.best_path),
        ...     protection_mode=True,
        ... )
        >>> backup = strategy.select_routes("A", "Z", 100, state, constraints)
    """

    def __init__(self, node_disjoint: bool = False) -> None:
        """
        Initialize protection-aware strategy.

        :param node_disjoint: If True, require node-disjoint paths.
        :type node_disjoint: bool
        """
        self.node_disjoint = node_disjoint
        self._name = "protection_aware"

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    def select_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        constraints: RouteConstraints | None = None,
    ) -> RouteResult:
        """
        Select routes for protection scenarios.

        When protection_mode=True in constraints, finds paths avoiding
        the excluded links/nodes (typically the working path).

        :param source: Source node identifier.
        :type source: str
        :param destination: Destination node identifier.
        :type destination: str
        :param bandwidth_gbps: Required bandwidth.
        :type bandwidth_gbps: int
        :param network_state: Current network state.
        :type network_state: NetworkState
        :param constraints: Constraints including exclusions for protection.
        :type constraints: RouteConstraints | None
        :return: RouteResult with candidate paths.
        :rtype: RouteResult
        """
        from fusion.domain.results import RouteResult

        # Use k-shortest with appropriate constraints
        ksp = KShortestPathStrategy(k=3)

        if constraints and constraints.protection_mode:
            # Finding protection path - ensure we have exclusions
            if self.node_disjoint and constraints.exclude_links:
                # Extract nodes from excluded links (except source/dest)
                nodes_to_exclude: set[str] = set()
                for u, v in constraints.exclude_links:
                    if u != source and u != destination:
                        nodes_to_exclude.add(u)
                    if v != source and v != destination:
                        nodes_to_exclude.add(v)

                constraints = constraints.with_exclusions(nodes=nodes_to_exclude)

        result = ksp.select_routes(source, destination, bandwidth_gbps, network_state, constraints)

        # Update strategy name
        if not result.is_empty:
            return RouteResult(
                paths=result.paths,
                weights_km=result.weights_km,
                modulations=result.modulations,
                strategy_name=self._name,
            )

        return RouteResult.empty(self._name)

    def find_disjoint_pair(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> tuple[RouteResult, RouteResult]:
        """
        Find a pair of disjoint paths (working + protection).

        Convenience method that finds both paths in one call.

        :param source: Source node identifier.
        :type source: str
        :param destination: Destination node identifier.
        :type destination: str
        :param bandwidth_gbps: Required bandwidth.
        :type bandwidth_gbps: int
        :param network_state: Current network state.
        :type network_state: NetworkState
        :return: Tuple of (working_result, protection_result).
        :rtype: tuple[RouteResult, RouteResult]
        """
        from fusion.domain.results import RouteResult

        # Find working path
        working = self.select_routes(source, destination, bandwidth_gbps, network_state)

        if working.is_empty or working.best_path is None:
            return working, RouteResult.empty(self._name)

        # Get links from working path
        working_path = working.best_path
        exclude_links: set[tuple[str, str]] = set()
        for i in range(len(working_path) - 1):
            exclude_links.add((working_path[i], working_path[i + 1]))
            exclude_links.add((working_path[i + 1], working_path[i]))

        # Find protection path
        constraints = RouteConstraints(
            exclude_links=frozenset(exclude_links),
            protection_mode=True,
        )

        protection = self.select_routes(source, destination, bandwidth_gbps, network_state, constraints)

        return working, protection


# Import RouteResult for type hints at module level
# This is done at the end to avoid circular imports
if TYPE_CHECKING:
    import networkx as nx

    from fusion.domain.results import RouteResult
