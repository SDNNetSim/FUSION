"""
Protected routing pipeline implementation.

This module provides ProtectedRoutingPipeline for 1+1 protection
routing scenarios where both working and backup paths are needed.

Phase: P3.1 - Pipeline Factory Scaffolding
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fusion.pipelines.routing_strategies import (
    ProtectionAwareStrategy,
    RouteConstraints,
)

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.network_state import NetworkState
    from fusion.domain.results import RouteResult

logger = logging.getLogger(__name__)


class ProtectedRoutingPipeline:
    """
    Routing pipeline with 1+1 protection support.

    This pipeline finds both working and protection paths that are
    link-disjoint (or node-disjoint based on configuration). It uses
    ProtectionAwareStrategy to ensure backup paths avoid working path.

    Attributes:
        _config: Simulation configuration
        _strategy: Protection-aware routing strategy
        _node_disjoint: Whether to require node-disjoint paths

    Usage:
        >>> pipeline = ProtectedRoutingPipeline(config)
        >>> result = pipeline.find_routes("A", "Z", 100, network_state)
        >>> if result.has_protection:
        ...     print(f"Working: {result.best_path}")
        ...     print(f"Backup: {result.backup_paths[0]}")

    Phase: P3.1 - Pipeline Factory Scaffolding
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize protected routing pipeline.

        Args:
            config: Simulation configuration
        """
        self._config = config

        # Determine if node-disjoint protection is required
        self._node_disjoint = getattr(config, "node_disjoint_protection", False)

        # Create protection-aware strategy
        self._strategy = ProtectionAwareStrategy(
            node_disjoint=self._node_disjoint
        )

        logger.debug(
            f"ProtectedRoutingPipeline initialized "
            f"(node_disjoint={self._node_disjoint})"
        )

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        """
        Find working and protection paths between source and destination.

        For 1+1 protection, this method finds both the primary (working)
        path and a disjoint backup (protection) path. Both paths are
        returned in the RouteResult.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            bandwidth_gbps: Required bandwidth (for modulation selection)
            network_state: Current network state
            forced_path: If provided, use as working path (from grooming)

        Returns:
            RouteResult containing:
            - paths: Working path candidates
            - backup_paths: Protection path for each working path
            - weights_km, modulations for both paths

        Notes:
            - Returns empty RouteResult if no disjoint pair can be found
            - backup_paths[i] corresponds to protection for paths[i]
        """
        from fusion.domain.results import RouteResult

        # Handle forced path (from partial grooming)
        if forced_path is not None:
            return self._handle_forced_path(
                forced_path, source, destination, bandwidth_gbps, network_state
            )

        # Find disjoint path pair
        working, protection = self._strategy.find_disjoint_pair(
            source, destination, bandwidth_gbps, network_state
        )

        # Check if we got valid paths
        if working.is_empty:
            logger.debug(f"No working path found from {source} to {destination}")
            return RouteResult.empty("protected_routing")

        if protection.is_empty:
            logger.warning(
                f"No protection path found from {source} to {destination}. "
                f"Working path exists but protection unavailable."
            )
            # Return working path without protection
            # Caller must decide whether to proceed without protection
            return RouteResult(
                paths=working.paths,
                weights_km=working.weights_km,
                modulations=working.modulations,
                backup_paths=None,
                backup_weights_km=None,
                backup_modulations=None,
                strategy_name="protected_routing",
            )

        # Build result with both paths
        # For simplicity, we pair first working with first protection
        return RouteResult(
            paths=working.paths[:1],  # Just the best working path
            weights_km=working.weights_km[:1],
            modulations=working.modulations[:1],
            backup_paths=protection.paths[:1],  # Just the best protection path
            backup_weights_km=protection.weights_km[:1],
            backup_modulations=protection.modulations[:1],
            strategy_name="protected_routing",
        )

    def _handle_forced_path(
        self,
        forced_path: list[str],
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> RouteResult:
        """
        Handle forced working path from grooming.

        When a request is partially groomed, the remaining bandwidth
        must use the same path as the groomed portion. This method
        finds a protection path for that forced working path.

        Args:
            forced_path: The working path to use
            source: Source node identifier
            destination: Destination node identifier
            bandwidth_gbps: Required bandwidth
            network_state: Current network state

        Returns:
            RouteResult with forced working path and computed protection
        """
        from fusion.domain.results import RouteResult

        # Get links from forced path
        path_tuple = tuple(str(n) for n in forced_path)
        exclude_links: set[tuple[str, str]] = set()
        for i in range(len(path_tuple) - 1):
            exclude_links.add((path_tuple[i], path_tuple[i + 1]))
            exclude_links.add((path_tuple[i + 1], path_tuple[i]))

        # Calculate weight and modulations for forced path
        weight_km = self._calculate_path_weight(path_tuple, network_state)
        modulations = self._select_modulations(weight_km, network_state)

        # Find protection path avoiding forced path
        constraints = RouteConstraints(
            exclude_links=frozenset(exclude_links),
            protection_mode=True,
        )
        protection = self._strategy.select_routes(
            source, destination, bandwidth_gbps, network_state, constraints
        )

        if protection.is_empty:
            logger.warning(
                f"No protection path for forced path from {source} to {destination}"
            )
            return RouteResult(
                paths=(path_tuple,),
                weights_km=(weight_km,),
                modulations=(modulations,),
                backup_paths=None,
                strategy_name="protected_routing",
            )

        return RouteResult(
            paths=(path_tuple,),
            weights_km=(weight_km,),
            modulations=(modulations,),
            backup_paths=protection.paths[:1],
            backup_weights_km=protection.weights_km[:1],
            backup_modulations=protection.modulations[:1],
            strategy_name="protected_routing",
        )

    def _calculate_path_weight(
        self,
        path: tuple[str, ...],
        network_state: NetworkState,
    ) -> float:
        """Calculate path weight (distance) in km."""
        topology = network_state.topology
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
        network_state: NetworkState,
    ) -> tuple[str, ...]:
        """Select valid modulation formats for path distance."""
        # Default modulation reach
        default_reach = {
            "BPSK": 9600.0,
            "QPSK": 4800.0,
            "8-QAM": 2400.0,
            "16-QAM": 1200.0,
            "32-QAM": 600.0,
            "64-QAM": 300.0,
        }

        valid_mods = [mod for mod, reach in default_reach.items() if weight_km <= reach]

        efficiency_order = ["64-QAM", "32-QAM", "16-QAM", "8-QAM", "QPSK", "BPSK"]
        valid_mods.sort(
            key=lambda m: efficiency_order.index(m) if m in efficiency_order else 99
        )

        return tuple(valid_mods) if valid_mods else ("BPSK",)
