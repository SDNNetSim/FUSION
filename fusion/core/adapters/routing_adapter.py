"""
RoutingAdapter - Adapts legacy Routing class to RoutingPipeline protocol.

ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
It will be replaced with a clean implementation in Phase 4.

Phase: P2.4 - Legacy Adapters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fusion.interfaces.pipelines import RoutingPipeline

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.network_state import NetworkState
    from fusion.domain.results import RouteResult

logger = logging.getLogger(__name__)


@dataclass
class SDNPropsProxy:
    """
    Minimal proxy for SDNProps to satisfy legacy Routing class.

    Only implements attributes that Routing.get_route() actually reads.
    This is a read-only proxy - mutations don't persist.
    """

    topology: Any  # nx.Graph
    source: str = ""
    destination: str = ""
    bandwidth: float = 0.0
    network_spectrum_dict: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    lightpath_status_dict: dict[tuple[str, str], dict[int, dict[str, Any]]] = field(
        default_factory=dict
    )
    # Required for modulation format selection in routing algorithms
    modulation_formats_dict: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    # Alias used by some routing algorithms
    mod_formats: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )

    @classmethod
    def from_network_state(
        cls,
        network_state: NetworkState,
        source: str,
        destination: str,
        bandwidth: float,
        modulation_formats_dict: dict[str, dict[str, Any]] | None = None,
    ) -> SDNPropsProxy:
        """Create proxy from NetworkState with request context."""
        mod_dict = modulation_formats_dict or {}
        return cls(
            topology=network_state.topology,
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            network_spectrum_dict=network_state.network_spectrum_dict,
            lightpath_status_dict=network_state.lightpath_status_dict,
            modulation_formats_dict=mod_dict,
            mod_formats=mod_dict,  # Alias for backwards compatibility
        )


class RoutingAdapter(RoutingPipeline):
    """
    Adapts legacy Routing class to RoutingPipeline protocol.

    ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
    It will be replaced with a clean implementation in Phase 4.

    The adapter:
    1. Receives Phase 1 objects (NetworkState, config)
    2. Creates proxy objects for legacy code
    3. Calls legacy Routing.get_route()
    4. Converts route_props to RouteResult

    Removal Checklist:
    [ ] Clean RoutingPipeline implementation exists
    [ ] All callers migrated to clean implementation
    [ ] run_comparison.py passes without this adapter
    [ ] grep 'RoutingAdapter' returns only this definition

    Example:
        >>> config = SimulationConfig.from_engine_props(engine_props)
        >>> adapter = RoutingAdapter(config)
        >>> result = adapter.find_routes("A", "B", 100, network_state)
        >>> print(result.paths)
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize adapter with configuration.

        Args:
            config: SimulationConfig for creating legacy engine_props

        Note:
            Does NOT store NetworkState - receives per-call
        """
        self._config = config
        self._engine_props = config.to_engine_props()

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
        Find candidate routes between source and destination.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            bandwidth_gbps: Required bandwidth
            network_state: Current network state
            forced_path: Optional forced path (from grooming)

        Returns:
            RouteResult containing paths, weights, and modulations
        """
        from fusion.domain.results import RouteResult

        # Handle forced path (from partial grooming)
        if forced_path is not None:
            return self._handle_forced_path(forced_path, network_state)

        try:
            # Get modulation formats dict for routing algorithms
            # Use mod_per_bw for bandwidth-specific modulations, or global modulation_formats
            mod_formats_dict = self._get_modulation_formats_for_bandwidth(bandwidth_gbps)

            # Create proxy for legacy code
            sdn_props = SDNPropsProxy.from_network_state(
                network_state,
                source,
                destination,
                float(bandwidth_gbps),
                modulation_formats_dict=mod_formats_dict,
            )

            # Make a copy of engine_props with topology from network_state
            engine_props = dict(self._engine_props)
            engine_props["topology"] = network_state.topology

            # Import and instantiate legacy Routing
            from fusion.core.routing import Routing

            legacy_routing = Routing(
                engine_props=engine_props,
                sdn_props=sdn_props,
            )

            # Call legacy method
            legacy_routing.get_route()

            # Convert to Phase 1 result
            return self._convert_route_props(legacy_routing.route_props)

        except Exception as e:
            logger.warning("RoutingAdapter.find_routes failed: %s", e)
            return RouteResult.empty("legacy_error")

    def _get_modulation_formats_for_bandwidth(
        self, bandwidth_gbps: int
    ) -> dict[str, dict[str, Any]]:
        """
        Get modulation formats dict for the given bandwidth.

        Checks mod_per_bw first for bandwidth-specific modulations,
        then falls back to global modulation_formats.

        Args:
            bandwidth_gbps: Requested bandwidth in Gbps

        Returns:
            Dict of modulation format name to format info (with max_length key)
        """
        # Try bandwidth-specific modulation formats first
        mod_per_bw = self._config.mod_per_bw
        bw_key = str(bandwidth_gbps)
        if bw_key in mod_per_bw:
            bw_mods = mod_per_bw[bw_key]
            # Convert to format expected by routing: {mod: {"max_length": X}}
            # mod_per_bw structure is {bw: {mod: {slots_needed: X, max_length: Y}}}
            result = {}
            for mod_name, mod_info in bw_mods.items():
                if isinstance(mod_info, dict):
                    result[mod_name] = mod_info
            if result:
                return result

        # Fall back to global modulation formats
        return self._config.modulation_formats

    def _handle_forced_path(
        self,
        forced_path: list[str],
        network_state: NetworkState,
    ) -> RouteResult:
        """
        Handle forced path case (from partial grooming).

        When grooming partially succeeds, it specifies a forced path
        for the new lightpath that must be co-located with groomed traffic.
        """
        from fusion.domain.results import RouteResult

        # Calculate path weight
        weight = self._calculate_path_weight(forced_path, network_state)

        # Get modulation formats for this path length
        modulations = self._get_modulations_for_weight(weight)

        if not modulations:
            # No valid modulations for this path length - return empty result
            logger.warning(
                "No valid modulation formats for forced path with weight %.2f km",
                weight,
            )
            return RouteResult.empty("no_modulation")

        return RouteResult(
            paths=(tuple(forced_path),),
            weights_km=(weight,),
            modulations=(modulations,),
            strategy_name="forced",
        )

    def _calculate_path_weight(
        self,
        path: list[str],
        network_state: NetworkState,
    ) -> float:
        """Calculate total path weight (distance) in km."""
        total = 0.0
        topology = network_state.topology

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if topology.has_edge(u, v):
                edge_data = topology.edges[u, v]
                total += float(
                    edge_data.get("length", edge_data.get("weight", 0.0))
                )

        return total

    def _get_modulations_for_weight(self, weight_km: float) -> tuple[str, ...]:
        """Get valid modulation formats for given path weight."""
        modulations = []
        mod_formats = self._config.modulation_formats

        for mod_name, mod_info in mod_formats.items():
            if isinstance(mod_info, dict):
                # Check both legacy key (max_length) and new key (max_reach_km)
                max_reach = mod_info.get("max_length", mod_info.get("max_reach_km"))
                if max_reach is not None and weight_km <= max_reach:
                    modulations.append(mod_name)

        # Sort by efficiency (higher order first)
        return tuple(sorted(modulations, reverse=True)) if modulations else ()

    def _convert_route_props(self, route_props: Any) -> RouteResult:
        """Convert legacy RoutingProps to RouteResult."""
        from fusion.domain.results import RouteResult

        # Check for empty results
        if not route_props.paths_matrix:
            return RouteResult.empty(
                self._engine_props.get("route_method", "legacy")
            )

        # Convert lists to tuples for immutability
        paths = tuple(
            tuple(str(n) for n in p) for p in route_props.paths_matrix
        )
        weights = tuple(route_props.weights_list)

        # Handle modulation formats
        if route_props.modulation_formats_matrix:
            modulations = tuple(
                tuple(mods) for mods in route_props.modulation_formats_matrix
            )
        else:
            # No modulation formats from routing - this is an error condition
            logger.error(
                "Routing returned paths but no modulation_formats_matrix. "
                "Check that modulation_formats_dict is properly configured."
            )
            return RouteResult.empty("no_modulation_formats")

        # Handle backup paths if present
        backup_paths = None
        backup_weights = None
        backup_mods = None

        # Check for backup paths - must be a list with content (not just truthy MagicMock)
        backup_matrix = getattr(route_props, "backup_paths_matrix", None)
        if isinstance(backup_matrix, list) and backup_matrix:
            backup_paths = tuple(
                tuple(str(n) for n in p) if p else ()
                for p in backup_matrix
            )
            backup_weights_raw = getattr(route_props, "backup_weights_list", None)
            if isinstance(backup_weights_raw, list) and backup_weights_raw:
                backup_weights = tuple(backup_weights_raw)
            backup_mods_raw = getattr(route_props, "backup_modulation_formats_matrix", None)
            if isinstance(backup_mods_raw, list) and backup_mods_raw:
                backup_mods = tuple(
                    tuple(mods) for mods in backup_mods_raw
                )

        strategy_name = self._engine_props.get("route_method", "legacy")

        return RouteResult(
            paths=paths,
            weights_km=weights,
            modulations=modulations,
            backup_paths=backup_paths,
            backup_weights_km=backup_weights,
            backup_modulations=backup_mods,
            strategy_name=strategy_name,
        )
