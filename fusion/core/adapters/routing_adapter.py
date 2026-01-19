"""
RoutingAdapter - Adapts legacy Routing class to RoutingPipeline protocol.

ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
It will be replaced with a clean implementation in v6.1.0.
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
    network_spectrum_dict: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    lightpath_status_dict: dict[tuple[str, str], dict[int, dict[str, Any]]] = field(default_factory=dict)
    # Required for modulation format selection in routing algorithms
    modulation_formats_dict: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Alias used by some routing algorithms
    mod_formats: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_network_state(
        cls,
        network_state: NetworkState,
        source: str,
        destination: str,
        bandwidth: float,
        modulation_formats_dict: dict[str, dict[str, Any]] | None = None,
    ) -> SDNPropsProxy:
        """
        Create proxy from NetworkState with request context.

        :param network_state: Current network state
        :type network_state: NetworkState
        :param source: Source node identifier
        :type source: str
        :param destination: Destination node identifier
        :type destination: str
        :param bandwidth: Requested bandwidth in Gbps
        :type bandwidth: float
        :param modulation_formats_dict: Modulation format configurations
        :type modulation_formats_dict: dict[str, dict[str, Any]] | None
        :return: Proxy instance populated from network state
        :rtype: SDNPropsProxy
        """
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
    It will be replaced with a clean implementation in v6.1.0.

    The adapter:
    1. Receives Phase 1 objects (NetworkState, config)
    2. Creates proxy objects for legacy code
    3. Calls legacy Routing.get_route()
    4. Converts route_props to RouteResult

    Example:
        >>> config = SimulationConfig.from_engine_props(engine_props)
        >>> adapter = RoutingAdapter(config)
        >>> result = adapter.find_routes("A", "B", 100, network_state)
        >>> print(result.paths)
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize adapter with configuration.

        Does NOT store NetworkState - receives per-call.

        :param config: SimulationConfig for creating legacy engine_props
        :type config: SimulationConfig
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

        :param source: Source node identifier
        :type source: str
        :param destination: Destination node identifier
        :type destination: str
        :param bandwidth_gbps: Required bandwidth in Gbps
        :type bandwidth_gbps: int
        :param network_state: Current network state
        :type network_state: NetworkState
        :param forced_path: Optional forced path (from grooming)
        :type forced_path: list[str] | None
        :return: RouteResult containing paths, weights, and modulations
        :rtype: RouteResult
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

    def _get_modulation_formats_for_bandwidth(self, bandwidth_gbps: int) -> dict[str, dict[str, Any]]:
        """
        Get modulation formats dict for the given bandwidth.

        Checks mod_per_bw first for bandwidth-specific modulations,
        then falls back to global modulation_formats.

        :param bandwidth_gbps: Requested bandwidth in Gbps
        :type bandwidth_gbps: int
        :return: Dict of modulation format name to format info (with max_length key)
        :rtype: dict[str, dict[str, Any]]
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

        Legacy behavior: return ALL modulation formats and let spectrum
        assignment determine which one works. This is important because
        the existing lightpath already has a valid modulation.

        :param forced_path: Path forced by partial grooming
        :type forced_path: list[str]
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: RouteResult with the forced path and all modulation formats
        :rtype: RouteResult
        """
        from fusion.domain.results import RouteResult

        # Calculate path weight
        weight = self._calculate_path_weight(forced_path, network_state)

        # For forced paths (partial grooming), return ALL modulation formats
        # Legacy behavior: spectrum assignment determines the valid modulation
        modulations = self._get_all_modulation_names()

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
        """
        Calculate total path weight (distance) in km.

        :param path: List of node identifiers forming the path
        :type path: list[str]
        :param network_state: Current network state with topology
        :type network_state: NetworkState
        :return: Total path distance in kilometers
        :rtype: float
        """
        total = 0.0
        topology = network_state.topology

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if topology.has_edge(u, v):
                edge_data = topology.edges[u, v]
                total += float(edge_data.get("length", edge_data.get("weight", 0.0)))

        return total

    def _get_modulations_for_weight(self, weight_km: float) -> tuple[str, ...]:
        """
        Get valid modulation formats for given path weight.

        :param weight_km: Path weight (distance) in kilometers
        :type weight_km: float
        :return: Tuple of valid modulation format names
        :rtype: tuple[str, ...]
        """
        modulations = []
        mod_formats = self._config.modulation_formats

        for mod_name, mod_info in mod_formats.items():
            if isinstance(mod_info, dict):
                # Check both legacy key (max_length) and new key (max_reach_km)
                max_reach = mod_info.get("max_length", mod_info.get("max_reach_km"))
                if max_reach is not None and weight_km <= max_reach:
                    modulations.append(mod_name)

        # Sort by efficiency (higher bits_per_symbol first)
        return self._sort_modulations_by_efficiency(modulations)

    def _get_all_modulation_names(self) -> tuple[str, ...]:
        """
        Get all modulation format names without filtering by weight.

        Used for forced paths (partial grooming) where spectrum assignment
        determines which modulation works based on the existing lightpath.

        :return: Tuple of all modulation format names sorted by efficiency
        :rtype: tuple[str, ...]
        """
        # First try global modulation_formats
        mod_formats = self._config.modulation_formats
        modulations = [name for name, info in mod_formats.items() if isinstance(info, dict)]

        # If empty, collect from mod_per_bw (fixed_grid mode)
        if not modulations:
            mod_per_bw = self._config.mod_per_bw
            # Use a list to preserve deterministic order (avoid set() which has non-deterministic iteration)
            seen: set[str] = set()
            for bw_mods in mod_per_bw.values():
                if isinstance(bw_mods, dict):
                    for mod_name, mod_info in bw_mods.items():
                        if isinstance(mod_info, dict) and mod_name not in seen:
                            modulations.append(mod_name)
                            seen.add(mod_name)

        # Sort by efficiency (higher bits_per_symbol first)
        return self._sort_modulations_by_efficiency(modulations)

    def _sort_modulations_by_efficiency(self, modulations: list[str]) -> tuple[str, ...]:
        """
        Sort modulation formats by efficiency in descending order.

        Uses max_length (ascending = higher efficiency) to match legacy behavior,
        or falls back to bits_per_symbol or name-based inference.

        Higher-order modulations (e.g., 64-QAM) should be tried first as they
        use fewer spectrum slots for the same bandwidth.

        :param modulations: List of modulation format names
        :type modulations: list[str]
        :return: Tuple of modulation names sorted by efficiency (descending)
        :rtype: tuple[str, ...]
        """
        if not modulations:
            return ()

        # Get sorting key for each modulation from config
        mod_formats = self._config.modulation_formats
        mod_per_bw = self._config.mod_per_bw

        def get_sort_key(mod_name: str) -> tuple[int, int]:
            """
            Return (max_length, -bits_per_symbol) for sorting.

            Lower max_length = higher efficiency (tried first).
            This matches legacy sort_nested_dict_values behavior.
            """
            max_length = None
            bits_per_symbol = None

            # Try global modulation_formats first
            if mod_name in mod_formats:
                info = mod_formats[mod_name]
                if isinstance(info, dict):
                    max_length = info.get("max_length")
                    bits_per_symbol = info.get("bits_per_symbol")

            # Try mod_per_bw (check first bandwidth key that has this mod)
            if max_length is None:
                for bw_mods in mod_per_bw.values():
                    if isinstance(bw_mods, dict) and mod_name in bw_mods:
                        info = bw_mods[mod_name]
                        if isinstance(info, dict):
                            max_length = info.get("max_length")
                            bits_per_symbol = info.get("bits_per_symbol")
                            break

            # If we have max_length, use it (lower = better = sorted first)
            if max_length is not None:
                return (max_length, 0)

            # Fallback to bits_per_symbol (higher = better, so negate for ascending sort)
            if bits_per_symbol is not None:
                return (0, -bits_per_symbol)

            # TODO: Remove this hardcoded name-based inference. Modulation format sorting
            # should be driven entirely by configuration (max_length or bits_per_symbol).
            # If config is missing this data, it should be fixed at the config level,
            # not guessed from string patterns. This is brittle and will break for
            # non-standard naming conventions.
            name_upper = mod_name.upper()
            if "64-QAM" in name_upper or "64QAM" in name_upper:
                return (0, -6)
            if "32-QAM" in name_upper or "32QAM" in name_upper:
                return (0, -5)
            if "16-QAM" in name_upper or "16QAM" in name_upper:
                return (0, -4)
            if "8-QAM" in name_upper or "8QAM" in name_upper:
                return (0, -3)
            if "QPSK" in name_upper:
                return (0, -2)
            if "BPSK" in name_upper:
                return (0, -1)
            return (999999, 0)

        # Sort by key (ascending max_length means higher efficiency first)
        return tuple(sorted(modulations, key=get_sort_key))

    def _convert_route_props(self, route_props: Any) -> RouteResult:
        """
        Convert legacy RoutingProps to RouteResult.

        :param route_props: Legacy RoutingProps object with paths_matrix,
            weights_list, and modulation_formats_matrix
        :type route_props: Any
        :return: Converted RouteResult
        :rtype: RouteResult
        """
        from fusion.domain.results import RouteResult

        # Check for empty results
        if not route_props.paths_matrix:
            return RouteResult.empty(self._engine_props.get("route_method", "legacy"))

        # Convert lists to tuples for immutability
        paths = tuple(tuple(str(n) for n in p) for p in route_props.paths_matrix)
        weights = tuple(route_props.weights_list)

        # Handle modulation formats
        if route_props.modulation_formats_matrix:
            modulations = tuple(tuple(mods) for mods in route_props.modulation_formats_matrix)
        else:
            # No modulation formats from routing - this is an error condition
            logger.error(
                "Routing returned paths but no modulation_formats_matrix. Check that modulation_formats_dict is properly configured."
            )
            return RouteResult.empty("no_modulation_formats")

        # Handle backup paths if present
        backup_paths = None
        backup_weights = None
        backup_mods = None

        # Check for backup paths - must be a list with content (not just truthy MagicMock)
        backup_matrix = getattr(route_props, "backup_paths_matrix", None)
        if isinstance(backup_matrix, list) and backup_matrix:
            backup_paths = tuple(tuple(str(n) for n in p) if p else () for p in backup_matrix)
            backup_weights_raw = getattr(route_props, "backup_weights_list", None)
            if isinstance(backup_weights_raw, list) and backup_weights_raw:
                backup_weights = tuple(backup_weights_raw)
            backup_mods_raw = getattr(route_props, "backup_modulation_formats_matrix", None)
            if isinstance(backup_mods_raw, list) and backup_mods_raw:
                backup_mods = tuple(tuple(mods) for mods in backup_mods_raw)

        strategy_name = self._engine_props.get("route_method", "legacy")

        # Capture connection_index for external SNR lookup
        connection_index = getattr(route_props, "connection_index", None)

        return RouteResult(
            paths=paths,
            weights_km=weights,
            modulations=modulations,
            backup_paths=backup_paths,
            backup_weights_km=backup_weights,
            backup_modulations=backup_mods,
            strategy_name=strategy_name,
            connection_index=connection_index,
        )
