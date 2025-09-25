"""
Non-linear impairment (NLI) aware routing algorithm implementation.
"""
# pylint: disable=duplicate-code

from typing import Any

import networkx as nx

from fusion.core.properties import RoutingProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers
from fusion.sim.utils import find_path_length, get_path_modulation


class NLIAwareRouting(AbstractRoutingAlgorithm):
    """NLI-aware routing algorithm.

    This algorithm finds paths by considering non-linear impairments,
    selecting the path with the least amount of NLI.
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        """Initialize NLI-aware routing algorithm.

        Args:
            engine_props: Dictionary containing engine configuration
            sdn_props: Object containing SDN controller properties
        """
        super().__init__(engine_props, sdn_props)
        self.route_props = RoutingProps()
        self.route_help_obj = RoutingHelpers(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            route_props=self.route_props,
        )
        self._path_count = 0
        self._total_nli = 0

    @property
    def algorithm_name(self) -> str:
        """Return the name of the routing algorithm."""
        return "nli_aware"

    @property
    def supported_topologies(self) -> list[str]:
        """Return list of supported topology types."""
        return ["NSFNet", "USBackbone60", "Pan-European", "Generic"]

    def validate_environment(self, topology: Any) -> bool:
        """Validate that the routing algorithm can work with the given topology.

        Args:
            topology: NetworkX graph representing the network topology

        Returns:
            True if the algorithm can route in this environment
        """
        return (
            hasattr(topology, "nodes")
            and hasattr(topology, "edges")
            and hasattr(self.sdn_props, "network_spectrum_dict")
        )

    def route(self, source: Any, destination: Any, request: Any) -> list[Any] | None:
        """Find a route from source to destination considering NLI.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details

        Returns:
            Path with least NLI, or None if no path found
        """
        # Store source/destination in sdn_props for compatibility
        self.sdn_props.source = source
        self.sdn_props.destination = destination

        # Reset paths matrix for new calculation
        self.route_props.paths_matrix = []
        self.route_props.weights_list = []
        self.route_props.modulation_formats_matrix = []

        try:
            # Update NLI costs for all links
            self._update_nli_costs()

            # Find least NLI path
            self._find_least_weight("nli_cost")

            path = None
            if self.route_props.paths_matrix:
                path = self.route_props.paths_matrix[0]
                self._path_count += 1

                # Calculate NLI metric for this path
                nli = self._calculate_path_nli(path)
                self._total_nli += nli

            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _update_nli_costs(self):
        """Update NLI costs for all links in the topology."""
        topology = self.engine_props.get("topology", self.sdn_props.topology)

        # Bidirectional links are identical, therefore, we don't have to check each one
        for link_tuple in list(self.sdn_props.network_spectrum_dict.keys())[::2]:
            source_node, destination_node = link_tuple[0], link_tuple[1]
            span_count = (
                topology[source_node][destination_node]["length"]
                / self.route_props.span_length
            )
            connection_bandwidth = self.sdn_props.bandwidth

            # Get slots needed for bandwidth (using QPSK as default)
            if (
                hasattr(self.engine_props, "mod_per_bw")
                and connection_bandwidth in self.engine_props["mod_per_bw"]
            ):
                required_slots = (
                    self.engine_props["mod_per_bw"][connection_bandwidth]
                    .get("QPSK", {})
                    .get("slots_needed", 1)
                )
            else:
                required_slots = 1

            self.sdn_props.slots_needed = required_slots

            nli_link_cost = self.route_help_obj.get_nli_cost(
                link_tuple=link_tuple, num_span=span_count
            )

            if hasattr(topology, "edges"):
                topology[source_node][destination_node]["nli_cost"] = nli_link_cost
                topology[destination_node][source_node]["nli_cost"] = nli_link_cost

    def _find_least_weight(self, weight: str):
        """Find the path with least weight (NLI cost)."""
        topology = self.engine_props.get("topology", self.sdn_props.topology)

        paths_generator = nx.shortest_simple_paths(
            G=topology,
            source=self.sdn_props.source,
            target=self.sdn_props.destination,
            weight=weight,
        )

        for path_list in paths_generator:
            # Calculate path weight as sum across the path
            path_weight = sum(
                topology[path_list[i]][path_list[i + 1]][weight]
                for i in range(len(path_list) - 1)
            )

            # Calculate actual path length for modulation format selection
            path_length = find_path_length(path_list=path_list, topology=topology)
            # Get modulation format
            modulation_formats = getattr(self.sdn_props, "mod_formats", {})
            modulation_format = get_path_modulation(modulation_formats, path_length)
            modulation_format_list = [
                modulation_format if modulation_format else "QPSK"
            ]

            self.route_props.weights_list.append(path_weight)
            self.route_props.paths_matrix.append(path_list)
            self.route_props.modulation_formats_matrix.append(modulation_format_list)
            # For NLI-aware, we typically take the first (best) path
            break

    def _calculate_path_nli(self, path: list[Any]) -> float:
        """Calculate NLI metric for a path."""
        if not path or len(path) < 2:
            return 0.0

        topology = self.engine_props.get("topology", self.sdn_props.topology)
        total_nli = 0.0

        for link_index in range(len(path) - 1):
            source_node, destination_node = path[link_index], path[link_index + 1]
            if hasattr(topology, "edges") and topology.has_edge(
                source_node, destination_node
            ):
                link_nli_cost = topology[source_node][destination_node].get(
                    "nli_cost", 0.0
                )
                total_nli += link_nli_cost

        return total_nli

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """Get k paths ordered by NLI level.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            k: Number of paths to return

        Returns:
            List of k paths ordered by NLI (least NLI first)
        """
        # Update NLI costs first
        self._update_nli_costs()
        topology = self.engine_props.get("topology", self.sdn_props.topology)

        try:
            paths_generator = nx.shortest_simple_paths(
                topology, source, destination, weight="nli_cost"
            )

            paths_list = []
            for path_index, path in enumerate(paths_generator):
                if path_index >= k:
                    break
                paths_list.append(list(path))

            return paths_list

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def update_weights(self, topology: Any) -> None:
        """Update NLI weights based on current network state.

        Args:
            topology: NetworkX graph to update weights for
        """
        # Recalculate NLI costs for all links
        for link_tuple in list(self.sdn_props.network_spectrum_dict.keys())[::2]:
            source_node, destination_node = link_tuple[0], link_tuple[1]
            span_count = (
                topology[source_node][destination_node]["length"]
                / self.route_props.span_length
            )

            nli_link_cost = self.route_help_obj.get_nli_cost(
                link_tuple=link_tuple, num_span=span_count
            )

            if hasattr(topology, "edges"):
                topology[source_node][destination_node]["nli_cost"] = nli_link_cost
                topology[destination_node][source_node]["nli_cost"] = nli_link_cost

    def get_metrics(self) -> dict[str, Any]:
        """Get routing algorithm performance metrics.

        Returns:
            Dictionary containing algorithm-specific metrics
        """
        avg_nli = self._total_nli / self._path_count if self._path_count > 0 else 0

        return {
            "algorithm": self.algorithm_name,
            "paths_computed": self._path_count,
            "average_nli": avg_nli,
            "total_nli_considered": self._total_nli,
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_nli = 0
        self.route_props = RoutingProps()
