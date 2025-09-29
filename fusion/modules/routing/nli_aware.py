"""
Non-linear impairment (NLI) aware routing algorithm implementation.
"""

from typing import Any

import networkx as nx

from fusion.core.properties import RoutingProps, SDNProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers
from fusion.sim.utils import find_path_length, get_path_modulation


class NLIAwareRouting(AbstractRoutingAlgorithm):
    """
    NLI-aware routing algorithm.

    This algorithm finds paths by considering non-linear impairments,
    selecting the path with the least amount of NLI.
    """

    def __init__(self, engine_props: dict[str, Any], sdn_props: SDNProps) -> None:
        """
        Initialize NLI-aware routing algorithm.

        :param engine_props: Dictionary containing engine configuration.
        :type engine_props: dict[str, Any]
        :param sdn_props: Object containing SDN controller properties.
        :type sdn_props: Any
        """
        super().__init__(engine_props, sdn_props)
        self.route_props = RoutingProps()
        self.route_help_obj = RoutingHelpers(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            route_props=self.route_props,
        )
        self._path_count = 0
        self._total_nli = 0.0

    @property
    def algorithm_name(self) -> str:
        """
        Get the name of the routing algorithm.

        :return: The algorithm name 'nli_aware'.
        :rtype: str
        """
        return "nli_aware"

    @property
    def supported_topologies(self) -> list[str]:
        """
        Get the list of supported topology types.

        :return: List of supported topology names including NSFNet,
            USBackbone60, Pan-European, and Generic.
        :rtype: list[str]
        """
        return ["NSFNet", "USBackbone60", "Pan-European", "Generic"]

    def validate_environment(self, topology: Any) -> bool:
        """
        Validate that the routing algorithm can work with the given topology.

        :param topology: NetworkX graph representing the network topology.
        :type topology: Any
        :return: True if the algorithm can route in this environment.
        :rtype: bool
        """
        return (
            hasattr(topology, "nodes")
            and hasattr(topology, "edges")
            and hasattr(self.sdn_props, "network_spectrum_dict")
        )

    def route(self, source: Any, destination: Any, request: Any) -> list[Any] | None:
        """
        Find a route from source to destination considering NLI.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param request: Request object containing traffic demand details.
        :type request: Any
        :return: Path with least NLI, or None if no path found.
        :rtype: list[Any] | None
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
                self._total_nli += float(nli)

            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _update_nli_costs(self) -> None:
        """
        Update NLI costs for all links in the topology.

        Calculates and assigns non-linear impairment scores to all network links
        based on current spectrum utilization and span count. Updates both directions
        of bidirectional links with the same NLI cost.
        """
        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, 'topology', None)
        )

        # Bidirectional links are identical, therefore, we don't have to check each one
        network_spectrum_dict = getattr(self.sdn_props, 'network_spectrum_dict', {})
        for link_tuple in list(network_spectrum_dict.keys())[::2]:
            source_node, destination_node = link_tuple[0], link_tuple[1]
            if topology is not None:
                span_count = (
                    topology[source_node][destination_node]["length"]
                    / self.route_props.span_length
                )
            else:
                span_count = 1.0
            connection_bandwidth = getattr(self.sdn_props, 'bandwidth', None)

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

            if topology is not None and hasattr(topology, "edges"):
                topology[source_node][destination_node]["nli_cost"] = nli_link_cost
                topology[destination_node][source_node]["nli_cost"] = nli_link_cost

    def _find_least_weight(self, weight: str) -> None:
        """
        Find the path with the least weight based on NLI cost.

        Updates the route properties with the path having minimum NLI,
        along with its weight and modulation format.

        :param weight: The edge attribute name to use for weight calculation,
            typically 'nli_cost'.
        :type weight: str
        """
        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, 'topology', None)
        )

        paths_generator = nx.shortest_simple_paths(
            G=topology,
            source=self.sdn_props.source,
            target=self.sdn_props.destination,
            weight=weight,
        )

        for path_list in paths_generator:
            # Calculate path weight as sum across the path
            path_weight = 0.0
            if topology is not None:
                path_weight = sum(
                    topology[path_list[i]][path_list[i + 1]][weight]
                    for i in range(len(path_list) - 1)
                )

            # Calculate actual path length for modulation format selection
            path_length = find_path_length(path_list=path_list, topology=topology)
            # Get modulation format
            modulation_formats = getattr(self.sdn_props, "mod_formats", {})
            modulation_format = get_path_modulation(modulation_formats, path_length)
            if modulation_format and modulation_format is not True:
                modulation_format_list = [str(modulation_format)]
            else:
                modulation_format_list = ["QPSK"]

            self.route_props.weights_list.append(path_weight)
            self.route_props.paths_matrix.append(path_list)
            self.route_props.modulation_formats_matrix.append(modulation_format_list)
            # For NLI-aware, we typically take the first (best) path
            break

    def _calculate_path_nli(self, path: list[Any]) -> float:
        """
        Calculate the NLI metric for a given path.

        :param path: List of node identifiers representing the path.
        :type path: list[Any]
        :return: NLI metric value for the path. Returns 0.0 if
            the path is invalid or has less than 2 nodes.
        :rtype: float
        """
        if not path or len(path) < 2:
            return 0.0

        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, 'topology', None)
        )
        total_nli = 0.0

        for link_index in range(len(path) - 1):
            source_node, destination_node = path[link_index], path[link_index + 1]
            if (topology is not None and
                hasattr(topology, "edges") and
                topology.has_edge(source_node, destination_node)):
                link_nli_cost = topology[source_node][destination_node].get(
                    "nli_cost", 0.0
                )
                total_nli += link_nli_cost

        return total_nli

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """
        Get k paths ordered by NLI level.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param k: Number of paths to return.
        :type k: int
        :return: List of k paths ordered by NLI (least NLI first).
        :rtype: list[list[Any]]
        """
        # Update NLI costs first
        self._update_nli_costs()
        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, 'topology', None)
        )

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
        """
        Update NLI weights based on current network state.

        :param topology: NetworkX graph to update weights for.
        :type topology: Any
        """
        # Recalculate NLI costs for all links
        network_spectrum_dict = getattr(self.sdn_props, 'network_spectrum_dict', {})
        for link_tuple in list(network_spectrum_dict.keys())[::2]:
            source_node, destination_node = link_tuple[0], link_tuple[1]
            span_count = (
                topology[source_node][destination_node]["length"]
                / self.route_props.span_length
            )

            nli_link_cost = self.route_help_obj.get_nli_cost(
                link_tuple=link_tuple, num_span=span_count
            )

            if topology is not None and hasattr(topology, "edges"):
                topology[source_node][destination_node]["nli_cost"] = nli_link_cost
                topology[destination_node][source_node]["nli_cost"] = nli_link_cost

    def get_metrics(self) -> dict[str, Any]:
        """
        Get routing algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics including
            algorithm name, paths computed, average NLI, and total NLI considered.
        :rtype: dict[str, Any]
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
        self._total_nli = 0.0
        self.route_props = RoutingProps()
