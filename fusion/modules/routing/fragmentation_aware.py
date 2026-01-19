"""
Fragmentation-aware routing algorithm implementation.
"""

from typing import Any

import networkx as nx

from fusion.core.properties import RoutingProps, SDNProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers
from fusion.utils.network import (
    find_path_fragmentation,
    find_path_length,
    get_path_modulation,
)


class FragmentationAwareRouting(AbstractRoutingAlgorithm):
    """
    Fragmentation-aware routing algorithm.

    This algorithm finds paths by considering spectrum fragmentation levels,
    selecting the path with the least fragmentation.
    """

    def __init__(self, engine_props: dict[str, Any], sdn_props: SDNProps) -> None:
        """
        Initialize fragmentation-aware routing algorithm.

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
        self._total_fragmentation = 0.0

    @property
    def algorithm_name(self) -> str:
        """
        Get the name of the routing algorithm.

        :return: The algorithm name 'fragmentation_aware'.
        :rtype: str
        """
        return "fragmentation_aware"

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
        return hasattr(topology, "nodes") and hasattr(topology, "edges") and hasattr(self.sdn_props, "network_spectrum_dict")

    def route(self, source: Any, destination: Any, request: Any) -> None:
        """
        Find a route from source to destination considering fragmentation.

        Results are stored in route_props (paths_matrix, modulation_formats_matrix,
        weights_list). Consumers should access route_props.paths_matrix for paths.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param request: Request object containing traffic demand details.
        :type request: Any
        """
        # Store source/destination in sdn_props for compatibility
        self.sdn_props.source = source
        self.sdn_props.destination = destination

        # Reset paths matrix for new calculation
        self.route_props.paths_matrix = []
        self.route_props.weights_list = []
        self.route_props.modulation_formats_matrix = []

        try:
            # Update fragmentation costs for all links
            self._update_fragmentation_costs()

            # Find least fragmented path
            self._find_least_weight("frag_cost")

            if self.route_props.paths_matrix:
                path = self.route_props.paths_matrix[0]
                self._path_count += 1

                # Calculate fragmentation metric for this path
                fragmentation = self._calculate_path_fragmentation(path)
                self._total_fragmentation += float(fragmentation)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    def _update_fragmentation_costs(self) -> None:
        """
        Update fragmentation costs for all links in the topology.

        Calculates and assigns fragmentation scores to all network links
        based on current spectrum utilization. Updates both directions
        of bidirectional links with the same fragmentation cost.
        """
        topology = self.engine_props.get("topology", getattr(self.sdn_props, "topology", None))

        # Calculate fragmentation for each direct link
        network_spectrum_dict = getattr(self.sdn_props, "network_spectrum_dict", {})
        for link_tuple in list(network_spectrum_dict.keys())[::2]:
            source_node, destination_node = link_tuple
            direct_path = [source_node, destination_node]

            # Compute average fragmentation on this direct link
            fragmentation_score = find_path_fragmentation(
                path_list=direct_path,
                network_spectrum=network_spectrum_dict,
            )

            # Store fragmentation score for both directions if bidirectional
            if topology is not None and hasattr(topology, "edges"):
                topology[source_node][destination_node]["frag_cost"] = fragmentation_score
                topology[destination_node][source_node]["frag_cost"] = fragmentation_score

    def _find_least_weight(self, weight: str) -> None:
        """
        Find the path with the least weight based on fragmentation cost.

        Updates the route properties with the path having minimum fragmentation,
        along with its weight and modulation format.

        :param weight: The edge attribute name to use for weight calculation,
            typically 'frag_cost'.
        :type weight: str
        """
        topology = self.engine_props.get("topology", getattr(self.sdn_props, "topology", None))

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
                path_weight = sum(topology[path_list[i]][path_list[i + 1]][weight] for i in range(len(path_list) - 1))

            # Calculate actual path length for modulation format selection
            path_length = find_path_length(path_list=path_list, topology=topology)
            # Get modulation format
            modulation_formats = getattr(self.sdn_props, "mod_formats", {})
            modulation_format = get_path_modulation(modulation_formats, path_length)
            modulation_format_list: list[str | bool]
            if modulation_format and modulation_format is not True:
                modulation_format_list = [str(modulation_format)]
            else:
                modulation_format_list = ["QPSK"]

            self.route_props.weights_list.append(path_weight)
            self.route_props.paths_matrix.append(path_list)
            self.route_props.modulation_formats_matrix.append(modulation_format_list)
            # For fragmentation-aware, we typically take the first (best) path
            break

    def _calculate_path_fragmentation(self, path: list[Any]) -> float:
        """
        Calculate the fragmentation metric for a given path.

        :param path: List of node identifiers representing the path.
        :type path: list[Any]
        :return: Fragmentation metric value for the path. Returns 0.0 if
            the path is invalid or has less than 2 nodes.
        :rtype: float
        """
        if not path or len(path) < 2:
            return 0.0

        try:
            network_spectrum_dict = getattr(self.sdn_props, "network_spectrum_dict", {})
            return find_path_fragmentation(
                path_list=path,
                network_spectrum=network_spectrum_dict,
            )
        except (KeyError, AttributeError, TypeError):
            return 0.0

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """
        Get k paths ordered by fragmentation level.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param k: Number of paths to return.
        :type k: int
        :return: List of k paths ordered by fragmentation (least fragmented first).
        :rtype: list[list[Any]]
        """
        # Update fragmentation costs first
        topology = self.engine_props.get("topology", getattr(self.sdn_props, "topology", None))
        self._update_fragmentation_costs()

        try:
            paths_generator = nx.shortest_simple_paths(topology, source, destination, weight="frag_cost")

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
        Update fragmentation weights based on current spectrum state.

        :param topology: NetworkX graph to update weights for.
        :type topology: Any
        """
        # Recalculate fragmentation costs for all links
        network_spectrum_dict = getattr(self.sdn_props, "network_spectrum_dict", {})
        for link_tuple in list(network_spectrum_dict.keys())[::2]:
            source_node, destination_node = link_tuple
            direct_path = [source_node, destination_node]

            fragmentation_score = find_path_fragmentation(
                path_list=direct_path,
                network_spectrum=network_spectrum_dict,
            )

            if hasattr(topology, "edges"):
                topology[source_node][destination_node]["frag_cost"] = fragmentation_score
                topology[destination_node][source_node]["frag_cost"] = fragmentation_score

    def get_metrics(self) -> dict[str, Any]:
        """
        Get routing algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics including
            algorithm name, paths computed, average fragmentation, and total
            fragmentation considered.
        :rtype: dict[str, Any]
        """
        avg_fragmentation = self._total_fragmentation / self._path_count if self._path_count > 0 else 0

        return {
            "algorithm": self.algorithm_name,
            "paths_computed": self._path_count,
            "average_fragmentation": avg_fragmentation,
            "total_fragmentation_considered": self._total_fragmentation,
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_fragmentation = 0.0
        self.route_props = RoutingProps()
