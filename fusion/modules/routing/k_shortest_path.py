"""
K-Shortest Path routing algorithm implementation.
"""

from typing import Any

import networkx as nx

from fusion.core.properties import RoutingProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.utils.data import sort_nested_dict_values
from fusion.utils.network import find_path_length, get_path_modulation


class KShortestPath(AbstractRoutingAlgorithm):
    """
    K-Shortest Path routing algorithm.

    This algorithm finds the k shortest paths between source and destination
    based on hop count or other specified weights.
    """

    def __init__(self, engine_props: dict[str, Any], sdn_props: Any) -> None:
        """
        Initialize K-Shortest Path routing algorithm.

        :param engine_props: Dictionary containing engine configuration.
        :type engine_props: dict[str, Any]
        :param sdn_props: Object containing SDN controller properties.
        :type sdn_props: Any
        """
        super().__init__(engine_props, sdn_props)
        self.k_paths_count = engine_props.get("k_paths", 3)
        self.routing_weight = engine_props.get("routing_weight", "length")
        self._path_count = 0
        self._total_hops = 0

        # Initialize route properties for legacy compatibility
        self.route_props = RoutingProps()

    @property
    def algorithm_name(self) -> str:
        """
        Get the name of the routing algorithm.

        :return: The algorithm name 'k_shortest_path'.
        :rtype: str
        """
        return "k_shortest_path"

    @property
    def supported_topologies(self) -> list[str]:
        """
        Get the list of supported topology types.

        :return: List of supported topology names including NSFNet,
            USBackbone60, Pan-European, and Generic.
        :rtype: list[str]
        """
        return ["NSFNet", "USBackbone60", "Pan-European", "Generic"]

    def validate_environment(self, topology: nx.Graph) -> bool:
        """
        Validate that the routing algorithm can work with the given topology.

        :param topology: NetworkX graph representing the network topology.
        :type topology: nx.Graph
        :return: True if the algorithm can route in this environment.
        :rtype: bool
        """
        # K-shortest path works with any connected graph
        try:
            return bool(nx.is_connected(topology))
        except Exception:
            return False

    def route(self, source: Any, destination: Any, request: Any) -> None:
        """
        Find a route from source to destination for the given request.

        Results are stored in route_props (paths_matrix, modulation_formats_matrix,
        weights_list). Consumers should access route_props.paths_matrix for paths.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param request: Request object containing traffic demand details.
        :type request: Any
        """
        # Clear previous route properties
        self.route_props.paths_matrix = []
        self.route_props.modulation_formats_matrix = []
        self.route_props.weights_list = []
        self.route_props.path_index_list = []
        self.route_props.connection_index = None

        paths = self.get_paths(source, destination, k=self.k_paths_count)

        if not paths:
            return

        # Populate route_props
        topology = self.engine_props.get("topology", getattr(self.sdn_props, "topology", None))

        for path in paths:
            # Calculate path length
            path_length = find_path_length(path_list=path, topology=topology)

            # Get modulation formats
            # Note: False is used as sentinel for infeasible modulation formats
            modulation_formats_list: list[str | bool]
            chosen_bandwidth = getattr(self.sdn_props, "bandwidth", None)
            pre_calc = self.engine_props.get("pre_calc_mod_selection", False)
            if chosen_bandwidth and pre_calc:
                # Use mod_per_bw if available
                if "mod_per_bw" in self.engine_props and chosen_bandwidth in self.engine_props["mod_per_bw"]:
                    modulation_format = get_path_modulation(
                        mods_dict=self.engine_props["mod_per_bw"][chosen_bandwidth],
                        path_len=path_length,
                    )
                    modulation_formats_list = [str(modulation_format)]
                else:
                    # Fallback to mod_formats
                    modulation_formats = getattr(self.sdn_props, "mod_formats", {})
                    modulation_format = get_path_modulation(modulation_formats, path_length)
                    modulation_formats_list = [str(modulation_format)]
            else:
                # Use all modulation formats sorted by max_length
                has_mod_dict = hasattr(self.sdn_props, "modulation_formats_dict")
                if has_mod_dict and self.sdn_props.modulation_formats_dict is not None:
                    modulation_formats_dict = sort_nested_dict_values(
                        original_dict=self.sdn_props.modulation_formats_dict,
                        nested_key="max_length",
                    )
                    # Filter modulations by path length feasibility
                    # Only include modulations that can reach the path distance
                    modulation_formats_list = []
                    for mod_format in modulation_formats_dict.keys():
                        mod_info = self.sdn_props.modulation_formats_dict[mod_format]
                        max_len = mod_info.get("max_length", 0)
                        if max_len >= path_length:
                            modulation_formats_list.append(str(mod_format))
                        else:
                            modulation_formats_list.append(False)
                else:
                    # Fallback to simple list
                    modulation_formats_list = ["QPSK"]

            self.route_props.paths_matrix.append(path)
            self.route_props.modulation_formats_matrix.append(modulation_formats_list)
            self.route_props.weights_list.append(path_length)

        # Update metrics based on shortest path
        if paths:
            self._path_count += 1
            self._total_hops += len(paths[0]) - 1

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """
        Get k shortest paths between source and destination.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param k: Number of paths to return.
        :type k: int
        :return: List of k paths, where each path is a list of nodes.
        :rtype: list[list[Any]]
        """
        topology = self.engine_props.get("topology", getattr(self.sdn_props, "topology", None))

        try:
            if self.routing_weight:
                # Use specified weight for path calculation
                paths_generator = nx.shortest_simple_paths(topology, source, destination, weight=self.routing_weight)
            else:
                # Use hop count (unweighted)
                paths_generator = nx.shortest_simple_paths(topology, source, destination)

            paths_list = []
            for path_index, path in enumerate(paths_generator):
                if path_index >= k:
                    break
                paths_list.append(list(path))

            return paths_list

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def update_weights(self, topology: nx.Graph) -> None:
        """
        Update edge weights based on current network state.

        This implementation doesn't dynamically update weights.
        Subclasses can override this method for adaptive routing.

        :param topology: NetworkX graph to update weights for.
        :type topology: nx.Graph
        """

    def get_metrics(self) -> dict[str, Any]:
        """
        Get routing algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics including
            algorithm name, paths computed, average hop count, k value, and
            weight metric.
        :rtype: dict[str, Any]
        """
        avg_hops = self._total_hops / self._path_count if self._path_count > 0 else 0

        return {
            "algorithm": self.algorithm_name,
            "paths_computed": self._path_count,
            "average_hop_count": avg_hops,
            "k_value": self.k_paths_count,
            "weight_metric": self.routing_weight or "hop_count",
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_hops = 0
