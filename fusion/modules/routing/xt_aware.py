"""
Cross-talk (XT) aware routing algorithm implementation.
"""

from typing import Any

import networkx as nx

from fusion.core.properties import RoutingProps, SDNProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers
from fusion.utils.data import sort_nested_dict_values
from fusion.utils.network import find_path_length, get_path_modulation
from fusion.utils.spectrum import find_free_slots


class XTAwareRouting(AbstractRoutingAlgorithm):
    """
    Cross-talk aware routing algorithm.

    This algorithm finds paths by considering intra-core crosstalk interference,
    selecting the path with the least amount of cross-talk.
    """

    def __init__(self, engine_props: dict[str, Any], sdn_props: SDNProps) -> None:
        """
        Initialize XT-aware routing algorithm.

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
        self._total_xt = 0.0

    @property
    def algorithm_name(self) -> str:
        """
        Get the name of the routing algorithm.

        :return: The algorithm name 'xt_aware'.
        :rtype: str
        """
        return "xt_aware"

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
        Find a route from source to destination considering cross-talk.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param request: Request object containing traffic demand details.
        :type request: Any
        :return: Path with least cross-talk, or None if no path found.
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
            # Update XT costs for all links
            self._update_xt_costs()

            # Find least XT path
            self._find_least_weight("xt_cost")

            path = None
            if self.route_props.paths_matrix:
                path = self.route_props.paths_matrix[0]
                self._path_count += 1

                # Calculate XT metric for this path
                xt = self._calculate_path_xt(path)
                self._total_xt += float(xt)

            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _update_xt_costs(self) -> None:
        """
        Update cross-talk costs for all links in the topology.

        Calculates and assigns cross-talk scores to all network links
        based on current spectrum utilization and XT calculation type.
        Updates both directions of bidirectional links with the same XT cost.
        """
        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, 'topology', None)
        )

        # At the moment, we have identical bidirectional links
        # (no need to loop over all links)
        network_spectrum_dict = getattr(self.sdn_props, 'network_spectrum_dict', {})
        for link_tuple in list(network_spectrum_dict.keys())[::2]:
            source_node, destination_node = link_tuple[0], link_tuple[1]
            span_count = 1.0
            if topology is not None:
                span_count = (
                    topology[source_node][destination_node]["length"]
                    / self.route_props.span_length
                )

            available_slots_dict = find_free_slots(
                network_spectrum_dict=network_spectrum_dict,
                link_tuple=link_tuple,
            )
            crosstalk_cost = self.route_help_obj.find_xt_link_cost(
                free_slots_dict=available_slots_dict, link_list=link_tuple
            )

            # Consider XT type configuration
            xt_calculation_type = self.engine_props.get("xt_type")
            beta_coefficient = self.engine_props.get("beta", 0.5)

            if xt_calculation_type == "with_length":
                if self.route_props.max_link_length is None:
                    self.route_help_obj.get_max_link_length()

                normalized_length = 1.0
                if topology is not None:
                    normalized_length = (
                        topology[source_node][destination_node]["length"]
                        / self.route_props.max_link_length
                    )
                length_weighted_cost = normalized_length * beta_coefficient
                crosstalk_weighted_cost = (1 - beta_coefficient) * crosstalk_cost
                final_link_cost = length_weighted_cost + crosstalk_weighted_cost
            elif xt_calculation_type == "without_length":
                final_link_cost = span_count * crosstalk_cost
            else:
                # Default behavior
                final_link_cost = crosstalk_cost

            if topology is not None and hasattr(topology, "edges"):
                topology[source_node][destination_node]["xt_cost"] = final_link_cost
                topology[destination_node][source_node]["xt_cost"] = final_link_cost

    def _find_least_weight(self, weight: str) -> None:
        """
        Find the path with the least weight based on cross-talk cost.

        Updates the route properties with the path having minimum cross-talk,
        along with its weight and modulation format.

        :param weight: The edge attribute name to use for weight calculation,
            typically 'xt_cost'.
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

            # Get modulation formats based on path length
            has_mod_dict = hasattr(self.sdn_props, "modulation_formats_dict")
            if (
                has_mod_dict
                and self.sdn_props.modulation_formats_dict is not None
            ):
                # Use sorted modulation formats
                mod_formats_dict = sort_nested_dict_values(
                    original_dict=self.sdn_props.modulation_formats_dict,
                    nested_key="max_length",
                )
                mod_format_list = []
                for mod_format in mod_formats_dict:
                    if (
                        self.sdn_props.modulation_formats_dict[mod_format]["max_length"]
                        >= path_length
                    ):
                        mod_format_list.append(mod_format)
                    else:
                        mod_format_list.append(False)  # type: ignore[arg-type]
            else:
                # Fallback to simple modulation selection
                mod_formats = getattr(self.sdn_props, "mod_formats", {})
                modulation_format = get_path_modulation(mod_formats, path_length)
                if modulation_format and modulation_format is not True:
                    mod_format_list = [str(modulation_format)]
                else:
                    mod_format_list = ["QPSK"]

            self.route_props.weights_list.append(path_weight)
            self.route_props.paths_matrix.append(path_list)
            self.route_props.modulation_formats_matrix.append(mod_format_list)
            # For XT-aware, we typically take the first (best) path
            break

    def _calculate_path_xt(self, path: list[Any]) -> float:
        """
        Calculate the cross-talk metric for a given path.

        :param path: List of node identifiers representing the path.
        :type path: list[Any]
        :return: Cross-talk metric value for the path. Returns 0.0 if
            the path is invalid or has less than 2 nodes.
        :rtype: float
        """
        if not path or len(path) < 2:
            return 0.0

        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, 'topology', None)
        )
        total_xt = 0.0

        for i in range(len(path) - 1):
            source, destination = path[i], path[i + 1]
            if (topology is not None and
                hasattr(topology, "edges") and
                topology.has_edge(source, destination)):
                xt_cost = topology[source][destination].get("xt_cost", 0.0)
                total_xt += xt_cost

        return total_xt

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """
        Get k paths ordered by cross-talk level.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param k: Number of paths to return.
        :type k: int
        :return: List of k paths ordered by cross-talk (least XT first).
        :rtype: list[list[Any]]
        """
        # Update XT costs first
        self._update_xt_costs()
        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, 'topology', None)
        )

        try:
            paths_generator = nx.shortest_simple_paths(
                topology, source, destination, weight="xt_cost"
            )

            paths = []
            for i, path in enumerate(paths_generator):
                if i >= k:
                    break
                paths.append(list(path))

            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def update_weights(self, topology: Any) -> None:
        """
        Update cross-talk weights based on current spectrum state.

        :param topology: NetworkX graph to update weights for.
        :type topology: Any
        """
        # Recalculate XT costs for all links
        network_spectrum_dict = getattr(self.sdn_props, 'network_spectrum_dict', {})
        for link_list in list(network_spectrum_dict.keys())[::2]:
            source, destination = link_list[0], link_list[1]
            num_spans = (
                topology[source][destination]["length"] / self.route_props.span_length
            )

            free_slots_dict = find_free_slots(
                network_spectrum_dict=network_spectrum_dict,
                link_tuple=link_list,
            )
            xt_cost = self.route_help_obj.find_xt_link_cost(
                free_slots_dict=free_slots_dict, link_list=link_list
            )

            # Apply XT type configuration
            if self.engine_props.get("xt_type") == "with_length":
                if self.route_props.max_link_length is None:
                    self.route_help_obj.get_max_link_length()

                link_cost = (
                    topology[source][destination]["length"]
                    / self.route_props.max_link_length
                )
                link_cost *= self.engine_props.get("beta", 0.5)
                link_cost += (1 - self.engine_props.get("beta", 0.5)) * xt_cost
            elif self.engine_props.get("xt_type") == "without_length":
                link_cost = num_spans * xt_cost
            else:
                link_cost = xt_cost

            if hasattr(topology, "edges"):
                topology[source][destination]["xt_cost"] = link_cost
                topology[destination][source]["xt_cost"] = link_cost

    def get_metrics(self) -> dict[str, Any]:
        """
        Get routing algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics including
            algorithm name, paths computed, average XT, total XT considered,
            and XT calculation type.
        :rtype: dict[str, Any]
        """
        avg_xt = self._total_xt / self._path_count if self._path_count > 0 else 0

        return {
            "algorithm": self.algorithm_name,
            "paths_computed": self._path_count,
            "average_xt": avg_xt,
            "total_xt_considered": self._total_xt,
            "xt_type": self.engine_props.get("xt_type", "default"),
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_xt = 0.0
        self.route_props = RoutingProps()
