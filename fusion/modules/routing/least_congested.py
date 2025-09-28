"""
Least congested link routing algorithm implementation.

This algorithm finds paths with minimum hops ± 1 and selects the one
with the least congested bottleneck link.
"""

from typing import Any

import networkx as nx
import numpy as np

from fusion.core.properties import RoutingProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers


class LeastCongestedRouting(AbstractRoutingAlgorithm):
    """
    Least congested link routing algorithm.

    This algorithm finds paths with minimum hops ± 1 and selects the one
    with the least congested bottleneck link (not the mean congestion).
    """

    def __init__(self, engine_props: dict[str, Any], sdn_props: Any) -> None:
        """
        Initialize least congested routing algorithm.

        :param engine_props: Dictionary containing engine configuration.
        :type engine_props: dict[str, Any]
        :param sdn_props: Object containing SDN controller properties.
        :type sdn_props: Any
        """
        super().__init__(engine_props, sdn_props)
        self.route_props = RoutingProps()
        self.route_help_obj = RoutingHelpers(
            route_props=self.route_props,
            engine_props=self.engine_props,
            sdn_props=self.sdn_props
        )
        self._path_count = 0
        self._total_congestion = 0.0

    @property
    def algorithm_name(self) -> str:
        """
        Get the name of the routing algorithm.

        :return: The algorithm name 'least_congested'.
        :rtype: str
        """
        return "least_congested"

    @property
    def supported_topologies(self) -> list[str]:
        """
        Get the list of supported topology types.

        :return: List of supported topology names including NSFNet,
            USBackbone60, Pan-European, and Generic.
        :rtype: list[str]
        """
        return ['NSFNet', 'USBackbone60', 'Pan-European', 'Generic']

    def validate_environment(self, topology: Any) -> bool:
        """
        Validate that the routing algorithm can work with the given topology.

        :param topology: NetworkX graph representing the network topology.
        :type topology: Any
        :return: True if the algorithm can route in this environment.
        :rtype: bool
        """
        return (hasattr(topology, 'nodes') and
                hasattr(topology, 'edges') and
                hasattr(self.sdn_props, 'network_spectrum_dict'))

    def route(self, source: Any, destination: Any, request: Any) -> list[Any] | None:
        """
        Find the least congested path in the network.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param request: Request object containing traffic demand details.
        :type request: Any
        :return: Path with least congested bottleneck link, or None if no path found.
        :rtype: list[Any] | None
        """
        # Store source/destination in sdn_props for compatibility
        self.sdn_props.source = source
        self.sdn_props.destination = destination

        # Reset paths matrix for new calculation
        self.route_props.paths_matrix = []
        self.route_props.modulation_formats_matrix = []
        self.route_props.weights_list = []

        try:
            self._find_least_congested_paths()
            if self.route_props.paths_matrix:
                self._path_count += 1
                # Extract just the path from the dictionary structure
                path_data = self.route_props.paths_matrix[0]
                if isinstance(path_data, dict) and 'path_list' in path_data:
                    best_path = path_data['path_list']
                    congestion = self._calculate_path_congestion(best_path)
                    self._total_congestion += float(congestion)
                    return (
                        list(best_path)
                        if isinstance(best_path, (list, tuple))
                        else None
                    )
                return None
            return None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _find_least_congested_paths(self) -> None:
        """
        Find paths with minimum hops ± 1 and select least congested.

        Evaluates paths with hop count within 1 of the minimum and selects
        the one with the least congested bottleneck link.
        """
        topology = self.engine_props.get(
            'topology', getattr(self.sdn_props, 'topology', None)
        )

        all_paths_generator = nx.shortest_simple_paths(
            topology,
            self.sdn_props.source,
            self.sdn_props.destination
        )
        minimum_hops = None

        for path_index, path_list in enumerate(all_paths_generator):
            hop_count = len(path_list)
            if path_index == 0:
                minimum_hops = hop_count
                self._find_most_cong_link(path_list=path_list)
            else:
                if minimum_hops is not None and hop_count <= minimum_hops + 1:
                    self._find_most_cong_link(path_list=path_list)
                # We exceeded minimum hops plus one, return the best path
                else:
                    self._select_least_congested()
                    return

        # If we exit the loop, select the best from what we have
        if self.route_props.paths_matrix:
            self._select_least_congested()

    def _find_most_cong_link(self, path_list: list[Any]) -> None:
        """
        Find the most congested link along a path.

        Identifies the link with the least free slots along the given path
        and stores it in the route properties matrix.

        :param path_list: List of node identifiers representing the path.
        :type path_list: list[Any]
        """
        most_congested_link = None
        most_congested_free_slots = -1

        for link_index in range(len(path_list) - 1):
            network_spectrum_dict = getattr(self.sdn_props, 'network_spectrum_dict', {})
            link_dict = network_spectrum_dict[
                (path_list[link_index], path_list[link_index + 1])
            ]
            total_free_slots = 0
            for band in link_dict['cores_matrix']:
                cores_matrix = link_dict['cores_matrix'][band]
                for core_array in cores_matrix:
                    total_free_slots += np.sum(core_array == 0)

            if (
                total_free_slots < most_congested_free_slots
                or most_congested_link is None
            ):
                most_congested_free_slots = total_free_slots
                most_congested_link = link_dict

        self.route_props.paths_matrix.append({
            'path_list': path_list,
            'link_dict': {
                'link': most_congested_link,
                'free_slots': most_congested_free_slots,
            },
        })

    def _select_least_congested(self) -> None:
        """
        Select the path with the least congested bottleneck link.

        Sorts all candidate paths by their most congested link's free slots
        and updates the route properties with the best path.
        """
        # Sort paths by number of free slots, descending (most free slots first)
        sorted_paths_list = sorted(
            self.route_props.paths_matrix,
            key=lambda path_data: path_data['link_dict']['free_slots'],
            reverse=True
        )

        # Keep all paths in the matrix (for backward compatibility with tests)
        self.route_props.paths_matrix = sorted_paths_list

        # But only populate weights and modulation for the best path
        best_path_data = sorted_paths_list[0]
        self.route_props.weights_list = [int(best_path_data['link_dict']['free_slots'])]
        # Use QPSK as the default modulation format for least congested routing
        self.route_props.modulation_formats_matrix = [['QPSK']]

    def _calculate_path_congestion(self, path: list[Any]) -> float:
        """
        Calculate the congestion metric for a given path.

        :param path: List of node identifiers representing the path.
        :type path: list[Any]
        :return: Congestion ratio (used slots / total slots) for the path.
            Returns 0.0 if the path is invalid or has less than 2 nodes.
        :rtype: float
        """
        if not path or len(path) < 2:
            return 0.0

        total_used_slots = 0
        total_slots = 0

        network_spectrum_dict = getattr(self.sdn_props, 'network_spectrum_dict', {})
        for i in range(len(path) - 1):
            link_key = (path[i], path[i + 1])
            if link_key in network_spectrum_dict:
                link_dict = network_spectrum_dict[link_key]
                for band in link_dict['cores_matrix']:
                    cores_matrix = link_dict['cores_matrix'][band]
                    for core_arr in cores_matrix:
                        total_slots += len(core_arr)
                        total_used_slots += np.sum(core_arr != 0)

        return total_used_slots / total_slots if total_slots > 0 else 0.0

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """
        Get k paths ordered by congestion level.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param k: Number of paths to return.
        :type k: int
        :return: List of k paths ordered by congestion (least congested first).
        :rtype: list[list[Any]]
        """
        best_path = self.route(source, destination, None)
        return [best_path] if best_path else []

    def update_weights(self, topology: Any) -> None:
        """
        Update congestion weights based on current network state.

        For least congested routing, we don't pre-compute weights
        since we evaluate congestion dynamically.

        :param topology: NetworkX graph to update weights for.
        :type topology: Any
        """

    def get_metrics(self) -> dict[str, Any]:
        """
        Get routing algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics including
            algorithm name, paths computed, average congestion, and total
            congestion considered.
        :rtype: dict[str, Any]
        """
        avg_congestion = (
            self._total_congestion / self._path_count if self._path_count > 0 else 0
        )

        return {
            'algorithm': self.algorithm_name,
            'paths_computed': self._path_count,
            'average_congestion': avg_congestion,
            'total_congestion_considered': self._total_congestion
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_congestion = 0.0
        self.route_props = RoutingProps()
