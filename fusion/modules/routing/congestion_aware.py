"""
Congestion-aware routing algorithm implementation.
"""

from typing import Any

import networkx as nx
import numpy as np

from fusion.core.properties import RoutingProps, SDNProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers
from fusion.utils.data import sort_nested_dict_values
from fusion.utils.network import (
    find_path_congestion,
    find_path_length,
    get_path_modulation,
)


class CongestionAwareRouting(AbstractRoutingAlgorithm):
    """
    Congestion-aware routing algorithm.

    This algorithm finds paths by considering network congestion levels,
    selecting the path with the least congested link.
    """

    def __init__(self, engine_props: dict[str, Any], sdn_props: SDNProps) -> None:
        """
        Initialize congestion-aware routing algorithm.

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
            sdn_props=self.sdn_props,
        )
        self._path_count = 0
        self._total_congestion = 0.0

    @property
    def algorithm_name(self) -> str:
        """
        Get the name of the routing algorithm.

        :return: The algorithm name 'congestion_aware'.
        :rtype: str
        """
        return "congestion_aware"

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
        # Check if topology has the required attributes for congestion calculation
        return (
            hasattr(topology, "nodes")
            and hasattr(topology, "edges")
            and hasattr(self.sdn_props, "network_spectrum_dict")
        )

    def route(self, source: Any, destination: Any, request: Any) -> list[Any] | None:
        """
        Find a route from source to destination using congestion-aware k-shortest.

        For the first k shortest-length candidate paths we compute:
            score = alpha * mean_path_congestion + (1 - alpha) * (path_len / max_len)

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :param request: Request object containing traffic demand details.
        :type request: Any
        :return: Best path based on congestion score, or None if no path found.
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
            self._find_congestion_aware_paths()
            if self.route_props.paths_matrix:
                self._path_count += 1
                # Calculate congestion metric for the best path
                best_path = self.route_props.paths_matrix[0]
                congestion = self._calculate_path_congestion(best_path)
                self._total_congestion += float(congestion)
                return list(best_path) if isinstance(best_path, (list, tuple)) else None
            return None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _find_congestion_aware_paths(self) -> None:
        """
        Implement the sophisticated congestion-aware k-shortest routing.

        For the first k shortest-length candidate paths computes:
            score = alpha * mean_path_congestion + (1 - alpha) * (path_len / max_len)

        All k paths are stored in route_props.*, sorted by score so the
        downstream allocator will try the most promising path first.
        """
        candidate_paths_data = self._gather_candidate_paths()
        if not candidate_paths_data:
            # Set blocked state - using block_reason on sdn_props
            self.sdn_props.block_reason = "congestion"
            return

        scored_paths = self._calculate_path_scores(candidate_paths_data)
        self._populate_route_properties(scored_paths)

    def _gather_candidate_paths(self) -> dict[str, list[Any]]:
        """
        Gather k shortest-length candidate paths with their metrics.

        :return: Dictionary containing 'paths', 'lengths', and 'congestions' lists.
        :rtype: dict[str, list[Any]]
        """
        k_paths = int(self.engine_props.get("k_paths", 1))

        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, "topology", None)
        )
        paths_iterator = nx.shortest_simple_paths(
            G=topology,
            source=self.sdn_props.source,
            target=self.sdn_props.destination,
            weight="length",
        )

        candidate_paths, path_lengths, path_congestions = [], [], []
        for path_index, path in enumerate(paths_iterator):
            if path_index >= k_paths:
                break
            candidate_paths.append(path)

            path_length = find_path_length(path_list=path, topology=topology)
            path_lengths.append(path_length)

            mean_congestion, _ = find_path_congestion(
                path_list=path,
                network_spectrum=getattr(self.sdn_props, "network_spectrum_dict", {}),
            )
            path_congestions.append(mean_congestion)

        return {
            "paths": candidate_paths,
            "lengths": path_lengths,
            "congestions": path_congestions,
        }

    def _calculate_path_scores(
        self, candidate_paths_data: dict[str, list[Any]]
    ) -> list[tuple]:
        """
        Calculate congestion-aware scores for candidate paths.

        :param candidate_paths_data: Dictionary with paths, lengths, and congestions.
        :type candidate_paths_data: dict[str, list[Any]]
        :return: List of tuples (path, length, score) sorted by score.
        :rtype: list[tuple]
        """
        alpha = float(self.engine_props.get("ca_alpha", 0.3))

        path_lengths_array = np.asarray(candidate_paths_data["lengths"], dtype=float)
        path_congestions_array = np.asarray(
            candidate_paths_data["congestions"], dtype=float
        )

        max_length = path_lengths_array.max() if path_lengths_array.max() > 0 else 1.0
        normalized_hop_counts = path_lengths_array / max_length
        scores = alpha * path_congestions_array + (1.0 - alpha) * normalized_hop_counts

        # Create list of (path, length, score) tuples sorted by score
        scored_paths = []
        for idx in scores.argsort():
            scored_paths.append(
                (
                    candidate_paths_data["paths"][idx],
                    candidate_paths_data["lengths"][idx],
                    float(scores[idx]),
                )
            )

        return scored_paths

    def _populate_route_properties(self, scored_paths: list[tuple]) -> None:
        """
        Populate route properties with scored paths in order.

        :param scored_paths: List of (path, length, score) tuples sorted by score.
        :type scored_paths: list[tuple]
        """
        chosen_bandwidth = getattr(self.sdn_props, "bandwidth", None)

        for path, path_length, score in scored_paths:
            modulation_list = self._get_modulation_formats(
                path_length, chosen_bandwidth
            )

            self.route_props.paths_matrix.append(path)
            self.route_props.modulation_formats_matrix.append(modulation_list)
            self.route_props.weights_list.append(score)

    def _get_modulation_formats(
        self, path_length: float, chosen_bandwidth: Any
    ) -> list[str]:
        """
        Get appropriate modulation formats for the given path length.

        :param path_length: Length of the path.
        :type path_length: float
        :param chosen_bandwidth: Selected bandwidth for the connection.
        :type chosen_bandwidth: Any
        :return: List of modulation format strings.
        :rtype: list[str]
        """
        if not self.engine_props.get("pre_calc_mod_selection", False):
            modulation_format = get_path_modulation(
                mods_dict=self.engine_props["mod_per_bw"][chosen_bandwidth],
                path_len=path_length,
            )
            if modulation_format and modulation_format is not True:
                return [str(modulation_format)]
            return ["QPSK"]

        modulation_formats_sorted = sort_nested_dict_values(
            original_dict=getattr(self.sdn_props, "mod_formats_dict", {}),
            nested_key="max_length",
        )
        return list(modulation_formats_sorted.keys())

    def _find_least_congested_path(self) -> list[Any] | None:
        """
        Find the least congested path using the original algorithm logic.

        :return: Path with the least congested bottleneck link, or None if no path
            found.
        :rtype: list[Any] | None
        """
        topology = self.engine_props.get(
            "topology", getattr(self.sdn_props, "topology", None)
        )

        all_paths_obj = nx.shortest_simple_paths(
            topology, self.sdn_props.source, self.sdn_props.destination
        )
        min_hops = None

        for i, path_list in enumerate(all_paths_obj):
            num_hops = len(path_list)
            if i == 0:
                min_hops = num_hops
                self._find_most_cong_link(path_list=path_list)
            else:
                if min_hops is not None and num_hops <= min_hops + 1:
                    self._find_most_cong_link(path_list=path_list)
                else:
                    # We exceeded minimum hops plus one, return the best path
                    self._find_least_cong()
                    if self.route_props.paths_matrix:
                        first_path = self.route_props.paths_matrix[0]
                        if isinstance(first_path, dict) and "path_list" in first_path:
                            path_list = first_path["path_list"]
                            return (
                                list(path_list)
                                if isinstance(path_list, (list, tuple))
                                else None
                            )
                        return (
                            list(first_path)
                            if isinstance(first_path, (list, tuple))
                            else None
                        )
                    break

        return None

    def _find_most_cong_link(self, path_list: list[Any]) -> None:
        """
        Find the most congested link along a path.

        Identifies the link with the least free slots along the given path
        and stores it in the route properties matrix.

        :param path_list: List of node identifiers representing the path.
        :type path_list: list[Any]
        """
        most_cong_link = None
        most_cong_slots = -1

        for i in range(len(path_list) - 1):
            network_spectrum_dict = getattr(self.sdn_props, "network_spectrum_dict", {})
            link_dict = network_spectrum_dict[(path_list[i], path_list[i + 1])]
            free_slots = 0
            for band in link_dict["cores_matrix"]:
                cores_matrix = link_dict["cores_matrix"][band]
                for core_arr in cores_matrix:
                    free_slots += np.sum(core_arr == 0)

            if free_slots < most_cong_slots or most_cong_link is None:
                most_cong_slots = free_slots
                most_cong_link = link_dict

        self.route_props.paths_matrix.append(
            {
                "path_list": path_list,
                "link_dict": {"link": most_cong_link, "free_slots": most_cong_slots},
            }
        )

    def _find_least_cong(self) -> None:
        """
        Select the path with the least congested bottleneck link.

        Sorts all candidate paths by their most congested link's free slots
        and updates the route properties with the best path.
        """
        # Sort dictionary by number of free slots, descending
        sorted_paths_list = sorted(
            self.route_props.paths_matrix,
            key=lambda d: d["link_dict"]["free_slots"],
            reverse=True,
        )

        self.route_props.paths_matrix = [sorted_paths_list[0]["path_list"]]
        self.route_props.weights_list = [
            int(sorted_paths_list[0]["link_dict"]["free_slots"])
        ]

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

        network_spectrum_dict = getattr(self.sdn_props, "network_spectrum_dict", {})
        for i in range(len(path) - 1):
            link_key = (path[i], path[i + 1])
            if link_key in network_spectrum_dict:
                link_dict = network_spectrum_dict[link_key]
                for band in link_dict["cores_matrix"]:
                    cores_matrix = link_dict["cores_matrix"][band]
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
        # For congestion-aware routing, we typically return the single best path
        # But we can extend this to return multiple paths ordered by congestion
        best_path = self.route(source, destination, None)
        return [best_path] if best_path else []

    def update_weights(self, topology: Any) -> None:
        """
        Update congestion weights based on current network state.

        :param topology: NetworkX graph to update weights for.
        :type topology: Any
        """
        # Update congestion costs for all links based on current spectrum usage
        network_spectrum_dict = getattr(self.sdn_props, "network_spectrum_dict", {})
        for link_tuple in list(network_spectrum_dict.keys())[::2]:
            source, destination = link_tuple
            congestion = self._calculate_link_congestion(source, destination)

            # Update both directions
            if hasattr(topology, "edges"):
                topology[source][destination]["cong_cost"] = congestion
                topology[destination][source]["cong_cost"] = congestion

    def _calculate_link_congestion(self, source: Any, destination: Any) -> float:
        """
        Calculate congestion level for a specific link.

        :param source: Source node identifier.
        :type source: Any
        :param destination: Destination node identifier.
        :type destination: Any
        :return: Congestion ratio (used slots / total slots) for the link.
        :rtype: float
        """
        link_key = (source, destination)
        network_spectrum_dict = getattr(self.sdn_props, "network_spectrum_dict", {})
        if link_key not in network_spectrum_dict:
            return 0.0

        link_dict = network_spectrum_dict[link_key]
        total_used_slots = 0
        total_slots = 0

        for band in link_dict["cores_matrix"]:
            cores_matrix = link_dict["cores_matrix"][band]
            for core_arr in cores_matrix:
                total_slots += len(core_arr)
                total_used_slots += np.sum(core_arr != 0)

        return total_used_slots / total_slots if total_slots > 0 else 0.0

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
            "algorithm": self.algorithm_name,
            "paths_computed": self._path_count,
            "average_congestion": avg_congestion,
            "total_congestion_considered": self._total_congestion,
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_congestion = 0.0
        self.route_props = RoutingProps()
