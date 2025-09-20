"""
Least congested link routing algorithm implementation.

This algorithm finds paths with minimum hops ± 1 and selects the one
with the least congested bottleneck link.
"""

# pylint: disable=duplicate-code

from typing import Any

import networkx as nx
import numpy as np

from fusion.core.properties import RoutingProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers


class LeastCongestedRouting(AbstractRoutingAlgorithm):
    """Least congested link routing algorithm.

    This algorithm finds paths with minimum hops ± 1 and selects the one
    with the least congested bottleneck link (not the mean congestion).
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        """Initialize least congested routing algorithm.

        Args:
            engine_props: Dictionary containing engine configuration
            sdn_props: Object containing SDN controller properties
        """
        super().__init__(engine_props, sdn_props)
        self.route_props = RoutingProps()
        self.route_help_obj = RoutingHelpers(
            route_props=self.route_props,
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
        )
        self._path_count = 0
        self._total_congestion = 0

    @property
    def algorithm_name(self) -> str:
        """Return the name of the routing algorithm."""
        return "least_congested"

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
        """Find the least congested path in the network.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details

        Returns:
            Path with least congested bottleneck link, or None if no path found
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
                best_path = self.route_props.paths_matrix[0]["path_list"]
                congestion = self._calculate_path_congestion(best_path)
                self._total_congestion += congestion
                return best_path
            return None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _find_least_congested_paths(self) -> None:
        """Find paths with minimum hops ± 1 and select least congested."""
        topology = self.engine_props.get("topology", self.sdn_props.topology)

        all_paths_generator = nx.shortest_simple_paths(
            topology, self.sdn_props.source, self.sdn_props.destination
        )
        minimum_hops = None

        for path_index, path_list in enumerate(all_paths_generator):
            hop_count = len(path_list)
            if path_index == 0:
                minimum_hops = hop_count
                self._find_most_cong_link(path_list=path_list)
            else:
                if hop_count <= minimum_hops + 1:
                    self._find_most_cong_link(path_list=path_list)
                # We exceeded minimum hops plus one, return the best path
                else:
                    self._select_least_congested()
                    return

        # If we exit the loop, select the best from what we have
        if self.route_props.paths_matrix:
            self._select_least_congested()

    def _find_most_cong_link(self, path_list: list) -> None:
        """Find the most congested link along a path."""
        most_congested_link = None
        most_congested_free_slots = -1

        for link_index in range(len(path_list) - 1):
            link_dict = self.sdn_props.network_spectrum_dict[
                (path_list[link_index], path_list[link_index + 1])
            ]
            total_free_slots = 0
            for band in link_dict["cores_matrix"]:
                cores_matrix = link_dict["cores_matrix"][band]
                for core_array in cores_matrix:
                    total_free_slots += np.sum(core_array == 0)

            if (
                total_free_slots < most_congested_free_slots
                or most_congested_link is None
            ):
                most_congested_free_slots = total_free_slots
                most_congested_link = link_dict

        self.route_props.paths_matrix.append(
            {
                "path_list": path_list,
                "link_dict": {
                    "link": most_congested_link,
                    "free_slots": most_congested_free_slots,
                },
            }
        )

    def _select_least_congested(self) -> None:
        """Select the path with the least congested link."""
        # Sort paths by number of free slots, descending (most free slots first)
        sorted_paths_list = sorted(
            self.route_props.paths_matrix,
            key=lambda path_data: path_data["link_dict"]["free_slots"],
            reverse=True,
        )

        # Keep all paths in the matrix (for backward compatibility with tests)
        self.route_props.paths_matrix = sorted_paths_list

        # But only populate weights and modulation for the best path
        best_path_data = sorted_paths_list[0]
        self.route_props.weights_list = [int(best_path_data["link_dict"]["free_slots"])]
        # Use QPSK as the default modulation format for least congested routing
        self.route_props.modulation_formats_matrix = [["QPSK"]]

    def _calculate_path_congestion(self, path: list[Any]) -> float:
        """Calculate congestion metric for a path."""
        if not path or len(path) < 2:
            return 0.0

        total_used_slots = 0
        total_slots = 0

        for i in range(len(path) - 1):
            link_key = (path[i], path[i + 1])
            if link_key in self.sdn_props.network_spectrum_dict:
                link_dict = self.sdn_props.network_spectrum_dict[link_key]
                for band in link_dict["cores_matrix"]:
                    cores_matrix = link_dict["cores_matrix"][band]
                    for core_arr in cores_matrix:
                        total_slots += len(core_arr)
                        total_used_slots += np.sum(core_arr != 0)

        return total_used_slots / total_slots if total_slots > 0 else 0.0

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """Get k paths ordered by congestion level.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            k: Number of paths to return

        Returns:
            List of k paths ordered by congestion (least congested first)
        """
        best_path = self.route(source, destination, None)
        return [best_path] if best_path else []

    def update_weights(self, topology: Any) -> None:
        """Update congestion weights based on current network state.

        Args:
            topology: NetworkX graph to update weights for
        """
        # For least congested, we don't pre-compute weights
        # since we evaluate congestion dynamically

    def get_metrics(self) -> dict[str, Any]:
        """Get routing algorithm performance metrics.

        Returns:
            Dictionary containing algorithm-specific metrics
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
        self._total_congestion = 0
        self.route_props = RoutingProps()
