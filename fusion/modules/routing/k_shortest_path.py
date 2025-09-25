"""
K-Shortest Path routing algorithm implementation.
"""
# pylint: disable=duplicate-code

from typing import Any

import networkx as nx

from fusion.core.properties import RoutingProps
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.sim.utils import (
    find_path_length,
    get_path_modulation,
    sort_nested_dict_values,
)


class KShortestPath(AbstractRoutingAlgorithm):
    """K-Shortest Path routing algorithm.

    This algorithm finds the k shortest paths between source and destination
    based on hop count or other specified weights.
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        """Initialize K-Shortest Path routing algorithm.

        Args:
            engine_props: Dictionary containing engine configuration
            sdn_props: Object containing SDN controller properties
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
        """Return the name of the routing algorithm."""
        return "k_shortest_path"

    @property
    def supported_topologies(self) -> list[str]:
        """Return list of supported topology types."""
        return ["NSFNet", "USBackbone60", "Pan-European", "Generic"]

    def validate_environment(self, topology: nx.Graph) -> bool:
        """Validate that the routing algorithm can work with the given topology.

        Args:
            topology: NetworkX graph representing the network topology

        Returns:
            True if the algorithm can route in this environment
        """
        # K-shortest path works with any connected graph
        return nx.is_connected(topology)

    def route(self, source: Any, destination: Any, request: Any) -> list[Any] | None:
        """Find a route from source to destination for the given request.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details

        Returns:
            Shortest available path, or None if no path found
        """
        # Clear previous route properties
        self.route_props.paths_matrix = []
        self.route_props.modulation_formats_matrix = []
        self.route_props.weights_list = []
        self.route_props.path_index_list = []
        self.route_props.connection_index = None

        paths = self.get_paths(source, destination, k=self.k_paths_count)

        if not paths:
            return None

        # Populate route_props for legacy compatibility
        topology = self.engine_props.get("topology", self.sdn_props.topology)

        for path in paths:
            # Calculate path length
            path_length = find_path_length(path_list=path, topology=topology)

            # Get modulation formats
            chosen_bandwidth = getattr(self.sdn_props, "bandwidth", None)
            if chosen_bandwidth and not self.engine_props.get(
                "pre_calc_mod_selection", False
            ):
                # Use mod_per_bw if available
                if (
                    "mod_per_bw" in self.engine_props
                    and chosen_bandwidth in self.engine_props["mod_per_bw"]
                ):
                    modulation_formats_list = [
                        get_path_modulation(
                            mods_dict=self.engine_props["mod_per_bw"][chosen_bandwidth],
                            path_len=path_length,
                        )
                    ]
                else:
                    # Fallback to mod_formats
                    modulation_formats = getattr(self.sdn_props, "mod_formats", {})
                    modulation_format = get_path_modulation(
                        modulation_formats, path_length
                    )
                    modulation_formats_list = [modulation_format]
            else:
                # Use all modulation formats sorted by max_length
                if hasattr(self.sdn_props, "modulation_formats_dict"):
                    modulation_formats_dict = sort_nested_dict_values(
                        original_dict=self.sdn_props.modulation_formats_dict,
                        nested_key="max_length",
                    )
                    modulation_formats_list = list(modulation_formats_dict.keys())[::-1]
                else:
                    # Fallback to simple list
                    modulation_formats_list = ["QPSK"]

            self.route_props.paths_matrix.append(path)
            self.route_props.modulation_formats_matrix.append(modulation_formats_list)
            self.route_props.weights_list.append(path_length)

        # Return the first (shortest) path
        selected_path = paths[0] if paths else None

        # Update metrics
        if selected_path:
            self._path_count += 1
            self._total_hops += len(selected_path) - 1

        return selected_path

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """Get k shortest paths between source and destination.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            k: Number of paths to return

        Returns:
            List of k paths, where each path is a list of nodes
        """
        topology = self.engine_props.get("topology", self.sdn_props.topology)

        try:
            if self.routing_weight:
                # Use specified weight for path calculation
                paths_generator = nx.shortest_simple_paths(
                    topology, source, destination, weight=self.routing_weight
                )
            else:
                # Use hop count (unweighted)
                paths_generator = nx.shortest_simple_paths(
                    topology, source, destination
                )

            paths_list = []
            for path_index, path in enumerate(paths_generator):
                if path_index >= k:
                    break
                paths_list.append(list(path))

            return paths_list

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def update_weights(self, topology: nx.Graph) -> None:
        """Update edge weights based on current network state.

        Args:
            topology: NetworkX graph to update weights for
        """
        # This implementation doesn't dynamically update weights
        # Subclasses can override this method for adaptive routing

    def get_metrics(self) -> dict[str, Any]:
        """Get routing algorithm performance metrics.

        Returns:
            Dictionary containing algorithm-specific metrics
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
