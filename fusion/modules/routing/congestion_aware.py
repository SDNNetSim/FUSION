"""
Congestion-aware routing algorithm implementation.
"""
# pylint: disable=duplicate-code

from typing import List, Dict, Any, Optional
import networkx as nx
import numpy as np

from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers
from fusion.core.properties import RoutingProps


class CongestionAwareRouting(AbstractRoutingAlgorithm):
    """Congestion-aware routing algorithm.
    
    This algorithm finds paths by considering network congestion levels,
    selecting the path with the least congested link.
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        """Initialize congestion-aware routing algorithm.
        
        Args:
            engine_props: Dictionary containing engine configuration
            sdn_props: Object containing SDN controller properties
        """
        super().__init__(engine_props, sdn_props)
        self.route_props = RoutingProps()
        self.route_help_obj = RoutingHelpers(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            route_props=self.route_props
        )
        self._path_count = 0
        self._total_congestion = 0

    @property
    def algorithm_name(self) -> str:
        """Return the name of the routing algorithm."""
        return "congestion_aware"

    @property
    def supported_topologies(self) -> List[str]:
        """Return list of supported topology types."""
        return ['NSFNet', 'USBackbone60', 'Pan-European', 'Generic']

    def validate_environment(self, topology: Any) -> bool:
        """Validate that the routing algorithm can work with the given topology.
        
        Args:
            topology: NetworkX graph representing the network topology
            
        Returns:
            True if the algorithm can route in this environment
        """
        # Check if topology has the required attributes for congestion calculation
        return (hasattr(topology, 'nodes') and
                hasattr(topology, 'edges') and
                hasattr(self.sdn_props, 'net_spec_dict'))

    def route(self, source: Any, destination: Any, request: Any) -> Optional[List[Any]]:
        """Find a route from source to destination considering congestion.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details
            
        Returns:
            Least congested path, or None if no path found
        """
        # Store source/destination in sdn_props for compatibility
        self.sdn_props.source = source
        self.sdn_props.destination = destination

        # Reset paths matrix for new calculation
        self.route_props.paths_matrix = []

        try:
            path = self._find_least_congested_path()
            if path:
                self._path_count += 1
                # Calculate congestion metric for this path
                congestion = self._calculate_path_congestion(path)
                self._total_congestion += congestion

            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _find_least_congested_path(self) -> Optional[List[Any]]:
        """Find the least congested path using the original algorithm logic."""
        topology = self.engine_props.get('topology', self.sdn_props.topology)

        all_paths_obj = nx.shortest_simple_paths(topology, self.sdn_props.source,
                                                 self.sdn_props.destination)
        min_hops = None

        for i, path_list in enumerate(all_paths_obj):
            num_hops = len(path_list)
            if i == 0:
                min_hops = num_hops
                self._find_most_cong_link(path_list=path_list)
            else:
                if num_hops <= min_hops + 1:
                    self._find_most_cong_link(path_list=path_list)
                else:
                    # We exceeded minimum hops plus one, return the best path
                    self._find_least_cong()
                    if self.route_props.paths_matrix:
                        return self.route_props.paths_matrix[0]
                    break

        return None

    def _find_most_cong_link(self, path_list: list):
        """Find the most congested link along a path."""
        most_cong_link = None
        most_cong_slots = -1

        for i in range(len(path_list) - 1):
            link_dict = self.sdn_props.net_spec_dict[(path_list[i], path_list[i + 1])]
            free_slots = 0
            for band in link_dict['cores_matrix']:
                cores_matrix = link_dict['cores_matrix'][band]
                for core_arr in cores_matrix:
                    free_slots += np.sum(core_arr == 0)

            if free_slots < most_cong_slots or most_cong_link is None:
                most_cong_slots = free_slots
                most_cong_link = link_dict

        self.route_props.paths_matrix.append({
            'path_list': path_list,
            'link_dict': {'link': most_cong_link, 'free_slots': most_cong_slots}
        })

    def _find_least_cong(self):
        """Select the path with the least congested link."""
        # Sort dictionary by number of free slots, descending
        sorted_paths_list = sorted(self.route_props.paths_matrix,
                                   key=lambda d: d['link_dict']['free_slots'],
                                   reverse=True)

        self.route_props.paths_matrix = [sorted_paths_list[0]['path_list']]
        self.route_props.weights_list = [int(sorted_paths_list[0]['link_dict']['free_slots'])]

    def _calculate_path_congestion(self, path: List[Any]) -> float:
        """Calculate congestion metric for a path."""
        if not path or len(path) < 2:
            return 0.0

        total_used_slots = 0
        total_slots = 0

        for i in range(len(path) - 1):
            link_key = (path[i], path[i + 1])
            if link_key in self.sdn_props.net_spec_dict:
                link_dict = self.sdn_props.net_spec_dict[link_key]
                for band in link_dict['cores_matrix']:
                    cores_matrix = link_dict['cores_matrix'][band]
                    for core_arr in cores_matrix:
                        total_slots += len(core_arr)
                        total_used_slots += np.sum(core_arr != 0)

        return total_used_slots / total_slots if total_slots > 0 else 0.0

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> List[List[Any]]:
        """Get k paths ordered by congestion level.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            k: Number of paths to return
            
        Returns:
            List of k paths ordered by congestion (least congested first)
        """
        # For congestion-aware routing, we typically return the single best path
        # But we can extend this to return multiple paths ordered by congestion
        best_path = self.route(source, destination, None)
        return [best_path] if best_path else []

    def update_weights(self, topology: Any) -> None:
        """Update congestion weights based on current network state.
        
        Args:
            topology: NetworkX graph to update weights for
        """
        # Update congestion costs for all links based on current spectrum usage
        for link_tuple in list(self.sdn_props.net_spec_dict.keys())[::2]:
            source, destination = link_tuple
            congestion = self._calculate_link_congestion(source, destination)

            # Update both directions
            if hasattr(topology, 'edges'):
                topology[source][destination]['cong_cost'] = congestion
                topology[destination][source]['cong_cost'] = congestion

    def _calculate_link_congestion(self, source: Any, destination: Any) -> float:
        """Calculate congestion level for a specific link."""
        link_key = (source, destination)
        if link_key not in self.sdn_props.net_spec_dict:
            return 0.0

        link_dict = self.sdn_props.net_spec_dict[link_key]
        total_used_slots = 0
        total_slots = 0

        for band in link_dict['cores_matrix']:
            cores_matrix = link_dict['cores_matrix'][band]
            for core_arr in cores_matrix:
                total_slots += len(core_arr)
                total_used_slots += np.sum(core_arr != 0)

        return total_used_slots / total_slots if total_slots > 0 else 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing algorithm performance metrics.
        
        Returns:
            Dictionary containing algorithm-specific metrics
        """
        avg_congestion = self._total_congestion / self._path_count if self._path_count > 0 else 0

        return {
            'algorithm': self.algorithm_name,
            'paths_computed': self._path_count,
            'average_congestion': avg_congestion,
            'total_congestion_considered': self._total_congestion
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_congestion = 0
        self.route_props = RoutingProps()
