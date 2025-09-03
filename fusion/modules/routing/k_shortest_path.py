"""
K-Shortest Path routing algorithm implementation.
"""
# pylint: disable=duplicate-code

from typing import List, Dict, Any, Optional
import networkx as nx

from fusion.interfaces.router import AbstractRoutingAlgorithm


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
        self.k = engine_props.get('k_paths', 3)
        self.weight = engine_props.get('routing_weight', None)
        self._path_count = 0
        self._total_hops = 0

    @property
    def algorithm_name(self) -> str:
        """Return the name of the routing algorithm."""
        return "k_shortest_path"

    @property
    def supported_topologies(self) -> List[str]:
        """Return list of supported topology types."""
        return ['NSFNet', 'USBackbone60', 'Pan-European', 'Generic']

    def validate_environment(self, topology: nx.Graph) -> bool:
        """Validate that the routing algorithm can work with the given topology.
        
        Args:
            topology: NetworkX graph representing the network topology
            
        Returns:
            True if the algorithm can route in this environment
        """
        # K-shortest path works with any connected graph
        return nx.is_connected(topology)

    def route(self, source: Any, destination: Any, request: Any) -> Optional[List[Any]]:
        """Find a route from source to destination for the given request.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details
            
        Returns:
            Shortest available path, or None if no path found
        """
        paths = self.get_paths(source, destination, k=self.k)

        if not paths:
            return None

        # Return the first (shortest) path
        # In a more sophisticated implementation, we might check
        # resource availability along each path
        selected_path = paths[0]

        # Update metrics
        self._path_count += 1
        self._total_hops += len(selected_path) - 1

        return selected_path

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> List[List[Any]]:
        """Get k shortest paths between source and destination.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            k: Number of paths to return
            
        Returns:
            List of k paths, where each path is a list of nodes
        """
        topology = self.engine_props.get('topology', self.sdn_props.topology)

        try:
            if self.weight:
                # Use specified weight for path calculation
                paths_generator = nx.shortest_simple_paths(
                    topology, source, destination, weight=self.weight
                )
            else:
                # Use hop count (unweighted)
                paths_generator = nx.shortest_simple_paths(
                    topology, source, destination
                )

            paths = []
            for i, path in enumerate(paths_generator):
                if i >= k:
                    break
                paths.append(list(path))

            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def update_weights(self, topology: nx.Graph) -> None:
        """Update edge weights based on current network state.
        
        Args:
            topology: NetworkX graph to update weights for
        """
        # This implementation doesn't dynamically update weights
        # Subclasses can override this method for adaptive routing

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing algorithm performance metrics.
        
        Returns:
            Dictionary containing algorithm-specific metrics
        """
        avg_hops = self._total_hops / self._path_count if self._path_count > 0 else 0

        return {
            'algorithm': self.algorithm_name,
            'paths_computed': self._path_count,
            'average_hop_count': avg_hops,
            'k_value': self.k,
            'weight_metric': self.weight or 'hop_count'
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_hops = 0
