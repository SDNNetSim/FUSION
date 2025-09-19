"""
Fragmentation-aware routing algorithm implementation.
"""
# pylint: disable=duplicate-code

from typing import List, Dict, Any, Optional
import networkx as nx

from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers
from fusion.core.properties import RoutingProps
from fusion.sim.utils import find_path_frag, find_path_len, get_path_mod


class FragmentationAwareRouting(AbstractRoutingAlgorithm):
    """Fragmentation-aware routing algorithm.
    
    This algorithm finds paths by considering spectrum fragmentation levels,
    selecting the path with the least fragmentation.
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        """Initialize fragmentation-aware routing algorithm.
        
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
        self._total_fragmentation = 0

    @property
    def algorithm_name(self) -> str:
        """Return the name of the routing algorithm."""
        return "fragmentation_aware"

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
        return (hasattr(topology, 'nodes') and
                hasattr(topology, 'edges') and
                hasattr(self.sdn_props, 'network_spectrum_dict'))

    def route(self, source: Any, destination: Any, request: Any) -> Optional[List[Any]]:
        """Find a route from source to destination considering fragmentation.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details
            
        Returns:
            Path with least fragmentation, or None if no path found
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
            self._find_least_weight('frag_cost')

            path = None
            if self.route_props.paths_matrix:
                path = self.route_props.paths_matrix[0]
                self._path_count += 1

                # Calculate fragmentation metric for this path
                fragmentation = self._calculate_path_fragmentation(path)
                self._total_fragmentation += fragmentation

            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _update_fragmentation_costs(self):
        """Update fragmentation costs for all links in the topology."""
        topology = self.engine_props.get('topology', self.sdn_props.topology)

        # Calculate fragmentation for each direct link
        for link_tuple in list(self.sdn_props.network_spectrum_dict.keys())[::2]:
            source, destination = link_tuple
            path_list = [source, destination]

            # Compute average fragmentation on this direct link
            frag_score = find_path_frag(path_list=path_list,
                                        network_spectrum_dict=self.sdn_props.network_spectrum_dict)

            # Store frag score for both directions if bidirectional
            if hasattr(topology, 'edges'):
                topology[source][destination]['frag_cost'] = frag_score
                topology[destination][source]['frag_cost'] = frag_score

    def _find_least_weight(self, weight: str):
        """Find the path with least weight (fragmentation cost)."""
        topology = self.engine_props.get('topology', self.sdn_props.topology)

        paths_obj = nx.shortest_simple_paths(G=topology,
                                             source=self.sdn_props.source,
                                             target=self.sdn_props.destination,
                                             weight=weight)

        for path_list in paths_obj:
            # Calculate path weight as sum across the path
            resp_weight = sum(topology[path_list[i]][path_list[i + 1]][weight]
                              for i in range(len(path_list) - 1))

            # Calculate actual path length for modulation format selection
            path_len = find_path_len(path_list=path_list, topology=topology)
            # Get modulation format
            mod_formats = getattr(self.sdn_props, 'mod_formats', {})
            mod_format = get_path_mod(mod_formats, path_len)
            mod_format_list = [mod_format if mod_format else 'QPSK']

            self.route_props.weights_list.append(resp_weight)
            self.route_props.paths_matrix.append(path_list)
            self.route_props.modulation_formats_matrix.append(mod_format_list)
            # For fragmentation-aware, we typically take the first (best) path
            break

    def _calculate_path_fragmentation(self, path: List[Any]) -> float:
        """Calculate fragmentation metric for a path."""
        if not path or len(path) < 2:
            return 0.0

        try:
            return find_path_frag(path_list=path,
                                  network_spectrum_dict=self.sdn_props.network_spectrum_dict)
        except (KeyError, AttributeError, TypeError):
            return 0.0

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> List[List[Any]]:
        """Get k paths ordered by fragmentation level.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            k: Number of paths to return
            
        Returns:
            List of k paths ordered by fragmentation (least fragmented first)
        """
        # Update fragmentation costs first
        topology = self.engine_props.get('topology', self.sdn_props.topology)
        self._update_fragmentation_costs()

        try:
            paths_generator = nx.shortest_simple_paths(topology, source, destination,
                                                       weight='frag_cost')

            paths = []
            for i, path in enumerate(paths_generator):
                if i >= k:
                    break
                paths.append(list(path))

            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def update_weights(self, topology: Any) -> None:
        """Update fragmentation weights based on current spectrum state.
        
        Args:
            topology: NetworkX graph to update weights for
        """
        # Recalculate fragmentation costs for all links
        for link_tuple in list(self.sdn_props.network_spectrum_dict.keys())[::2]:
            source, destination = link_tuple
            path_list = [source, destination]

            frag_score = find_path_frag(path_list=path_list,
                                        network_spectrum_dict=self.sdn_props.network_spectrum_dict)

            if hasattr(topology, 'edges'):
                topology[source][destination]['frag_cost'] = frag_score
                topology[destination][source]['frag_cost'] = frag_score

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing algorithm performance metrics.
        
        Returns:
            Dictionary containing algorithm-specific metrics
        """
        avg_fragmentation = self._total_fragmentation / self._path_count if self._path_count > 0 else 0

        return {
            'algorithm': self.algorithm_name,
            'paths_computed': self._path_count,
            'average_fragmentation': avg_fragmentation,
            'total_fragmentation_considered': self._total_fragmentation
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_fragmentation = 0
        self.route_props = RoutingProps()
