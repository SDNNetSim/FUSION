"""
Cross-talk (XT) aware routing algorithm implementation.
"""
# pylint: disable=duplicate-code

from typing import List, Dict, Any, Optional
import networkx as nx

from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.modules.routing.utils import RoutingHelpers
from fusion.core.properties import RoutingProps
from fusion.sim.utils import find_free_slots, find_path_len, get_path_mod, sort_nested_dict_vals


class XTAwareRouting(AbstractRoutingAlgorithm):
    """Cross-talk aware routing algorithm.
    
    This algorithm finds paths by considering intra-core crosstalk interference,
    selecting the path with the least amount of cross-talk.
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        """Initialize XT-aware routing algorithm.
        
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
        self._total_xt = 0

    @property
    def algorithm_name(self) -> str:
        """Return the name of the routing algorithm."""
        return "xt_aware"

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
        """Find a route from source to destination considering cross-talk.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details
            
        Returns:
            Path with least cross-talk, or None if no path found
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
            self._find_least_weight('xt_cost')

            path = None
            if self.route_props.paths_matrix:
                path = self.route_props.paths_matrix[0]
                self._path_count += 1

                # Calculate XT metric for this path
                xt = self._calculate_path_xt(path)
                self._total_xt += xt

            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _update_xt_costs(self):
        """Update cross-talk costs for all links in the topology."""
        topology = self.engine_props.get('topology', self.sdn_props.topology)

        # At the moment, we have identical bidirectional links (no need to loop over all links)
        for link_list in list(self.sdn_props.network_spectrum_dict.keys())[::2]:
            source, destination = link_list[0], link_list[1]
            num_spans = topology[source][destination]['length'] / self.route_props.span_length

            free_slots_dict = find_free_slots(network_spectrum_dict=self.sdn_props.network_spectrum_dict,
                                              link_tuple=link_list)
            xt_cost = self.route_help_obj.find_xt_link_cost(free_slots_dict=free_slots_dict,
                                                            link_list=link_list)

            # Consider XT type configuration
            if self.engine_props.get('xt_type') == 'with_length':
                if self.route_props.max_link_length is None:
                    self.route_help_obj.get_max_link_length()

                link_cost = topology[source][destination]['length'] / self.route_props.max_link_length
                link_cost *= self.engine_props.get('beta', 0.5)
                link_cost += (1 - self.engine_props.get('beta', 0.5)) * xt_cost
            elif self.engine_props.get('xt_type') == 'without_length':
                link_cost = num_spans * xt_cost
            else:
                # Default behavior
                link_cost = xt_cost

            if hasattr(topology, 'edges'):
                topology[source][destination]['xt_cost'] = link_cost
                topology[destination][source]['xt_cost'] = link_cost

    def _find_least_weight(self, weight: str):
        """Find the path with least weight (XT cost)."""
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

            # Get modulation formats based on path length
            if hasattr(self.sdn_props, 'modulation_formats_dict'):
                # Use sorted modulation formats
                mod_formats_dict = sort_nested_dict_vals(
                    original_dict=self.sdn_props.modulation_formats_dict,
                    nested_key='max_length'
                )
                mod_format_list = []
                for mod_format in mod_formats_dict:
                    if self.sdn_props.modulation_formats_dict[mod_format]['max_length'] >= path_len:
                        mod_format_list.append(mod_format)
                    else:
                        mod_format_list.append(False)
            else:
                # Fallback to simple modulation selection
                mod_formats = getattr(self.sdn_props, 'mod_formats', {})
                mod_format = get_path_mod(mod_formats, path_len)
                mod_format_list = [mod_format if mod_format else 'QPSK']

            self.route_props.weights_list.append(resp_weight)
            self.route_props.paths_matrix.append(path_list)
            self.route_props.modulation_formats_matrix.append(mod_format_list)
            # For XT-aware, we typically take the first (best) path
            break

    def _calculate_path_xt(self, path: List[Any]) -> float:
        """Calculate cross-talk metric for a path."""
        if not path or len(path) < 2:
            return 0.0

        topology = self.engine_props.get('topology', self.sdn_props.topology)
        total_xt = 0.0

        for i in range(len(path) - 1):
            source, destination = path[i], path[i + 1]
            if hasattr(topology, 'edges') and topology.has_edge(source, destination):
                xt_cost = topology[source][destination].get('xt_cost', 0.0)
                total_xt += xt_cost

        return total_xt

    def get_paths(self, source: Any, destination: Any, k: int = 1) -> List[List[Any]]:
        """Get k paths ordered by cross-talk level.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            k: Number of paths to return
            
        Returns:
            List of k paths ordered by cross-talk (least XT first)
        """
        # Update XT costs first
        self._update_xt_costs()
        topology = self.engine_props.get('topology', self.sdn_props.topology)

        try:
            paths_generator = nx.shortest_simple_paths(topology, source, destination,
                                                       weight='xt_cost')

            paths = []
            for i, path in enumerate(paths_generator):
                if i >= k:
                    break
                paths.append(list(path))

            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def update_weights(self, topology: Any) -> None:
        """Update cross-talk weights based on current spectrum state.
        
        Args:
            topology: NetworkX graph to update weights for
        """
        # Recalculate XT costs for all links
        for link_list in list(self.sdn_props.network_spectrum_dict.keys())[::2]:
            source, destination = link_list[0], link_list[1]
            num_spans = topology[source][destination]['length'] / self.route_props.span_length

            free_slots_dict = find_free_slots(network_spectrum_dict=self.sdn_props.network_spectrum_dict,
                                              link_tuple=link_list)
            xt_cost = self.route_help_obj.find_xt_link_cost(free_slots_dict=free_slots_dict,
                                                            link_list=link_list)

            # Apply XT type configuration
            if self.engine_props.get('xt_type') == 'with_length':
                if self.route_props.max_link_length is None:
                    self.route_help_obj.get_max_link_length()

                link_cost = topology[source][destination]['length'] / self.route_props.max_link_length
                link_cost *= self.engine_props.get('beta', 0.5)
                link_cost += (1 - self.engine_props.get('beta', 0.5)) * xt_cost
            elif self.engine_props.get('xt_type') == 'without_length':
                link_cost = num_spans * xt_cost
            else:
                link_cost = xt_cost

            if hasattr(topology, 'edges'):
                topology[source][destination]['xt_cost'] = link_cost
                topology[destination][source]['xt_cost'] = link_cost

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing algorithm performance metrics.
        
        Returns:
            Dictionary containing algorithm-specific metrics
        """
        avg_xt = self._total_xt / self._path_count if self._path_count > 0 else 0

        return {
            'algorithm': self.algorithm_name,
            'paths_computed': self._path_count,
            'average_xt': avg_xt,
            'total_xt_considered': self._total_xt,
            'xt_type': self.engine_props.get('xt_type', 'default')
        }

    def reset(self) -> None:
        """Reset the routing algorithm state."""
        self._path_count = 0
        self._total_xt = 0
        self.route_props = RoutingProps()
