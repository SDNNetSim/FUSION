"""
Routing algorithms registry for FUSION.

This module provides a centralized registry for all routing algorithm implementations
that follow the AbstractRoutingAlgorithm interface.
"""

from typing import Dict, Type, List
from fusion.interfaces.router import AbstractRoutingAlgorithm

# Import all routing algorithm implementations
from .k_shortest_path import KShortestPath
from .congestion_aware import CongestionAwareRouting
from .least_congested import LeastCongestedRouting
from .fragmentation_aware import FragmentationAwareRouting
from .nli_aware import NLIAwareRouting
from .xt_aware import XTAwareRouting


class RoutingRegistry:
    """Registry for managing routing algorithm implementations."""

    def __init__(self):
        """Initialize the routing registry."""
        self._algorithms: Dict[str, Type[AbstractRoutingAlgorithm]] = {}
        self._register_default_algorithms()

    def _register_default_algorithms(self):
        """Register all default routing algorithms."""
        algorithm_classes = [
            KShortestPath,
            CongestionAwareRouting,
            LeastCongestedRouting,
            FragmentationAwareRouting,
            NLIAwareRouting,
            XTAwareRouting
        ]

        for algorithm_class in algorithm_classes:
            # Create temporary instance to get algorithm name
            temporary_instance = algorithm_class({}, None)
            self.register(temporary_instance.algorithm_name, algorithm_class)

    def register(self, name: str, algorithm_class: Type[AbstractRoutingAlgorithm]):
        """Register a routing algorithm.
        
        Args:
            name: Unique name for the algorithm
            algorithm_class: Class that implements AbstractRoutingAlgorithm
            
        Raises:
            TypeError: If algorithm_class doesn't implement AbstractRoutingAlgorithm
            ValueError: If name is already registered
        """
        if not issubclass(algorithm_class, AbstractRoutingAlgorithm):
            raise TypeError(f"{algorithm_class.__name__} must implement AbstractRoutingAlgorithm")

        if name in self._algorithms:
            raise ValueError(f"Algorithm '{name}' is already registered")

        self._algorithms[name] = algorithm_class

    def get(self, name: str) -> Type[AbstractRoutingAlgorithm]:
        """Get a routing algorithm class by name.
        
        Args:
            name: Name of the algorithm
            
        Returns:
            Algorithm class that implements AbstractRoutingAlgorithm
            
        Raises:
            KeyError: If algorithm is not found
        """
        if name not in self._algorithms:
            raise KeyError(f"Routing algorithm '{name}' not found. "
                           f"Available algorithms: {list(self._algorithms.keys())}")

        return self._algorithms[name]

    def create(self, name: str, engine_props: dict, sdn_props: object) -> AbstractRoutingAlgorithm:
        """Create an instance of a routing algorithm.
        
        Args:
            name: Name of the algorithm
            engine_props: Engine configuration properties
            sdn_props: SDN controller properties
            
        Returns:
            Configured routing algorithm instance
        """
        algorithm_class = self.get(name)
        return algorithm_class(engine_props, sdn_props)

    def list_algorithms(self) -> List[str]:
        """List all registered algorithm names.
        
        Returns:
            List of algorithm names
        """
        return list(self._algorithms.keys())

    def get_algorithm_info(self, name: str) -> Dict[str, str]:
        """Get information about a specific algorithm.
        
        Args:
            name: Name of the algorithm
            
        Returns:
            Dictionary with algorithm information
        """
        algorithm_class = self.get(name)

        # Create temporary instance to get properties
        temporary_instance = algorithm_class({}, None)

        return {
            'name': name,
            'class': algorithm_class.__name__,
            'module': algorithm_class.__module__,
            'supported_topologies': ', '.join(temporary_instance.supported_topologies),
            'description': algorithm_class.__doc__.strip() if algorithm_class.__doc__ else 'No description'
        }

    def validate_algorithm(self, name: str, topology) -> bool:
        """Validate that an algorithm can work with the given topology.
        
        Args:
            name: Name of the algorithm
            topology: Network topology to validate against
            
        Returns:
            True if algorithm supports the topology
        """
        algorithm_class = self.get(name)
        temporary_instance = algorithm_class({}, None)
        return temporary_instance.validate_environment(topology)


# Global registry instance
_registry = RoutingRegistry()


# Convenience functions for global registry access
def register_algorithm(name: str, algorithm_class: Type[AbstractRoutingAlgorithm]):
    """Register an algorithm in the global registry."""
    _registry.register(name, algorithm_class)


def get_algorithm(name: str) -> Type[AbstractRoutingAlgorithm]:
    """Get an algorithm class from the global registry."""
    return _registry.get(name)


def create_algorithm(name: str, engine_props: dict, sdn_props: object) -> AbstractRoutingAlgorithm:
    """Create an algorithm instance from the global registry."""
    return _registry.create(name, engine_props, sdn_props)


def list_routing_algorithms() -> List[str]:
    """List all available routing algorithms."""
    return _registry.list_algorithms()


def get_routing_algorithm_info(name: str) -> Dict[str, str]:
    """Get information about a routing algorithm."""
    return _registry.get_algorithm_info(name)


# Dictionary for backward compatibility
ROUTING_ALGORITHMS = {
    'k_shortest_path': KShortestPath,
    'congestion_aware': CongestionAwareRouting,
    'least_congested': LeastCongestedRouting,
    'fragmentation_aware': FragmentationAwareRouting,
    'nli_aware': NLIAwareRouting,
    'xt_aware': XTAwareRouting
}
