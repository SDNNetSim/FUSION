"""
Routing algorithms registry for FUSION.

This module provides a centralized registry for all routing algorithm implementations
that follow the AbstractRoutingAlgorithm interface.
"""

from typing import Any

from fusion.interfaces.router import AbstractRoutingAlgorithm


class RoutingRegistry:
    """Registry for managing routing algorithm implementations."""

    def __init__(self) -> None:
        """Initialize the routing registry."""
        self._algorithms: dict[str, Any] = {}
        self._register_default_algorithms()

    def _register_default_algorithms(self) -> None:
        """
        Register all default routing algorithms.

        Registers the built-in routing algorithm implementations including
        k-shortest path, congestion aware, least congested, fragmentation aware,
        NLI aware, XT aware, and 1+1 protection algorithms.
        """
        # Import here to avoid circular dependency issues
        from .congestion_aware import CongestionAwareRouting
        from .fragmentation_aware import FragmentationAwareRouting
        from .k_shortest_path import KShortestPath
        from .least_congested import LeastCongestedRouting
        from .nli_aware import NLIAwareRouting
        from .one_plus_one_protection import OnePlusOneProtection
        from .xt_aware import XTAwareRouting

        algorithm_classes = [
            KShortestPath,
            CongestionAwareRouting,
            LeastCongestedRouting,
            FragmentationAwareRouting,
            NLIAwareRouting,
            XTAwareRouting,
            OnePlusOneProtection,
        ]

        # Map algorithm classes to their known names to avoid instantiation
        algorithm_name_mapping = {
            KShortestPath: "k_shortest_path",
            CongestionAwareRouting: "congestion_aware",
            LeastCongestedRouting: "least_congested",
            FragmentationAwareRouting: "fragmentation_aware",
            NLIAwareRouting: "nli_aware",
            XTAwareRouting: "xt_aware",
            OnePlusOneProtection: "1plus1_protection",
        }

        for algorithm_class in algorithm_classes:
            algorithm_name = algorithm_name_mapping.get(
                algorithm_class, algorithm_class.__name__.lower().replace("routing", "")
            )
            self.register(algorithm_name, algorithm_class)

    def register(self, name: str, algorithm_class: Any) -> None:
        """
        Register a routing algorithm.

        :param name: Unique name for the algorithm.
        :type name: str
        :param algorithm_class: Class that implements AbstractRoutingAlgorithm.
        :type algorithm_class: Any
        :raises TypeError: If algorithm_class doesn't implement
            AbstractRoutingAlgorithm.
        :raises ValueError: If name is already registered.
        """
        if not issubclass(algorithm_class, AbstractRoutingAlgorithm):
            raise TypeError(
                f"{algorithm_class.__name__} must implement AbstractRoutingAlgorithm"
            )

        if name in self._algorithms:
            raise ValueError(f"Algorithm '{name}' is already registered")

        self._algorithms[name] = algorithm_class

    def get(self, name: str) -> Any:
        """
        Get a routing algorithm class by name.

        :param name: Name of the algorithm.
        :type name: str
        :return: Algorithm class that implements AbstractRoutingAlgorithm.
        :rtype: Any
        :raises KeyError: If algorithm is not found.
        """
        if name not in self._algorithms:
            raise KeyError(
                f"Routing algorithm '{name}' not found. "
                f"Available algorithms: {list(self._algorithms.keys())}"
            )

        return self._algorithms[name]

    def create(self, name: str, engine_props: dict, sdn_props: object) -> Any:
        """
        Create an instance of a routing algorithm.

        :param name: Name of the algorithm.
        :type name: str
        :param engine_props: Engine configuration properties.
        :type engine_props: dict
        :param sdn_props: SDN controller properties.
        :type sdn_props: object
        :return: Configured routing algorithm instance.
        :rtype: Any
        """
        algorithm_class = self.get(name)
        return algorithm_class(engine_props, sdn_props)

    def list_algorithms(self) -> list[str]:
        """
        List all registered algorithm names.

        :return: List of algorithm names.
        :rtype: list[str]
        """
        return list(self._algorithms.keys())

    def get_algorithm_info(self, name: str) -> dict[str, str]:
        """
        Get information about a specific algorithm.

        :param name: Name of the algorithm.
        :type name: str
        :return: Dictionary with algorithm information including name, class,
            module, supported topologies, and description.
        :rtype: dict[str, str]
        """
        algorithm_class = self.get(name)

        # Create temporary instance to get properties
        try:
            temporary_instance = algorithm_class({}, None)
            supported_topologies = ", ".join(temporary_instance.supported_topologies)
        except Exception:
            supported_topologies = "Unknown"

        return {
            "name": name,
            "class": algorithm_class.__name__,
            "module": algorithm_class.__module__,
            "supported_topologies": supported_topologies,
            "description": (
                algorithm_class.__doc__.strip()
                if algorithm_class.__doc__
                else "No description"
            ),
        }

    def validate_algorithm(self, name: str, topology: Any) -> Any:
        """
        Validate that an algorithm can work with the given topology.

        :param name: Name of the algorithm.
        :type name: str
        :param topology: Network topology to validate against.
        :type topology: Any
        :return: True if algorithm supports the topology.
        :rtype: Any
        """
        algorithm_class = self.get(name)
        try:
            temporary_instance = algorithm_class({}, None)
            return temporary_instance.validate_environment(topology)
        except Exception:
            # If instantiation fails, assume algorithm is compatible
            return True


# Global registry instance - lazily initialized to avoid circular imports
_registry: RoutingRegistry | None = None


def _get_registry() -> RoutingRegistry:
    """
    Get the global registry instance, initializing if needed.

    Uses lazy initialization to avoid circular import issues that occur
    when the module is first loaded.

    :return: The global RoutingRegistry instance.
    :rtype: RoutingRegistry
    """
    global _registry
    if _registry is None:
        _registry = RoutingRegistry()
    return _registry


# Convenience functions for global registry access
def register_algorithm(name: str, algorithm_class: Any) -> None:
    """
    Register an algorithm in the global registry.

    :param name: Unique name for the algorithm.
    :type name: str
    :param algorithm_class: Class that implements AbstractRoutingAlgorithm.
    :type algorithm_class: Any
    :raises TypeError: If algorithm_class doesn't implement AbstractRoutingAlgorithm.
    :raises ValueError: If name is already registered.
    """
    _get_registry().register(name, algorithm_class)


def get_algorithm(name: str) -> Any:
    """
    Get an algorithm class from the global registry.

    :param name: Name of the algorithm.
    :type name: str
    :return: Algorithm class that implements AbstractRoutingAlgorithm.
    :rtype: Any
    :raises KeyError: If algorithm is not found.
    """
    return _get_registry().get(name)


def create_algorithm(name: str, engine_props: dict, sdn_props: object) -> Any:
    """
    Create an algorithm instance from the global registry.

    :param name: Name of the algorithm.
    :type name: str
    :param engine_props: Engine configuration properties.
    :type engine_props: dict
    :param sdn_props: SDN controller properties.
    :type sdn_props: object
    :return: Configured routing algorithm instance.
    :rtype: Any
    """
    return _get_registry().create(name, engine_props, sdn_props)


def list_routing_algorithms() -> list[str]:
    """
    List all available routing algorithms.

    :return: List of registered algorithm names.
    :rtype: list[str]
    """
    return _get_registry().list_algorithms()


def get_routing_algorithm_info(name: str) -> dict[str, str]:
    """
    Get information about a routing algorithm.

    :param name: Name of the algorithm.
    :type name: str
    :return: Dictionary with algorithm information including name, class,
        module, supported topologies, and description.
    :rtype: dict[str, str]
    """
    return _get_registry().get_algorithm_info(name)


# Dictionary for backward compatibility - populated from global registry
def _get_routing_algorithms_dict() -> dict[str, Any]:
    """Get dictionary of routing algorithms for backward compatibility."""
    reg = _get_registry()
    return {name: reg.get(name) for name in reg.list_algorithms()}


# Lazy-loaded dictionary using property-like access
class _RoutingAlgorithmsDict:
    """Lazy dictionary for backward compatibility with ROUTING_ALGORITHMS."""

    def __getitem__(self, key: str) -> Any:
        """Get algorithm class by name."""
        return _get_registry().get(key)

    def __contains__(self, key: str) -> bool:
        """Check if algorithm name exists."""
        return key in _get_registry().list_algorithms()

    def keys(self) -> list[str]:
        """Get all algorithm names."""
        return _get_registry().list_algorithms()

    def values(self) -> list[Any]:
        """Get all algorithm classes."""
        reg = _get_registry()
        return [reg.get(name) for name in reg.list_algorithms()]

    def items(self) -> list[tuple[str, Any]]:
        """Get all (name, class) pairs."""
        reg = _get_registry()
        return [(name, reg.get(name)) for name in reg.list_algorithms()]

    def get(self, key: str, default: Any = None) -> Any:
        """Get algorithm class with default."""
        try:
            return _get_registry().get(key)
        except KeyError:
            return default


ROUTING_ALGORITHMS = _RoutingAlgorithmsDict()
