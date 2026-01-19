"""
SNR measurement algorithms registry for FUSION.

This module provides a centralized registry for all SNR measurement algorithm
implementations that follow the AbstractSNRMeasurer interface.
"""

from typing import Any

from fusion.interfaces.snr import AbstractSNRMeasurer

# Import all SNR measurement algorithm implementations
from .snr import StandardSNRMeasurer


class SNRRegistry:
    """Registry for managing SNR measurement algorithm implementations."""

    def __init__(self) -> None:
        """Initialize the SNR measurement registry."""
        self._algorithms: dict[str, type[AbstractSNRMeasurer]] = {}
        self._register_default_algorithms()

    def _register_default_algorithms(self) -> None:
        """Register all default SNR measurement algorithms."""
        algorithms = [
            StandardSNRMeasurer,
        ]

        for algorithm_class in algorithms:
            # Create temporary instance to get algorithm name
            temp_instance = algorithm_class({}, None, None, None)
            self.register(temp_instance.algorithm_name, algorithm_class)

    def register(self, name: str, algorithm_class: type[AbstractSNRMeasurer]) -> None:
        """
        Register an SNR measurement algorithm.

        :param name: Unique name for the algorithm
        :param algorithm_class: Class that implements AbstractSNRMeasurer
        :raises TypeError: If algorithm_class doesn't implement AbstractSNRMeasurer
        :raises ValueError: If name is already registered
        """
        if not issubclass(algorithm_class, AbstractSNRMeasurer):
            raise TypeError(
                f"{algorithm_class.__name__} must implement AbstractSNRMeasurer"
            )

        if name in self._algorithms:
            raise ValueError(f"SNR algorithm '{name}' is already registered")

        self._algorithms[name] = algorithm_class

    def get(self, name: str) -> type[AbstractSNRMeasurer]:
        """
        Get an SNR measurement algorithm class by name.

        :param name: Name of the algorithm
        :return: Algorithm class that implements AbstractSNRMeasurer
        :raises KeyError: If algorithm is not found
        """
        if name not in self._algorithms:
            raise KeyError(
                f"SNR measurement algorithm '{name}' not found. "
                f"Available algorithms: {list(self._algorithms.keys())}"
            )

        return self._algorithms[name]

    def create(
        self,
        name: str,
        engine_props: dict,
        sdn_props: object,
        spectrum_props: object,
        route_props: object,
    ) -> AbstractSNRMeasurer:
        """
        Create an instance of an SNR measurement algorithm.

        :param name: Name of the algorithm
        :param engine_props: Engine configuration properties
        :param sdn_props: SDN controller properties
        :param spectrum_props: Spectrum assignment properties
        :param route_props: Routing properties
        :return: Configured SNR measurement algorithm instance
        """
        algorithm_class = self.get(name)
        return algorithm_class(engine_props, sdn_props, spectrum_props, route_props)

    def list_algorithms(self) -> list[str]:
        """
        List all registered algorithm names.

        :return: List of algorithm names
        """
        return list(self._algorithms.keys())

    def get_algorithm_info(self, name: str) -> dict[str, Any]:
        """
        Get information about a specific algorithm.

        :param name: Name of the algorithm
        :return: Dictionary with algorithm information
        """
        algorithm_class = self.get(name)

        # Create temporary instance to get properties
        temp_instance = algorithm_class({}, None, None, None)

        return {
            "name": name,
            "class": algorithm_class.__name__,
            "module": algorithm_class.__module__,
            "supports_multicore": temp_instance.supports_multicore,
            "description": (
                algorithm_class.__doc__.strip()
                if algorithm_class.__doc__
                else "No description"
            ),
        }

    def get_multicore_algorithms(self) -> list[str]:
        """
        Get list of algorithms that support multi-core fiber measurements.

        :return: List of algorithm names that support multi-core
        """
        multicore_algos = []

        for name, algorithm_class in self._algorithms.items():
            temp_instance = algorithm_class({}, None, None, None)
            if temp_instance.supports_multicore:
                multicore_algos.append(name)

        return multicore_algos


# Global registry instance
_registry = SNRRegistry()


# Convenience functions for global registry access
def register_snr_algorithm(
    name: str, algorithm_class: type[AbstractSNRMeasurer]
) -> None:
    """Register an SNR algorithm in the global registry."""
    _registry.register(name, algorithm_class)


def get_snr_algorithm(name: str) -> type[AbstractSNRMeasurer]:
    """Get an SNR algorithm class from the global registry."""
    return _registry.get(name)


def create_snr_algorithm(
    name: str,
    engine_props: dict,
    sdn_props: object,
    spectrum_props: object,
    route_props: object,
) -> AbstractSNRMeasurer:
    """Create an SNR algorithm instance from the global registry."""
    return _registry.create(name, engine_props, sdn_props, spectrum_props, route_props)


def list_snr_algorithms() -> list[str]:
    """List all available SNR measurement algorithms."""
    return _registry.list_algorithms()


def get_snr_algorithm_info(name: str) -> dict[str, Any]:
    """Get information about an SNR measurement algorithm."""
    return _registry.get_algorithm_info(name)


def get_multicore_snr_algorithms() -> list[str]:
    """Get algorithms that support multi-core fiber measurements."""
    return _registry.get_multicore_algorithms()


# Dictionary for backward compatibility
SNR_ALGORITHMS = {
    "standard_snr": StandardSNRMeasurer,
}
