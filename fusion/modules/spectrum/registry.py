"""
Spectrum assignment algorithms registry for FUSION.

This module provides a centralized registry for all spectrum assignment algorithm
implementations that follow the AbstractSpectrumAssigner interface.
"""

from typing import Any

from fusion.interfaces.spectrum import AbstractSpectrumAssigner

from .best_fit import BestFitSpectrum

# Import all spectrum assignment algorithm implementations
from .first_fit import FirstFitSpectrum
from .last_fit import LastFitSpectrum


class SpectrumRegistry:
    """Registry for managing spectrum assignment algorithm implementations."""

    def __init__(self):
        """Initialize the spectrum assignment registry."""
        self._algorithms: dict[str, type[AbstractSpectrumAssigner]] = {}
        self._register_default_algorithms()

    def _register_default_algorithms(self):
        """Register all default spectrum assignment algorithms."""
        algorithms = [FirstFitSpectrum, BestFitSpectrum, LastFitSpectrum]

        for algorithm_class in algorithms:
            # Create temporary instance to get algorithm name
            temp_instance = algorithm_class({}, None, None)
            self.register(temp_instance.algorithm_name, algorithm_class)

    def register(self, name: str, algorithm_class: type[AbstractSpectrumAssigner]):
        """Register a spectrum assignment algorithm.

        Args:
            name: Unique name for the algorithm
            algorithm_class: Class that implements AbstractSpectrumAssigner

        Raises:
            TypeError: If algorithm_class doesn't implement AbstractSpectrumAssigner
            ValueError: If name is already registered
        """
        if not issubclass(algorithm_class, AbstractSpectrumAssigner):
            raise TypeError(
                f"{algorithm_class.__name__} must implement AbstractSpectrumAssigner"
            )

        if name in self._algorithms:
            raise ValueError(f"Spectrum algorithm '{name}' is already registered")

        self._algorithms[name] = algorithm_class

    def get(self, name: str) -> type[AbstractSpectrumAssigner]:
        """Get a spectrum assignment algorithm class by name.

        Args:
            name: Name of the algorithm

        Returns:
            Algorithm class that implements AbstractSpectrumAssigner

        Raises:
            KeyError: If algorithm is not found
        """
        if name not in self._algorithms:
            raise KeyError(
                f"Spectrum assignment algorithm '{name}' not found. "
                f"Available algorithms: {list(self._algorithms.keys())}"
            )

        return self._algorithms[name]

    def create(
        self, name: str, engine_props: dict, sdn_props: object, route_props: object
    ) -> AbstractSpectrumAssigner:
        """Create an instance of a spectrum assignment algorithm.

        Args:
            name: Name of the algorithm
            engine_props: Engine configuration properties
            sdn_props: SDN controller properties
            route_props: Routing properties

        Returns:
            Configured spectrum assignment algorithm instance
        """
        algorithm_class = self.get(name)
        return algorithm_class(engine_props, sdn_props, route_props)

    def list_algorithms(self) -> list[str]:
        """List all registered algorithm names.

        Returns:
            List of algorithm names
        """
        return list(self._algorithms.keys())

    def get_algorithm_info(self, name: str) -> dict[str, Any]:
        """Get information about a specific algorithm.

        Args:
            name: Name of the algorithm

        Returns:
            Dictionary with algorithm information
        """
        algorithm_class = self.get(name)

        # Create temporary instance to get properties
        temp_instance = algorithm_class({}, None, None)

        return {
            "name": name,
            "class": algorithm_class.__name__,
            "module": algorithm_class.__module__,
            "supports_multiband": temp_instance.supports_multiband,
            "description": (
                algorithm_class.__doc__.strip()
                if algorithm_class.__doc__
                else "No description"
            ),
        }

    def get_multiband_algorithms(self) -> list[str]:
        """Get list of algorithms that support multi-band assignment.

        Returns:
            List of algorithm names that support multi-band
        """
        multiband_algos = []

        for name, algorithm_class in self._algorithms.items():
            temp_instance = algorithm_class({}, None, None)
            if temp_instance.supports_multiband:
                multiband_algos.append(name)

        return multiband_algos


# Global registry instance
_registry = SpectrumRegistry()


# Convenience functions for global registry access
def register_spectrum_algorithm(
    name: str, algorithm_class: type[AbstractSpectrumAssigner]
):
    """Register a spectrum algorithm in the global registry."""
    _registry.register(name, algorithm_class)


def get_spectrum_algorithm(name: str) -> type[AbstractSpectrumAssigner]:
    """Get a spectrum algorithm class from the global registry."""
    return _registry.get(name)


def create_spectrum_algorithm(
    name: str, engine_props: dict, sdn_props: object, route_props: object
) -> AbstractSpectrumAssigner:
    """Create a spectrum algorithm instance from the global registry."""
    return _registry.create(name, engine_props, sdn_props, route_props)


def list_spectrum_algorithms() -> list[str]:
    """List all available spectrum assignment algorithms."""
    return _registry.list_algorithms()


def get_spectrum_algorithm_info(name: str) -> dict[str, Any]:
    """Get information about a spectrum assignment algorithm."""
    return _registry.get_algorithm_info(name)


def get_multiband_spectrum_algorithms() -> list[str]:
    """Get algorithms that support multi-band assignment."""
    return _registry.get_multiband_algorithms()


# Dictionary for backward compatibility
SPECTRUM_ALGORITHMS = {
    "first_fit": FirstFitSpectrum,
    "best_fit": BestFitSpectrum,
    "last_fit": LastFitSpectrum,
}
