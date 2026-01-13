"""
Unit tests for fusion.modules.snr.registry module.

This module tests the SNR algorithm registry functionality including:
- Algorithm registration and retrieval
- Algorithm creation and instantiation
- Multi-core algorithm filtering
- Global registry convenience functions
"""

from typing import Any
from unittest.mock import Mock

import pytest

from fusion.interfaces.snr import AbstractSNRMeasurer
from fusion.modules.snr.registry import (
    SNR_ALGORITHMS,
    SNRRegistry,
    create_snr_algorithm,
    get_multicore_snr_algorithms,
    get_snr_algorithm,
    get_snr_algorithm_info,
    list_snr_algorithms,
)
from fusion.modules.snr.snr import StandardSNRMeasurer


class MockSNRAlgorithm(AbstractSNRMeasurer):
    """Mock SNR algorithm for testing."""

    @property
    def algorithm_name(self) -> str:
        """Return mock algorithm name."""
        return "mock_snr"

    @property
    def supports_multicore(self) -> bool:
        """Return multicore support status."""
        return False

    def calculate_snr(self, path: list[Any], spectrum_info: dict[str, Any]) -> float:
        """Mock SNR calculation."""
        return 20.0

    def calculate_link_snr(
        self, source: Any, destination: Any, spectrum_info: dict[str, Any]
    ) -> float:
        """Mock link SNR calculation."""
        return 15.0

    def calculate_crosstalk(
        self, path: list[Any], core_num: int, spectrum_info: dict[str, Any]
    ) -> float:
        """Mock crosstalk calculation."""
        return 0.0

    def calculate_nonlinear_noise(
        self, path: list[Any], spectrum_info: dict[str, Any]
    ) -> dict[str, float]:
        """Mock nonlinear noise calculation."""
        return {"sci": 0.0, "xci": 0.0, "xpm": 0.0, "fwm": 0.0}

    def get_required_snr_threshold(self, modulation: str, reach: float) -> float:
        """Mock SNR threshold calculation."""
        return 12.0

    def is_snr_acceptable(
        self, calculated_snr: float, required_snr: float, margin: float = 0.0
    ) -> bool:
        """Mock SNR acceptability check."""
        return calculated_snr >= (required_snr + margin)

    def update_link_state(
        self, source: Any, destination: Any, spectrum_info: dict[str, Any]
    ) -> None:
        """Mock link state update."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Mock metrics retrieval."""
        return {"algorithm": "mock_snr"}


class MockMulticoreSNRAlgorithm(AbstractSNRMeasurer):
    """Mock multi-core SNR algorithm for testing."""

    @property
    def algorithm_name(self) -> str:
        """Return mock algorithm name."""
        return "mock_multicore_snr"

    @property
    def supports_multicore(self) -> bool:
        """Return multicore support status."""
        return True

    def calculate_snr(self, path: list[Any], spectrum_info: dict[str, Any]) -> float:
        """Mock SNR calculation."""
        return 25.0

    def calculate_link_snr(
        self, source: Any, destination: Any, spectrum_info: dict[str, Any]
    ) -> float:
        """Mock link SNR calculation."""
        return 18.0

    def calculate_crosstalk(
        self, path: list[Any], core_num: int, spectrum_info: dict[str, Any]
    ) -> float:
        """Mock crosstalk calculation."""
        return 0.5

    def calculate_nonlinear_noise(
        self, path: list[Any], spectrum_info: dict[str, Any]
    ) -> dict[str, float]:
        """Mock nonlinear noise calculation."""
        return {"sci": 0.1, "xci": 0.2, "xpm": 0.0, "fwm": 0.0}

    def get_required_snr_threshold(self, modulation: str, reach: float) -> float:
        """Mock SNR threshold calculation."""
        return 15.0

    def is_snr_acceptable(
        self, calculated_snr: float, required_snr: float, margin: float = 0.0
    ) -> bool:
        """Mock SNR acceptability check."""
        return calculated_snr >= (required_snr + margin)

    def update_link_state(
        self, source: Any, destination: Any, spectrum_info: dict[str, Any]
    ) -> None:
        """Mock link state update."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Mock metrics retrieval."""
        return {"algorithm": "mock_multicore_snr", "supports_multicore": True}


class TestSNRRegistry:
    """Tests for SNRRegistry class."""

    def test_registry_initialization_registers_default_algorithms(self) -> None:
        """Test that registry initializes with default algorithms."""
        # Arrange & Act
        registry = SNRRegistry()

        # Assert
        algorithms = registry.list_algorithms()
        assert "standard_snr" in algorithms
        assert len(algorithms) >= 1

    def test_register_with_valid_algorithm_succeeds(self) -> None:
        """Test registering a valid algorithm class."""
        # Arrange
        registry = SNRRegistry()

        # Act
        registry.register("test_algo", MockSNRAlgorithm)

        # Assert
        assert "test_algo" in registry.list_algorithms()

    def test_register_with_duplicate_name_raises_value_error(self) -> None:
        """Test that registering duplicate algorithm name raises ValueError."""
        # Arrange
        registry = SNRRegistry()
        registry.register("test_algo", MockSNRAlgorithm)

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            registry.register("test_algo", MockSNRAlgorithm)

        assert "already registered" in str(exc_info.value)
        assert "test_algo" in str(exc_info.value)

    def test_register_with_invalid_class_raises_type_error(self) -> None:
        """Test that registering non-AbstractSNRMeasurer class raises TypeError."""
        # Arrange
        registry = SNRRegistry()

        class InvalidClass:
            """Invalid class that doesn't implement AbstractSNRMeasurer."""

            pass

        # Act & Assert
        with pytest.raises(TypeError) as exc_info:
            registry.register("invalid", InvalidClass)  # type: ignore[arg-type]

        assert "must implement AbstractSNRMeasurer" in str(exc_info.value)

    def test_get_with_valid_name_returns_algorithm_class(self) -> None:
        """Test retrieving algorithm class by valid name."""
        # Arrange
        registry = SNRRegistry()

        # Act
        algorithm_class = registry.get("standard_snr")

        # Assert
        assert algorithm_class == StandardSNRMeasurer
        assert issubclass(algorithm_class, AbstractSNRMeasurer)

    def test_get_with_invalid_name_raises_key_error(self) -> None:
        """Test that retrieving non-existent algorithm raises KeyError."""
        # Arrange
        registry = SNRRegistry()

        # Act & Assert
        with pytest.raises(KeyError) as exc_info:
            registry.get("nonexistent_algo")

        assert "not found" in str(exc_info.value)
        assert "nonexistent_algo" in str(exc_info.value)
        assert "Available algorithms" in str(exc_info.value)

    def test_create_with_valid_name_returns_instance(self) -> None:
        """Test creating algorithm instance with valid name."""
        # Arrange
        registry = SNRRegistry()
        engine_props = {"input_power": 1e-3}
        sdn_props = Mock()
        spectrum_props = Mock()
        route_props = Mock()

        # Act
        instance = registry.create(
            "standard_snr", engine_props, sdn_props, spectrum_props, route_props
        )

        # Assert
        assert isinstance(instance, StandardSNRMeasurer)
        assert instance.algorithm_name == "standard_snr"
        assert instance.engine_props == engine_props

    def test_create_with_invalid_name_raises_key_error(self) -> None:
        """Test that creating instance with invalid name raises KeyError."""
        # Arrange
        registry = SNRRegistry()

        # Act & Assert
        with pytest.raises(KeyError):
            registry.create("invalid_algo", {}, None, None, None)

    def test_list_algorithms_returns_all_registered_names(self) -> None:
        """Test that list_algorithms returns all registered algorithm names."""
        # Arrange
        registry = SNRRegistry()
        registry.register("test_algo1", MockSNRAlgorithm)
        registry.register("test_algo2", MockMulticoreSNRAlgorithm)

        # Act
        algorithms = registry.list_algorithms()

        # Assert
        assert "standard_snr" in algorithms
        assert "test_algo1" in algorithms
        assert "test_algo2" in algorithms
        assert isinstance(algorithms, list)

    def test_get_algorithm_info_returns_correct_information(self) -> None:
        """Test that get_algorithm_info returns complete algorithm details."""
        # Arrange
        registry = SNRRegistry()

        # Act
        info = registry.get_algorithm_info("standard_snr")

        # Assert
        assert info["name"] == "standard_snr"
        assert info["class"] == "StandardSNRMeasurer"
        assert "module" in info
        assert "supports_multicore" in info
        assert "description" in info
        assert isinstance(info["supports_multicore"], bool)

    def test_get_algorithm_info_with_invalid_name_raises_key_error(self) -> None:
        """Test that get_algorithm_info with invalid name raises KeyError."""
        # Arrange
        registry = SNRRegistry()

        # Act & Assert
        with pytest.raises(KeyError):
            registry.get_algorithm_info("nonexistent")

    def test_get_multicore_algorithms_returns_only_multicore_support(self) -> None:
        """Test that get_multicore_algorithms filters correctly."""
        # Arrange
        registry = SNRRegistry()
        registry.register("single_core", MockSNRAlgorithm)
        registry.register("multi_core", MockMulticoreSNRAlgorithm)

        # Act
        multicore_algos = registry.get_multicore_algorithms()

        # Assert
        assert "standard_snr" in multicore_algos
        assert "multi_core" in multicore_algos
        assert "single_core" not in multicore_algos

    def test_get_multicore_algorithms_returns_empty_list_when_none_exist(self) -> None:
        """Test get_multicore_algorithms when no multicore algorithms exist."""
        # Arrange
        registry = SNRRegistry()
        # Clear default algorithms by creating fresh registry
        registry._algorithms = {}
        registry.register("single_core", MockSNRAlgorithm)

        # Act
        multicore_algos = registry.get_multicore_algorithms()

        # Assert
        assert multicore_algos == []


class TestGlobalRegistryFunctions:
    """Tests for global registry convenience functions."""

    def test_get_snr_algorithm_returns_correct_class(self) -> None:
        """Test get_snr_algorithm returns the correct algorithm class."""
        # Act
        algorithm_class = get_snr_algorithm("standard_snr")

        # Assert
        assert algorithm_class == StandardSNRMeasurer

    def test_create_snr_algorithm_returns_instance(self) -> None:
        """Test create_snr_algorithm creates algorithm instance."""
        # Arrange
        engine_props = {"input_power": 1e-3}
        sdn_props = Mock()
        spectrum_props = Mock()
        route_props = Mock()

        # Act
        instance = create_snr_algorithm(
            "standard_snr", engine_props, sdn_props, spectrum_props, route_props
        )

        # Assert
        assert isinstance(instance, StandardSNRMeasurer)
        assert instance.algorithm_name == "standard_snr"

    def test_list_snr_algorithms_returns_available_algorithms(self) -> None:
        """Test list_snr_algorithms returns list of algorithm names."""
        # Act
        algorithms = list_snr_algorithms()

        # Assert
        assert "standard_snr" in algorithms
        assert isinstance(algorithms, list)
        assert len(algorithms) >= 1

    def test_get_snr_algorithm_info_returns_algorithm_details(self) -> None:
        """Test get_snr_algorithm_info returns algorithm information."""
        # Act
        info = get_snr_algorithm_info("standard_snr")

        # Assert
        assert info["name"] == "standard_snr"
        assert "class" in info
        assert "supports_multicore" in info

    def test_get_multicore_snr_algorithms_returns_multicore_list(self) -> None:
        """Test get_multicore_snr_algorithms returns multicore algorithms."""
        # Act
        multicore_algos = get_multicore_snr_algorithms()

        # Assert
        assert "standard_snr" in multicore_algos
        assert isinstance(multicore_algos, list)

    def test_snr_algorithms_dict_contains_standard_algorithm(self) -> None:
        """Test SNR_ALGORITHMS dictionary contains standard algorithm."""
        # Assert
        assert "standard_snr" in SNR_ALGORITHMS
        assert SNR_ALGORITHMS["standard_snr"] == StandardSNRMeasurer


class TestSNRRegistryEdgeCases:
    """Tests for SNRRegistry edge cases and error handling."""

    def test_registry_handles_multiple_registrations_correctly(self) -> None:
        """Test registry handles multiple algorithm registrations."""
        # Arrange
        registry = SNRRegistry()
        algorithms_to_register = [
            ("algo1", MockSNRAlgorithm),
            ("algo2", MockMulticoreSNRAlgorithm),
        ]

        # Act
        for name, algo_class in algorithms_to_register:
            registry.register(name, algo_class)  # type: ignore[type-abstract]

        # Assert
        all_algos = registry.list_algorithms()
        assert "algo1" in all_algos
        assert "algo2" in all_algos

    def test_create_passes_all_parameters_to_algorithm(self) -> None:
        """Test that create passes all parameters correctly to algorithm."""
        # Arrange
        registry = SNRRegistry()
        engine_props = {"power": 100}
        sdn_props = Mock()
        sdn_props.network = "test"
        spectrum_props = Mock()
        spectrum_props.band = "c"
        route_props = Mock()
        route_props.path = [1, 2, 3]

        # Act
        instance = registry.create(
            "standard_snr", engine_props, sdn_props, spectrum_props, route_props
        )

        # Assert
        assert instance.engine_props == engine_props
        assert instance.sdn_props == sdn_props
        assert instance.spectrum_props == spectrum_props
        assert instance.route_props == route_props

    def test_get_algorithm_info_handles_missing_docstring(self) -> None:
        """Test get_algorithm_info handles algorithms without docstrings."""
        # Arrange
        registry = SNRRegistry()

        class NoDocAlgorithm(AbstractSNRMeasurer):
            """Test algorithm."""

            __doc__ = None

            @property
            def algorithm_name(self) -> str:
                return "no_doc"

            @property
            def supports_multicore(self) -> bool:
                return False

            def calculate_snr(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> float:
                return 0.0

            def calculate_link_snr(
                self, source: Any, destination: Any, spectrum_info: dict[str, Any]
            ) -> float:
                return 0.0

            def calculate_crosstalk(
                self, path: list[Any], core_num: int, spectrum_info: dict[str, Any]
            ) -> float:
                return 0.0

            def calculate_nonlinear_noise(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> dict[str, float]:
                return {}

            def get_required_snr_threshold(
                self, modulation: str, reach: float
            ) -> float:
                return 0.0

            def is_snr_acceptable(
                self, calculated_snr: float, required_snr: float, margin: float = 0.0
            ) -> bool:
                return True

            def update_link_state(
                self, source: Any, destination: Any, spectrum_info: dict[str, Any]
            ) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        registry.register("no_doc", NoDocAlgorithm)

        # Act
        info = registry.get_algorithm_info("no_doc")

        # Assert
        assert "description" in info
        assert info["description"] == "No description"
