"""Unit tests for the SpectrumRegistry class."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from fusion.core.properties import SDNProps
from fusion.interfaces.spectrum import AbstractSpectrumAssigner
from fusion.modules.spectrum.best_fit import BestFitSpectrum
from fusion.modules.spectrum.first_fit import FirstFitSpectrum
from fusion.modules.spectrum.last_fit import LastFitSpectrum
from fusion.modules.spectrum.registry import (
    SPECTRUM_ALGORITHMS,
    SpectrumRegistry,
    create_spectrum_algorithm,
    get_multiband_spectrum_algorithms,
    get_spectrum_algorithm,
    get_spectrum_algorithm_info,
    list_spectrum_algorithms,
)


@pytest.fixture
def spectrum_registry() -> SpectrumRegistry:
    """Provide SpectrumRegistry instance for tests."""
    return SpectrumRegistry()


@pytest.fixture
def engine_props() -> dict[str, Any]:
    """Provide engine properties for tests."""
    return {"cores_per_link": 2, "guard_slots": 1, "band_list": ["c"]}


@pytest.fixture
def sdn_props() -> SDNProps:
    """Provide SDN properties for tests."""
    return SDNProps()


@pytest.fixture
def route_props() -> MagicMock:
    """Provide routing properties for tests."""
    return MagicMock()


class TestSpectrumRegistryInit:
    """Tests for SpectrumRegistry initialization."""

    def test_init_registers_default_algorithms(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test that default algorithms are registered on init."""
        # Act
        algorithms = spectrum_registry.list_algorithms()

        # Assert
        assert "first_fit" in algorithms
        assert "best_fit" in algorithms
        assert "last_fit" in algorithms


class TestRegisterMethod:
    """Tests for register method."""

    def test_register_with_valid_algorithm_succeeds(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test successful algorithm registration."""

        # Arrange
        class CustomAlgorithm(AbstractSpectrumAssigner):
            @property
            def algorithm_name(self) -> str:
                return "custom"

            @property
            def supports_multiband(self) -> bool:
                return False

            def assign(self, path: list[Any], request: Any) -> dict[str, Any] | None:
                return None

            def check_spectrum_availability(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return True

            def allocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
                request_id: Any,
            ) -> bool:
                return True

            def deallocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return True

            def get_fragmentation_metric(self, path: list[Any]) -> float:
                return 0.0

            def get_metrics(self) -> dict[str, Any]:
                return {}

            def reset(self) -> None:
                pass

        # Act
        spectrum_registry.register("custom", CustomAlgorithm)

        # Assert
        assert "custom" in spectrum_registry.list_algorithms()

    def test_register_with_duplicate_name_raises_error(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test that duplicate registration raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="already registered"):
            spectrum_registry.register("first_fit", FirstFitSpectrum)

    def test_register_with_non_abstract_class_raises_error(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test that non-AbstractSpectrumAssigner class raises TypeError."""

        # Arrange
        class InvalidAlgorithm:
            pass

        # Act & Assert
        with pytest.raises(TypeError, match="must implement AbstractSpectrumAssigner"):
            spectrum_registry.register("invalid", InvalidAlgorithm)  # type: ignore[arg-type]


class TestGetMethod:
    """Tests for get method."""

    def test_get_with_valid_name_returns_class(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test retrieving valid algorithm class."""
        # Act
        algorithm_class = spectrum_registry.get("first_fit")

        # Assert
        assert algorithm_class == FirstFitSpectrum

    def test_get_with_invalid_name_raises_error(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test that invalid algorithm name raises KeyError."""
        # Act & Assert
        with pytest.raises(KeyError, match="not found"):
            spectrum_registry.get("nonexistent")


class TestCreateMethod:
    """Tests for create method."""

    def test_create_with_valid_name_returns_instance(
        self,
        spectrum_registry: SpectrumRegistry,
        engine_props: dict[str, Any],
        sdn_props: SDNProps,
        route_props: MagicMock,
    ) -> None:
        """Test creating algorithm instance."""
        # Act
        instance = spectrum_registry.create("first_fit", engine_props, sdn_props, route_props)

        # Assert
        assert isinstance(instance, FirstFitSpectrum)
        assert instance.algorithm_name == "first_fit"

    def test_create_with_invalid_name_raises_error(
        self,
        spectrum_registry: SpectrumRegistry,
        engine_props: dict[str, Any],
        sdn_props: SDNProps,
        route_props: MagicMock,
    ) -> None:
        """Test creating instance with invalid name raises KeyError."""
        # Act & Assert
        with pytest.raises(KeyError, match="not found"):
            spectrum_registry.create("nonexistent", engine_props, sdn_props, route_props)


class TestListAlgorithms:
    """Tests for list_algorithms method."""

    def test_list_algorithms_returns_all_registered(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test listing all registered algorithms."""
        # Act
        algorithms = spectrum_registry.list_algorithms()

        # Assert
        assert isinstance(algorithms, list)
        assert len(algorithms) >= 3
        assert "first_fit" in algorithms
        assert "best_fit" in algorithms
        assert "last_fit" in algorithms


class TestGetAlgorithmInfo:
    """Tests for get_algorithm_info method."""

    def test_get_algorithm_info_returns_correct_details(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test retrieving algorithm information."""
        # Act
        info = spectrum_registry.get_algorithm_info("first_fit")

        # Assert
        assert info["name"] == "first_fit"
        assert info["class"] == "FirstFitSpectrum"
        assert "module" in info
        assert "supports_multiband" in info
        assert "description" in info

    def test_get_algorithm_info_with_invalid_name_raises_error(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test that invalid algorithm name raises KeyError."""
        # Act & Assert
        with pytest.raises(KeyError, match="not found"):
            spectrum_registry.get_algorithm_info("nonexistent")


class TestGetMultibandAlgorithms:
    """Tests for get_multiband_algorithms method."""

    def test_get_multiband_algorithms_returns_supporting_algorithms(self, spectrum_registry: SpectrumRegistry) -> None:
        """Test retrieving multiband-supporting algorithms."""
        # Act
        multiband_algos = spectrum_registry.get_multiband_algorithms()

        # Assert
        assert isinstance(multiband_algos, list)
        # All default algorithms support multiband
        assert "first_fit" in multiband_algos
        assert "best_fit" in multiband_algos
        assert "last_fit" in multiband_algos


class TestGlobalRegistryFunctions:
    """Tests for global registry access functions."""

    def test_get_spectrum_algorithm_returns_class(self) -> None:
        """Test global get function."""
        # Act
        algorithm_class = get_spectrum_algorithm("first_fit")

        # Assert
        assert algorithm_class == FirstFitSpectrum

    def test_list_spectrum_algorithms_returns_list(self) -> None:
        """Test global list function."""
        # Act
        algorithms = list_spectrum_algorithms()

        # Assert
        assert isinstance(algorithms, list)
        assert "first_fit" in algorithms

    def test_get_spectrum_algorithm_info_returns_dict(self) -> None:
        """Test global info function."""
        # Act
        info = get_spectrum_algorithm_info("first_fit")

        # Assert
        assert isinstance(info, dict)
        assert info["name"] == "first_fit"

    def test_get_multiband_spectrum_algorithms_returns_list(self) -> None:
        """Test global multiband function."""
        # Act
        multiband_algos = get_multiband_spectrum_algorithms()

        # Assert
        assert isinstance(multiband_algos, list)
        assert "first_fit" in multiband_algos

    def test_create_spectrum_algorithm_returns_instance(
        self,
        engine_props: dict[str, Any],
        sdn_props: SDNProps,
        route_props: MagicMock,
    ) -> None:
        """Test global create function."""
        # Act
        instance = create_spectrum_algorithm("first_fit", engine_props, sdn_props, route_props)

        # Assert
        assert isinstance(instance, FirstFitSpectrum)


class TestSpectrumAlgorithmsDict:
    """Tests for SPECTRUM_ALGORITHMS backward compatibility dict."""

    def test_spectrum_algorithms_dict_contains_all_algorithms(self) -> None:
        """Test that SPECTRUM_ALGORITHMS dict is properly populated."""
        # Assert
        assert "first_fit" in SPECTRUM_ALGORITHMS
        assert "best_fit" in SPECTRUM_ALGORITHMS
        assert "last_fit" in SPECTRUM_ALGORITHMS

    def test_spectrum_algorithms_dict_maps_to_correct_classes(self) -> None:
        """Test that dict maps to correct algorithm classes."""
        # Assert
        assert SPECTRUM_ALGORITHMS["first_fit"] == FirstFitSpectrum
        assert SPECTRUM_ALGORITHMS["best_fit"] == BestFitSpectrum
        assert SPECTRUM_ALGORITHMS["last_fit"] == LastFitSpectrum
