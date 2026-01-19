"""
Unit tests for fusion.interfaces.snr module.

Tests the AbstractSNRMeasurer abstract base class for SNR measurement algorithms.
"""

import inspect
from typing import Any
from unittest.mock import Mock

import pytest

from fusion.interfaces.snr import AbstractSNRMeasurer

# ============================================================================
# Test Abstract Interface Instantiation
# ============================================================================


class TestAbstractSNRMeasurerInstantiation:
    """Tests that AbstractSNRMeasurer cannot be directly instantiated."""

    def test_abstract_snr_measurer_cannot_be_instantiated(self) -> None:
        """Test AbstractSNRMeasurer cannot be directly instantiated."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError):
            AbstractSNRMeasurer({}, None, None, None)  # type: ignore[abstract,arg-type]


# ============================================================================
# Test Abstract Methods
# ============================================================================


class TestAbstractSNRMeasurerAbstractMethods:
    """Tests that required methods are marked as abstract."""

    def test_snr_measurer_has_correct_abstract_methods(self) -> None:
        """Test that AbstractSNRMeasurer has correct abstract methods."""
        # Arrange
        expected_methods = {
            "algorithm_name",
            "supports_multicore",
            "calculate_snr",
            "calculate_link_snr",
            "calculate_crosstalk",
            "calculate_nonlinear_noise",
            "get_required_snr_threshold",
            "is_snr_acceptable",
            "update_link_state",
            "get_metrics",
        }

        # Act
        abstract_methods = {
            method
            for method in dir(AbstractSNRMeasurer)
            if hasattr(getattr(AbstractSNRMeasurer, method), "__isabstractmethod__")
            and getattr(AbstractSNRMeasurer, method).__isabstractmethod__
        }

        # Assert
        assert abstract_methods == expected_methods


# ============================================================================
# Test Interface Method Signatures
# ============================================================================


class TestAbstractSNRMeasurerMethodSignatures:
    """Tests that AbstractSNRMeasurer method signatures are consistent."""

    def test_calculate_snr_method_signature(self) -> None:
        """Test AbstractSNRMeasurer.calculate_snr method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractSNRMeasurer.calculate_snr)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "path", "spectrum_info"]
        assert sig.return_annotation is float

    def test_calculate_link_snr_method_signature(self) -> None:
        """Test AbstractSNRMeasurer.calculate_link_snr method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractSNRMeasurer.calculate_link_snr)
        params = list(sig.parameters.keys())

        # Assert
        assert params == [
            "self",
            "source",
            "destination",
            "spectrum_info",
        ]
        assert sig.return_annotation is float

    def test_calculate_crosstalk_method_signature(self) -> None:
        """Test AbstractSNRMeasurer.calculate_crosstalk method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractSNRMeasurer.calculate_crosstalk)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "path", "core_num", "spectrum_info"]
        assert sig.return_annotation is float

    def test_is_snr_acceptable_method_signature(self) -> None:
        """Test AbstractSNRMeasurer.is_snr_acceptable method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractSNRMeasurer.is_snr_acceptable)
        params = list(sig.parameters.keys())

        # Assert
        assert params == [
            "self",
            "calculated_snr",
            "required_snr",
            "margin",
        ]
        assert sig.parameters["margin"].default == 0.0
        assert sig.return_annotation is bool

    def test_get_required_snr_threshold_method_signature(self) -> None:
        """Test get_required_snr_threshold method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractSNRMeasurer.get_required_snr_threshold)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "modulation", "reach"]
        assert sig.return_annotation is float

    def test_update_link_state_method_signature(self) -> None:
        """Test AbstractSNRMeasurer.update_link_state method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractSNRMeasurer.update_link_state)
        params = list(sig.parameters.keys())

        # Assert
        assert params == [
            "self",
            "source",
            "destination",
            "spectrum_info",
        ]


# ============================================================================
# Test Required Methods
# ============================================================================


class TestAbstractSNRMeasurerRequiredMethods:
    """Tests that AbstractSNRMeasurer has all required methods."""

    def test_snr_measurer_has_all_required_methods(self) -> None:
        """Test AbstractSNRMeasurer has all required methods."""
        # Arrange
        expected_methods = [
            "algorithm_name",
            "supports_multicore",
            "calculate_snr",
            "calculate_link_snr",
            "calculate_crosstalk",
            "calculate_nonlinear_noise",
            "get_required_snr_threshold",
            "is_snr_acceptable",
            "update_link_state",
            "get_metrics",
            "reset",
        ]

        # Act & Assert
        for method in expected_methods:
            assert hasattr(AbstractSNRMeasurer, method)


# ============================================================================
# Test Initialization
# ============================================================================


class TestAbstractSNRMeasurerInitialization:
    """Tests for AbstractSNRMeasurer initialization."""

    def test_initialization_stores_props(self) -> None:
        """Test that initialization stores props correctly."""

        # Arrange
        class ConcreteSNRMeasurer(AbstractSNRMeasurer):
            @property
            def algorithm_name(self) -> str:
                return "test_snr"

            @property
            def supports_multicore(self) -> bool:
                return False

            def calculate_snr(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> float:
                return 20.0

            def calculate_link_snr(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 25.0

            def calculate_crosstalk(
                self,
                path: list[Any],
                core_num: int,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 0.01

            def calculate_nonlinear_noise(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> dict[str, float]:
                return {"sci": 0.1, "xci": 0.05}

            def get_required_snr_threshold(
                self, modulation: str, reach: float
            ) -> float:
                return 15.0

            def is_snr_acceptable(
                self,
                calculated_snr: float,
                required_snr: float,
                margin: float = 0.0,
            ) -> bool:
                return calculated_snr >= (required_snr + margin)

            def update_link_state(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        engine_props = {"key": "value"}
        sdn_props = Mock()
        spectrum_props = Mock()
        route_props = Mock()

        # Act
        algo = ConcreteSNRMeasurer(engine_props, sdn_props, spectrum_props, route_props)

        # Assert
        assert algo.engine_props == engine_props
        assert algo.sdn_props == sdn_props
        assert algo.spectrum_props == spectrum_props
        assert algo.route_props == route_props


# ============================================================================
# Test Concrete Implementation
# ============================================================================


class TestConcreteSNRMeasurerImplementation:
    """Tests for concrete implementation of AbstractSNRMeasurer."""

    def test_concrete_snr_measurer_can_be_instantiated(self) -> None:
        """Test concrete implementation with all methods can be instantiated."""

        # Arrange
        class ConcreteSNRMeasurer(AbstractSNRMeasurer):
            @property
            def algorithm_name(self) -> str:
                return "standard_snr"

            @property
            def supports_multicore(self) -> bool:
                return True

            def calculate_snr(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> float:
                return 20.5

            def calculate_link_snr(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 22.0

            def calculate_crosstalk(
                self,
                path: list[Any],
                core_num: int,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 0.02

            def calculate_nonlinear_noise(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> dict[str, float]:
                return {"sci": 0.15, "xci": 0.08, "xpm": 0.03}

            def get_required_snr_threshold(
                self, modulation: str, reach: float
            ) -> float:
                return 18.0

            def is_snr_acceptable(
                self,
                calculated_snr: float,
                required_snr: float,
                margin: float = 0.0,
            ) -> bool:
                return calculated_snr >= (required_snr + margin)

            def update_link_state(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {"measurements": 50}

        engine_props: dict[str, Any] = {}
        sdn_props = Mock()
        spectrum_props = Mock()
        route_props = Mock()

        # Act
        algo = ConcreteSNRMeasurer(engine_props, sdn_props, spectrum_props, route_props)

        # Assert
        assert algo.algorithm_name == "standard_snr"
        assert algo.supports_multicore is True
        assert algo.calculate_snr([1, 2, 3], {}) == 20.5
        assert algo.get_metrics() == {"measurements": 50}

    def test_concrete_snr_measurer_missing_method_cannot_be_instantiated(
        self,
    ) -> None:
        """Test concrete implementation missing abstract methods."""

        # Arrange
        class IncompleteSNRMeasurer(AbstractSNRMeasurer):
            @property
            def algorithm_name(self) -> str:
                return "incomplete"

            # Missing other required abstract methods

        # Act & Assert
        with pytest.raises(TypeError):
            IncompleteSNRMeasurer({}, Mock(), Mock(), Mock())  # type: ignore[abstract]


# ============================================================================
# Test Reset Method
# ============================================================================


class TestAbstractSNRMeasurerReset:
    """Tests for the reset method."""

    def test_reset_method_has_default_implementation(self) -> None:
        """Test that reset method has default implementation."""

        # Arrange
        class ConcreteSNRMeasurer(AbstractSNRMeasurer):
            @property
            def algorithm_name(self) -> str:
                return "test_snr"

            @property
            def supports_multicore(self) -> bool:
                return False

            def calculate_snr(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> float:
                return 20.0

            def calculate_link_snr(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 25.0

            def calculate_crosstalk(
                self,
                path: list[Any],
                core_num: int,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 0.01

            def calculate_nonlinear_noise(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> dict[str, float]:
                return {}

            def get_required_snr_threshold(
                self, modulation: str, reach: float
            ) -> float:
                return 15.0

            def is_snr_acceptable(
                self,
                calculated_snr: float,
                required_snr: float,
                margin: float = 0.0,
            ) -> bool:
                return True

            def update_link_state(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        # Act
        algo = ConcreteSNRMeasurer({}, Mock(), Mock(), Mock())
        algo.reset()

        # Assert - reset doesn't return a value
        assert True


# ============================================================================
# Test Property Return Types
# ============================================================================


class TestAbstractSNRMeasurerPropertyReturnTypes:
    """Tests for property return types."""

    def test_algorithm_name_returns_string(self) -> None:
        """Test that algorithm_name property returns string."""
        # Arrange
        # For abstract properties, access the function directly
        prop = AbstractSNRMeasurer.algorithm_name
        if isinstance(prop, property) and prop.fget is not None:
            sig = inspect.signature(prop.fget)
        else:
            # Fallback for abstractmethod properties
            sig = inspect.signature(prop.fget)  # type: ignore[union-attr,attr-defined]

        # Assert
        assert sig.return_annotation is str

    def test_supports_multicore_returns_bool(self) -> None:
        """Test that supports_multicore property returns bool."""
        # Arrange
        sig = inspect.signature(
            AbstractSNRMeasurer.supports_multicore.fget  # type: ignore[attr-defined]
        )

        # Assert
        assert sig.return_annotation is bool


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestAbstractSNRMeasurerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_is_snr_acceptable_with_margin(self) -> None:
        """Test is_snr_acceptable correctly applies margin."""

        # Arrange
        class ConcreteSNRMeasurer(AbstractSNRMeasurer):
            @property
            def algorithm_name(self) -> str:
                return "test_snr"

            @property
            def supports_multicore(self) -> bool:
                return False

            def calculate_snr(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> float:
                return 20.0

            def calculate_link_snr(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 25.0

            def calculate_crosstalk(
                self,
                path: list[Any],
                core_num: int,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 0.01

            def calculate_nonlinear_noise(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> dict[str, float]:
                return {}

            def get_required_snr_threshold(
                self, modulation: str, reach: float
            ) -> float:
                return 15.0

            def is_snr_acceptable(
                self,
                calculated_snr: float,
                required_snr: float,
                margin: float = 0.0,
            ) -> bool:
                return calculated_snr >= (required_snr + margin)

            def update_link_state(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteSNRMeasurer({}, Mock(), Mock(), Mock())

        # Act
        result_without_margin = algo.is_snr_acceptable(20.0, 15.0, 0.0)
        result_with_margin = algo.is_snr_acceptable(20.0, 15.0, 10.0)

        # Assert
        assert result_without_margin is True
        assert result_with_margin is False

    def test_calculate_nonlinear_noise_returns_dict(self) -> None:
        """Test calculate_nonlinear_noise returns dict with noise components."""

        # Arrange
        class ConcreteSNRMeasurer(AbstractSNRMeasurer):
            @property
            def algorithm_name(self) -> str:
                return "test_snr"

            @property
            def supports_multicore(self) -> bool:
                return False

            def calculate_snr(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> float:
                return 20.0

            def calculate_link_snr(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 25.0

            def calculate_crosstalk(
                self,
                path: list[Any],
                core_num: int,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 0.01

            def calculate_nonlinear_noise(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> dict[str, float]:
                return {"sci": 0.1, "xci": 0.05, "xpm": 0.02, "fwm": 0.01}

            def get_required_snr_threshold(
                self, modulation: str, reach: float
            ) -> float:
                return 15.0

            def is_snr_acceptable(
                self,
                calculated_snr: float,
                required_snr: float,
                margin: float = 0.0,
            ) -> bool:
                return True

            def update_link_state(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteSNRMeasurer({}, Mock(), Mock(), Mock())

        # Act
        result = algo.calculate_nonlinear_noise([1, 2, 3], {})

        # Assert
        assert isinstance(result, dict)
        assert "sci" in result
        assert "xci" in result
        assert "xpm" in result
        assert "fwm" in result

    def test_get_required_snr_threshold_varies_by_modulation(self) -> None:
        """Test get_required_snr_threshold can vary by modulation format."""

        # Arrange
        class ConcreteSNRMeasurer(AbstractSNRMeasurer):
            @property
            def algorithm_name(self) -> str:
                return "test_snr"

            @property
            def supports_multicore(self) -> bool:
                return False

            def calculate_snr(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> float:
                return 20.0

            def calculate_link_snr(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 25.0

            def calculate_crosstalk(
                self,
                path: list[Any],
                core_num: int,
                spectrum_info: dict[str, Any],
            ) -> float:
                return 0.01

            def calculate_nonlinear_noise(
                self, path: list[Any], spectrum_info: dict[str, Any]
            ) -> dict[str, float]:
                return {}

            def get_required_snr_threshold(
                self, modulation: str, reach: float
            ) -> float:
                thresholds = {"QPSK": 10.0, "16QAM": 15.0, "64QAM": 20.0}
                return thresholds.get(modulation, 15.0)

            def is_snr_acceptable(
                self,
                calculated_snr: float,
                required_snr: float,
                margin: float = 0.0,
            ) -> bool:
                return True

            def update_link_state(
                self,
                source: Any,
                destination: Any,
                spectrum_info: dict[str, Any],
            ) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteSNRMeasurer({}, Mock(), Mock(), Mock())

        # Act
        qpsk_threshold = algo.get_required_snr_threshold("QPSK", 1000.0)
        qam16_threshold = algo.get_required_snr_threshold("16QAM", 1000.0)
        qam64_threshold = algo.get_required_snr_threshold("64QAM", 1000.0)

        # Assert
        assert qpsk_threshold == 10.0
        assert qam16_threshold == 15.0
        assert qam64_threshold == 20.0
        assert qpsk_threshold < qam16_threshold < qam64_threshold
