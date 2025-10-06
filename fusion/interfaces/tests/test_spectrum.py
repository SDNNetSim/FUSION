"""
Unit tests for fusion.interfaces.spectrum module.

Tests the AbstractSpectrumAssigner abstract base class for spectrum
assignment algorithms.
"""

import inspect
from typing import Any
from unittest.mock import Mock

import pytest

from fusion.interfaces.spectrum import AbstractSpectrumAssigner

# ============================================================================
# Test Abstract Interface Instantiation
# ============================================================================


class TestAbstractSpectrumAssignerInstantiation:
    """Tests that AbstractSpectrumAssigner cannot be directly instantiated."""

    def test_abstract_spectrum_assigner_cannot_be_instantiated(self) -> None:
        """Test AbstractSpectrumAssigner cannot be directly instantiated."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError):
            AbstractSpectrumAssigner({}, None, None)  # type: ignore[abstract,arg-type]


# ============================================================================
# Test Abstract Methods
# ============================================================================


class TestAbstractSpectrumAssignerAbstractMethods:
    """Tests that required methods are marked as abstract."""

    def test_spectrum_assigner_has_correct_abstract_methods(self) -> None:
        """Test AbstractSpectrumAssigner has correct abstract methods."""
        # Arrange
        expected_methods = {
            "algorithm_name",
            "supports_multiband",
            "assign",
            "check_spectrum_availability",
            "allocate_spectrum",
            "deallocate_spectrum",
            "get_fragmentation_metric",
            "get_metrics",
        }

        # Act
        abstract_methods = {
            method
            for method in dir(AbstractSpectrumAssigner)
            if hasattr(
                getattr(AbstractSpectrumAssigner, method), "__isabstractmethod__"
            )
            and getattr(
                AbstractSpectrumAssigner, method
            ).__isabstractmethod__
        }

        # Assert
        assert abstract_methods == expected_methods


# ============================================================================
# Test Interface Method Signatures
# ============================================================================


class TestAbstractSpectrumAssignerMethodSignatures:
    """Tests that AbstractSpectrumAssigner method signatures are consistent."""

    def test_assign_method_signature(self) -> None:
        """Test AbstractSpectrumAssigner.assign method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractSpectrumAssigner.assign)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "path", "request"]
        annotation_str = str(sig.return_annotation)
        assert "None" in annotation_str or "Optional" in annotation_str

    def test_check_spectrum_availability_method_signature(self) -> None:
        """Test check_spectrum_availability method signature."""
        # Arrange & Act
        sig = inspect.signature(
            AbstractSpectrumAssigner.check_spectrum_availability
        )
        params = list(sig.parameters.keys())

        # Assert
        assert params == [
            "self",
            "path",
            "start_slot",
            "end_slot",
            "core_num",
            "band",
        ]
        assert sig.return_annotation is bool

    def test_allocate_spectrum_method_signature(self) -> None:
        """Test AbstractSpectrumAssigner.allocate_spectrum signature."""
        # Arrange & Act
        sig = inspect.signature(
            AbstractSpectrumAssigner.allocate_spectrum
        )
        params = list(sig.parameters.keys())

        # Assert
        assert params == [
            "self",
            "path",
            "start_slot",
            "end_slot",
            "core_num",
            "band",
            "request_id",
        ]
        assert sig.return_annotation is bool

    def test_deallocate_spectrum_method_signature(self) -> None:
        """Test AbstractSpectrumAssigner.deallocate_spectrum signature."""
        # Arrange & Act
        sig = inspect.signature(
            AbstractSpectrumAssigner.deallocate_spectrum
        )
        params = list(sig.parameters.keys())

        # Assert
        assert params == [
            "self",
            "path",
            "start_slot",
            "end_slot",
            "core_num",
            "band",
        ]
        assert sig.return_annotation is bool

    def test_get_fragmentation_metric_method_signature(self) -> None:
        """Test get_fragmentation_metric method signature."""
        # Arrange & Act
        sig = inspect.signature(
            AbstractSpectrumAssigner.get_fragmentation_metric
        )
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "path"]
        assert sig.return_annotation is float


# ============================================================================
# Test Required Methods
# ============================================================================


class TestAbstractSpectrumAssignerRequiredMethods:
    """Tests that AbstractSpectrumAssigner has all required methods."""

    def test_spectrum_assigner_has_all_required_methods(self) -> None:
        """Test AbstractSpectrumAssigner has all required methods."""
        # Arrange
        expected_methods = [
            "algorithm_name",
            "supports_multiband",
            "assign",
            "check_spectrum_availability",
            "allocate_spectrum",
            "deallocate_spectrum",
            "get_fragmentation_metric",
            "get_metrics",
            "reset",
        ]

        # Act & Assert
        for method in expected_methods:
            assert hasattr(AbstractSpectrumAssigner, method)


# ============================================================================
# Test Initialization
# ============================================================================


class TestAbstractSpectrumAssignerInitialization:
    """Tests for AbstractSpectrumAssigner initialization."""

    def test_initialization_stores_props(self) -> None:
        """Test that initialization stores props correctly."""

        # Arrange
        class ConcreteSpectrumAssigner(AbstractSpectrumAssigner):
            @property
            def algorithm_name(self) -> str:
                return "test_spectrum"

            @property
            def supports_multiband(self) -> bool:
                return False

            def assign(
                self, path: list[Any], request: Any
            ) -> dict[str, Any] | None:
                return None

            def check_spectrum_availability(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def allocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
                request_id: Any,
            ) -> bool:
                return False

            def deallocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def get_fragmentation_metric(self, path: list[Any]) -> float:
                return 0.0

            def get_metrics(self) -> dict[str, Any]:
                return {}

        engine_props = {"key": "value"}
        sdn_props = Mock()
        route_props = Mock()

        # Act
        algo = ConcreteSpectrumAssigner(engine_props, sdn_props, route_props)

        # Assert
        assert algo.engine_props == engine_props
        assert algo.sdn_props == sdn_props
        assert algo.route_props == route_props


# ============================================================================
# Test Concrete Implementation
# ============================================================================


class TestConcreteSpectrumAssignerImplementation:
    """Tests for concrete implementation of AbstractSpectrumAssigner."""

    def test_concrete_spectrum_assigner_can_be_instantiated(self) -> None:
        """Test concrete implementation with all methods can be instantiated."""

        # Arrange
        class ConcreteSpectrumAssigner(AbstractSpectrumAssigner):
            @property
            def algorithm_name(self) -> str:
                return "first_fit"

            @property
            def supports_multiband(self) -> bool:
                return True

            def assign(
                self, path: list[Any], request: Any
            ) -> dict[str, Any] | None:
                return {
                    "start_slot": 0,
                    "end_slot": 5,
                    "core_num": 0,
                    "band": "C",
                    "is_free": True,
                }

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
                return 0.25

            def get_metrics(self) -> dict[str, Any]:
                return {"allocations": 10}

        engine_props: dict[str, Any] = {}
        sdn_props = Mock()
        route_props = Mock()

        # Act
        algo = ConcreteSpectrumAssigner(engine_props, sdn_props, route_props)

        # Assert
        assert algo.algorithm_name == "first_fit"
        assert algo.supports_multiband is True
        assert algo.get_fragmentation_metric([1, 2, 3]) == 0.25
        assert algo.get_metrics() == {"allocations": 10}

    def test_concrete_spectrum_assigner_missing_method_cannot_be_instantiated(
        self,
    ) -> None:
        """Test concrete implementation missing abstract methods."""

        # Arrange
        class IncompleteSpectrumAssigner(AbstractSpectrumAssigner):
            @property
            def algorithm_name(self) -> str:
                return "incomplete"

            # Missing other required abstract methods

        # Act & Assert
        with pytest.raises(TypeError):
            IncompleteSpectrumAssigner({}, Mock(), Mock())  # type: ignore[abstract]


# ============================================================================
# Test Reset Method
# ============================================================================


class TestAbstractSpectrumAssignerReset:
    """Tests for the reset method."""

    def test_reset_method_has_default_implementation(self) -> None:
        """Test that reset method has default implementation."""

        # Arrange
        class ConcreteSpectrumAssigner(AbstractSpectrumAssigner):
            @property
            def algorithm_name(self) -> str:
                return "test_spectrum"

            @property
            def supports_multiband(self) -> bool:
                return False

            def assign(
                self, path: list[Any], request: Any
            ) -> dict[str, Any] | None:
                return None

            def check_spectrum_availability(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def allocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
                request_id: Any,
            ) -> bool:
                return False

            def deallocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def get_fragmentation_metric(self, path: list[Any]) -> float:
                return 0.0

            def get_metrics(self) -> dict[str, Any]:
                return {}

        # Act
        algo = ConcreteSpectrumAssigner({}, Mock(), Mock())
        algo.reset()

        # Assert - reset doesn't return a value
        assert True


# ============================================================================
# Test Property Return Types
# ============================================================================


class TestAbstractSpectrumAssignerPropertyReturnTypes:
    """Tests for property return types."""

    def test_algorithm_name_returns_string(self) -> None:
        """Test that algorithm_name property returns string."""
        # Arrange
        # For abstract properties, access the function directly
        prop = AbstractSpectrumAssigner.algorithm_name
        if isinstance(prop, property) and prop.fget is not None:
            sig = inspect.signature(prop.fget)
        else:
            # Fallback for abstractmethod properties
            sig = inspect.signature(prop.fget)  # type: ignore[union-attr,attr-defined]

        # Assert
        assert sig.return_annotation is str

    def test_supports_multiband_returns_bool(self) -> None:
        """Test that supports_multiband property returns bool."""
        # Arrange
        # For abstract properties, access the function directly
        prop = AbstractSpectrumAssigner.supports_multiband
        if isinstance(prop, property) and prop.fget is not None:
            sig = inspect.signature(prop.fget)
        else:
            # Fallback for abstractmethod properties
            sig = inspect.signature(prop.fget)  # type: ignore[union-attr,attr-defined]

        # Assert
        assert sig.return_annotation is bool


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestAbstractSpectrumAssignerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_assign_can_return_none(self) -> None:
        """Test assign method can return None when no spectrum available."""

        # Arrange
        class ConcreteSpectrumAssigner(AbstractSpectrumAssigner):
            @property
            def algorithm_name(self) -> str:
                return "test_spectrum"

            @property
            def supports_multiband(self) -> bool:
                return False

            def assign(
                self, path: list[Any], request: Any
            ) -> dict[str, Any] | None:
                return None

            def check_spectrum_availability(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def allocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
                request_id: Any,
            ) -> bool:
                return False

            def deallocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def get_fragmentation_metric(self, path: list[Any]) -> float:
                return 0.0

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteSpectrumAssigner({}, Mock(), Mock())

        # Act
        result = algo.assign([1, 2, 3], Mock())

        # Assert
        assert result is None

    def test_check_spectrum_availability_returns_false_when_unavailable(
        self,
    ) -> None:
        """Test check_spectrum_availability returns False when unavailable."""

        # Arrange
        class ConcreteSpectrumAssigner(AbstractSpectrumAssigner):
            @property
            def algorithm_name(self) -> str:
                return "test_spectrum"

            @property
            def supports_multiband(self) -> bool:
                return False

            def assign(
                self, path: list[Any], request: Any
            ) -> dict[str, Any] | None:
                return None

            def check_spectrum_availability(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def allocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
                request_id: Any,
            ) -> bool:
                return False

            def deallocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def get_fragmentation_metric(self, path: list[Any]) -> float:
                return 0.0

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteSpectrumAssigner({}, Mock(), Mock())

        # Act
        result = algo.check_spectrum_availability([1, 2], 0, 5, 0, "C")

        # Assert
        assert result is False

    def test_fragmentation_metric_returns_value_between_zero_and_one(
        self,
    ) -> None:
        """Test get_fragmentation_metric returns value in valid range."""

        # Arrange
        class ConcreteSpectrumAssigner(AbstractSpectrumAssigner):
            @property
            def algorithm_name(self) -> str:
                return "test_spectrum"

            @property
            def supports_multiband(self) -> bool:
                return False

            def assign(
                self, path: list[Any], request: Any
            ) -> dict[str, Any] | None:
                return None

            def check_spectrum_availability(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def allocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
                request_id: Any,
            ) -> bool:
                return False

            def deallocate_spectrum(
                self,
                path: list[Any],
                start_slot: int,
                end_slot: int,
                core_num: int,
                band: str,
            ) -> bool:
                return False

            def get_fragmentation_metric(self, path: list[Any]) -> float:
                return 0.75

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteSpectrumAssigner({}, Mock(), Mock())

        # Act
        result = algo.get_fragmentation_metric([1, 2, 3])

        # Assert
        assert 0.0 <= result <= 1.0
