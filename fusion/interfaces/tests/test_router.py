"""
Unit tests for fusion.interfaces.router module.

Tests the AbstractRoutingAlgorithm abstract base class for routing algorithms.
"""

import inspect
from typing import Any
from unittest.mock import Mock

import pytest

from fusion.interfaces.router import AbstractRoutingAlgorithm

# ============================================================================
# Test Abstract Interface Instantiation
# ============================================================================


class TestAbstractRoutingAlgorithmInstantiation:
    """Tests that AbstractRoutingAlgorithm cannot be directly instantiated."""

    def test_abstract_routing_algorithm_cannot_be_instantiated(self) -> None:
        """Test AbstractRoutingAlgorithm cannot be directly instantiated."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError):
            # type: ignore[abstract,arg-type]
            AbstractRoutingAlgorithm({}, None)


# ============================================================================
# Test Abstract Methods
# ============================================================================


class TestAbstractRoutingAlgorithmAbstractMethods:
    """Tests that required methods are marked as abstract."""

    def test_routing_algorithm_has_correct_abstract_methods(self) -> None:
        """Test AbstractRoutingAlgorithm has correct abstract methods."""
        # Arrange
        expected_methods = {
            "algorithm_name",
            "supported_topologies",
            "validate_environment",
            "route",
            "get_paths",
            "update_weights",
            "get_metrics",
        }

        # Act
        abstract_methods = {
            method
            for method in dir(AbstractRoutingAlgorithm)
            if hasattr(
                getattr(AbstractRoutingAlgorithm, method), "__isabstractmethod__"
            )
            and getattr(
                AbstractRoutingAlgorithm, method
            ).__isabstractmethod__
        }

        # Assert
        assert abstract_methods == expected_methods


# ============================================================================
# Test Interface Method Signatures
# ============================================================================


class TestAbstractRoutingAlgorithmMethodSignatures:
    """Tests that AbstractRoutingAlgorithm method signatures are consistent."""

    def test_route_method_signature(self) -> None:
        """Test AbstractRoutingAlgorithm.route method signature."""
        # Arrange & Act
        route_sig = inspect.signature(AbstractRoutingAlgorithm.route)
        params = list(route_sig.parameters.keys())

        # Assert
        assert params == ["self", "source", "destination", "request"]
        annotation_str = str(route_sig.return_annotation)
        assert (
            "None" in annotation_str or "Optional" in annotation_str
        )

    def test_get_paths_method_signature(self) -> None:
        """Test AbstractRoutingAlgorithm.get_paths method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractRoutingAlgorithm.get_paths)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "source", "destination", "k"]
        assert sig.parameters["k"].default == 1

    def test_validate_environment_method_signature(self) -> None:
        """Test AbstractRoutingAlgorithm.validate_environment signature."""
        # Arrange & Act
        sig = inspect.signature(
            AbstractRoutingAlgorithm.validate_environment
        )
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "topology"]
        assert sig.return_annotation is bool

    def test_update_weights_method_signature(self) -> None:
        """Test AbstractRoutingAlgorithm.update_weights method signature."""
        # Arrange & Act
        sig = inspect.signature(AbstractRoutingAlgorithm.update_weights)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "topology"]


# ============================================================================
# Test Required Methods
# ============================================================================


class TestAbstractRoutingAlgorithmRequiredMethods:
    """Tests that AbstractRoutingAlgorithm has all required methods."""

    def test_routing_algorithm_has_all_required_methods(self) -> None:
        """Test AbstractRoutingAlgorithm has all required methods."""
        # Arrange
        expected_methods = [
            "algorithm_name",
            "supported_topologies",
            "validate_environment",
            "route",
            "get_paths",
            "update_weights",
            "get_metrics",
            "reset",
        ]

        # Act & Assert
        for method in expected_methods:
            assert hasattr(AbstractRoutingAlgorithm, method)


# ============================================================================
# Test Initialization
# ============================================================================


class TestAbstractRoutingAlgorithmInitialization:
    """Tests for AbstractRoutingAlgorithm initialization."""

    def test_initialization_stores_engine_props(self) -> None:
        """Test that initialization stores engine_props correctly."""

        # Arrange
        class ConcreteRoutingAlgorithm(AbstractRoutingAlgorithm):
            @property
            def algorithm_name(self) -> str:
                return "test_routing"

            @property
            def supported_topologies(self) -> list[str]:
                return ["NSFNet"]

            def validate_environment(self, topology: Any) -> bool:
                return True

            def route(
                self, source: Any, destination: Any, request: Any
            ) -> list[Any] | None:
                return None

            def get_paths(
                self, source: Any, destination: Any, k: int = 1
            ) -> list[list[Any]]:
                return []

            def update_weights(self, topology: Any) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        engine_props = {"topology": "test_topology"}
        sdn_props = Mock()

        # Act
        algo = ConcreteRoutingAlgorithm(engine_props, sdn_props)

        # Assert
        assert algo.engine_props == engine_props
        assert algo.sdn_props == sdn_props


# ============================================================================
# Test Concrete Implementation
# ============================================================================


class TestConcreteRoutingAlgorithmImplementation:
    """Tests for concrete implementation of AbstractRoutingAlgorithm."""

    def test_concrete_routing_algorithm_can_be_instantiated(self) -> None:
        """Test concrete implementation with all methods can be instantiated."""

        # Arrange
        class ConcreteRoutingAlgorithm(AbstractRoutingAlgorithm):
            @property
            def algorithm_name(self) -> str:
                return "test_routing"

            @property
            def supported_topologies(self) -> list[str]:
                return ["NSFNet", "USBackbone"]

            def validate_environment(self, topology: Any) -> bool:
                return True

            def route(
                self, source: Any, destination: Any, request: Any
            ) -> list[Any] | None:
                return [source, destination]

            def get_paths(
                self, source: Any, destination: Any, k: int = 1
            ) -> list[list[Any]]:
                return [[source, destination]]

            def update_weights(self, topology: Any) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {"paths_computed": 10}

        engine_props: dict[str, Any] = {}
        sdn_props = Mock()

        # Act
        algo = ConcreteRoutingAlgorithm(engine_props, sdn_props)

        # Assert
        assert algo.algorithm_name == "test_routing"
        assert algo.supported_topologies == ["NSFNet", "USBackbone"]
        assert algo.validate_environment(Mock()) is True
        assert algo.get_metrics() == {"paths_computed": 10}

    def test_concrete_routing_algorithm_missing_method_cannot_be_instantiated(
        self,
    ) -> None:
        """Test concrete implementation missing abstract methods."""

        # Arrange
        class IncompleteRoutingAlgorithm(AbstractRoutingAlgorithm):
            @property
            def algorithm_name(self) -> str:
                return "incomplete"

            # Missing other required abstract methods

        # Act & Assert
        with pytest.raises(TypeError):
            IncompleteRoutingAlgorithm({}, Mock())  # type: ignore[abstract]


# ============================================================================
# Test Reset Method
# ============================================================================


class TestAbstractRoutingAlgorithmReset:
    """Tests for the reset method."""

    def test_reset_method_has_default_implementation(self) -> None:
        """Test that reset method has default implementation."""

        # Arrange
        class ConcreteRoutingAlgorithm(AbstractRoutingAlgorithm):
            @property
            def algorithm_name(self) -> str:
                return "test_routing"

            @property
            def supported_topologies(self) -> list[str]:
                return ["NSFNet"]

            def validate_environment(self, topology: Any) -> bool:
                return True

            def route(
                self, source: Any, destination: Any, request: Any
            ) -> list[Any] | None:
                return None

            def get_paths(
                self, source: Any, destination: Any, k: int = 1
            ) -> list[list[Any]]:
                return []

            def update_weights(self, topology: Any) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        # Act
        algo = ConcreteRoutingAlgorithm({}, Mock())
        algo.reset()

        # Assert - reset doesn't return a value
        assert True


# ============================================================================
# Test Property Return Types
# ============================================================================


class TestAbstractRoutingAlgorithmPropertyReturnTypes:
    """Tests for property return types."""

    def test_algorithm_name_returns_string(self) -> None:
        """Test that algorithm_name property returns string."""
        # Arrange
        sig = inspect.signature(
            # type: ignore[attr-defined]
            AbstractRoutingAlgorithm.algorithm_name.fget
        )

        # Assert
        assert sig.return_annotation is str

    def test_supported_topologies_returns_list_of_strings(self) -> None:
        """Test supported_topologies property returns list of strings."""
        # Arrange
        sig = inspect.signature(
            # type: ignore[attr-defined]
            AbstractRoutingAlgorithm.supported_topologies.fget
        )
        annotation_str = str(sig.return_annotation)

        # Assert
        assert "list" in annotation_str


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestAbstractRoutingAlgorithmEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_route_can_return_none(self) -> None:
        """Test route method can return None for no path found."""

        # Arrange
        class ConcreteRoutingAlgorithm(AbstractRoutingAlgorithm):
            @property
            def algorithm_name(self) -> str:
                return "test_routing"

            @property
            def supported_topologies(self) -> list[str]:
                return ["NSFNet"]

            def validate_environment(self, topology: Any) -> bool:
                return True

            def route(
                self, source: Any, destination: Any, request: Any
            ) -> list[Any] | None:
                return None

            def get_paths(
                self, source: Any, destination: Any, k: int = 1
            ) -> list[list[Any]]:
                return []

            def update_weights(self, topology: Any) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteRoutingAlgorithm({}, Mock())

        # Act
        result = algo.route(1, 2, Mock())

        # Assert
        assert result is None

    def test_get_paths_can_return_empty_list(self) -> None:
        """Test get_paths method can return empty list when no paths exist."""

        # Arrange
        class ConcreteRoutingAlgorithm(AbstractRoutingAlgorithm):
            @property
            def algorithm_name(self) -> str:
                return "test_routing"

            @property
            def supported_topologies(self) -> list[str]:
                return ["NSFNet"]

            def validate_environment(self, topology: Any) -> bool:
                return True

            def route(
                self, source: Any, destination: Any, request: Any
            ) -> list[Any] | None:
                return None

            def get_paths(
                self, source: Any, destination: Any, k: int = 1
            ) -> list[list[Any]]:
                return []

            def update_weights(self, topology: Any) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteRoutingAlgorithm({}, Mock())

        # Act
        result = algo.get_paths(1, 2, k=3)

        # Assert
        assert result == []

    def test_validate_environment_can_return_false(self) -> None:
        """Test validate_environment returns False for incompatible topology."""

        # Arrange
        class ConcreteRoutingAlgorithm(AbstractRoutingAlgorithm):
            @property
            def algorithm_name(self) -> str:
                return "test_routing"

            @property
            def supported_topologies(self) -> list[str]:
                return ["NSFNet"]

            def validate_environment(self, topology: Any) -> bool:
                return False

            def route(
                self, source: Any, destination: Any, request: Any
            ) -> list[Any] | None:
                return None

            def get_paths(
                self, source: Any, destination: Any, k: int = 1
            ) -> list[list[Any]]:
                return []

            def update_weights(self, topology: Any) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        algo = ConcreteRoutingAlgorithm({}, Mock())

        # Act
        result = algo.validate_environment(Mock())

        # Assert
        assert result is False
