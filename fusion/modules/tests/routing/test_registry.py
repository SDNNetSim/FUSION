"""Unit tests for fusion.modules.routing.registry module."""

from typing import Any
from unittest.mock import Mock

import networkx as nx
import pytest


@pytest.fixture
def registry() -> Any:
    """Create a fresh RoutingRegistry instance for testing."""
    from fusion.modules.routing.registry import RoutingRegistry

    return RoutingRegistry()


@pytest.fixture
def engine_props() -> dict[str, Any]:
    """Create engine properties for testing."""
    topology = nx.Graph()
    topology.add_edge("A", "B", length=100.0, weight=1)
    topology.add_edge("B", "C", length=150.0, weight=1)

    return {
        "topology": topology,
        "k_paths": 3,
        "beta": 0.5,
        "spectral_slots": 320,
        "guard_slots": 1,
        "mod_per_bw": {"50GHz": {"QPSK": {"max_length": 1000, "slots_needed": 10}}},
    }


@pytest.fixture
def sdn_props() -> Mock:
    """Create mock SDN properties for testing."""
    props = Mock()
    props.source = "A"
    props.destination = "C"
    props.bandwidth = "50GHz"
    props.slots_needed = 10
    return props


class TestRoutingRegistry:
    """Tests for RoutingRegistry class."""

    def test_init_registers_default_algorithms(self, registry: Any) -> None:
        """Test that initialization registers all default algorithms."""
        # Assert
        expected_algorithms = [
            "k_shortest_path",
            "congestion_aware",
            "least_congested",
            "fragmentation_aware",
            "nli_aware",
            "xt_aware",
        ]

        for algorithm_name in expected_algorithms:
            assert algorithm_name in registry._algorithms

    def test_register_valid_algorithm_succeeds(self, registry: Any) -> None:
        """Test registering a valid algorithm class."""
        # Arrange
        from fusion.interfaces.router import AbstractRoutingAlgorithm

        class CustomAlgorithm(AbstractRoutingAlgorithm):
            """Custom algorithm for testing."""

            @property
            def algorithm_name(self) -> str:
                return "custom"

            @property
            def supported_topologies(self) -> list[str]:
                return ["Generic"]

            def validate_environment(self, topology: Any) -> bool:
                return True

            def route(
                self, source: Any, destination: Any, request: Any
            ) -> list[Any] | None:
                return []

            def get_paths(
                self, source: Any, destination: Any, k: int = 1
            ) -> list[list[Any]]:
                return []

            def update_weights(self, topology: Any) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

            def reset(self) -> None:
                pass

        # Act
        registry.register("custom", CustomAlgorithm)

        # Assert
        assert "custom" in registry._algorithms
        assert registry._algorithms["custom"] == CustomAlgorithm

    def test_register_non_abstract_algorithm_raises_type_error(
        self, registry: Any
    ) -> None:
        """Test that registering non-AbstractRoutingAlgorithm class raises TypeError."""

        # Arrange
        class NotAnAlgorithm:
            """Not a routing algorithm."""

            pass

        # Act & Assert
        with pytest.raises(TypeError, match="must implement AbstractRoutingAlgorithm"):
            registry.register("invalid", NotAnAlgorithm)  # type: ignore[arg-type]

    def test_register_duplicate_name_raises_value_error(self, registry: Any) -> None:
        """Test that registering duplicate algorithm name raises ValueError."""
        # Arrange
        from fusion.modules.routing.k_shortest_path import KShortestPath

        # Act & Assert
        with pytest.raises(ValueError, match="already registered"):
            registry.register("k_shortest_path", KShortestPath)

    def test_get_existing_algorithm_returns_class(self, registry: Any) -> None:
        """Test retrieving an existing algorithm class."""
        # Arrange
        from fusion.modules.routing.k_shortest_path import KShortestPath

        # Act
        algorithm_class = registry.get("k_shortest_path")

        # Assert
        assert algorithm_class == KShortestPath

    def test_get_nonexistent_algorithm_raises_key_error(self, registry: Any) -> None:
        """Test that getting non-existent algorithm raises KeyError."""
        # Act & Assert
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_create_algorithm_returns_configured_instance(
        self, registry: Any, engine_props: dict[str, Any], sdn_props: Mock
    ) -> None:
        """Test creating an algorithm instance with configuration."""
        # Arrange
        from fusion.modules.routing.k_shortest_path import KShortestPath

        # Act
        algorithm = registry.create("k_shortest_path", engine_props, sdn_props)

        # Assert
        assert isinstance(algorithm, KShortestPath)
        assert algorithm.engine_props == engine_props
        assert algorithm.sdn_props == sdn_props

    def test_list_algorithms_returns_all_registered_names(self, registry: Any) -> None:
        """Test listing all registered algorithm names."""
        # Act
        algorithms = registry.list_algorithms()

        # Assert
        assert isinstance(algorithms, list)
        assert "k_shortest_path" in algorithms
        assert "congestion_aware" in algorithms
        assert len(algorithms) >= 6

    def test_get_algorithm_info_returns_complete_info(self, registry: Any) -> None:
        """Test retrieving algorithm information."""
        # Act
        info = registry.get_algorithm_info("k_shortest_path")

        # Assert
        assert info["name"] == "k_shortest_path"
        assert info["class"] == "KShortestPath"
        assert "module" in info
        assert "supported_topologies" in info
        assert "description" in info

    def test_validate_algorithm_with_compatible_topology(
        self, registry: Any, engine_props: dict[str, Any]
    ) -> None:
        """Test algorithm validation with compatible topology."""
        # Act
        is_valid = registry.validate_algorithm(
            "k_shortest_path", engine_props["topology"]
        )

        # Assert
        assert is_valid is True

    @pytest.mark.parametrize(
        "algorithm_name,expected_class_name",
        [
            ("k_shortest_path", "KShortestPath"),
            ("congestion_aware", "CongestionAwareRouting"),
            ("least_congested", "LeastCongestedRouting"),
            ("fragmentation_aware", "FragmentationAwareRouting"),
            ("nli_aware", "NLIAwareRouting"),
            ("xt_aware", "XTAwareRouting"),
        ],
    )
    def test_all_default_algorithms_are_registered(
        self, registry: Any, algorithm_name: str, expected_class_name: str
    ) -> None:
        """Test that all default algorithms are properly registered."""
        # Act
        algorithm_class = registry.get(algorithm_name)

        # Assert
        assert algorithm_class.__name__ == expected_class_name


class TestGlobalRegistryFunctions:
    """Tests for global registry convenience functions."""

    def test_get_algorithm_from_global_registry(self) -> None:
        """Test getting algorithm from global registry."""
        # Arrange
        from fusion.modules.routing.k_shortest_path import KShortestPath
        from fusion.modules.routing.registry import get_algorithm

        # Act
        algorithm_class = get_algorithm("k_shortest_path")

        # Assert
        assert algorithm_class == KShortestPath

    def test_create_algorithm_from_global_registry(
        self, engine_props: dict[str, Any], sdn_props: Mock
    ) -> None:
        """Test creating algorithm instance from global registry."""
        # Arrange
        from fusion.modules.routing.congestion_aware import CongestionAwareRouting
        from fusion.modules.routing.registry import create_algorithm

        # Act
        algorithm = create_algorithm("congestion_aware", engine_props, sdn_props)

        # Assert
        assert isinstance(algorithm, CongestionAwareRouting)

    def test_list_routing_algorithms_from_global_registry(self) -> None:
        """Test listing algorithms from global registry."""
        # Arrange
        from fusion.modules.routing.registry import list_routing_algorithms

        # Act
        algorithms = list_routing_algorithms()

        # Assert
        assert isinstance(algorithms, list)
        assert len(algorithms) >= 6
        assert "k_shortest_path" in algorithms

    def test_get_routing_algorithm_info_from_global_registry(self) -> None:
        """Test getting algorithm info from global registry."""
        # Arrange
        from fusion.modules.routing.registry import get_routing_algorithm_info

        # Act
        info = get_routing_algorithm_info("nli_aware")

        # Assert
        assert info["name"] == "nli_aware"
        assert info["class"] == "NLIAwareRouting"
        assert "description" in info

    def test_register_algorithm_in_global_registry(self) -> None:
        """Test registering algorithm in global registry."""
        # Arrange
        from fusion.interfaces.router import AbstractRoutingAlgorithm
        from fusion.modules.routing.registry import get_algorithm, register_algorithm

        class TestAlgorithm(AbstractRoutingAlgorithm):
            """Test algorithm for global registry."""

            @property
            def algorithm_name(self) -> str:
                return "test_global"

            @property
            def supported_topologies(self) -> list[str]:
                return ["Generic"]

            def validate_environment(self, topology: Any) -> bool:
                return True

            def route(
                self, source: Any, destination: Any, request: Any
            ) -> list[Any] | None:
                return []

            def get_paths(
                self, source: Any, destination: Any, k: int = 1
            ) -> list[list[Any]]:
                return []

            def update_weights(self, topology: Any) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

            def reset(self) -> None:
                pass

        # Act
        register_algorithm("test_global_unique", TestAlgorithm)

        # Assert
        algorithm_class = get_algorithm("test_global_unique")
        assert algorithm_class == TestAlgorithm


class TestRoutingAlgorithmsDict:
    """Tests for ROUTING_ALGORITHMS backward compatibility dictionary."""

    def test_routing_algorithms_dict_contains_all_algorithms(self) -> None:
        """Test that ROUTING_ALGORITHMS dict contains all default algorithms."""
        # Arrange
        from fusion.modules.routing.registry import ROUTING_ALGORITHMS

        # Assert
        assert "k_shortest_path" in ROUTING_ALGORITHMS
        assert "congestion_aware" in ROUTING_ALGORITHMS
        assert "least_congested" in ROUTING_ALGORITHMS
        assert "fragmentation_aware" in ROUTING_ALGORITHMS
        assert "nli_aware" in ROUTING_ALGORITHMS
        assert "xt_aware" in ROUTING_ALGORITHMS

    @pytest.mark.parametrize(
        "key,expected_class_name",
        [
            ("k_shortest_path", "KShortestPath"),
            ("congestion_aware", "CongestionAwareRouting"),
            ("least_congested", "LeastCongestedRouting"),
            ("fragmentation_aware", "FragmentationAwareRouting"),
            ("nli_aware", "NLIAwareRouting"),
            ("xt_aware", "XTAwareRouting"),
        ],
    )
    def test_routing_algorithms_dict_maps_correctly(
        self, key: str, expected_class_name: str
    ) -> None:
        """Test that ROUTING_ALGORITHMS dict maps names to correct classes."""
        # Arrange
        from fusion.modules.routing.registry import ROUTING_ALGORITHMS

        # Assert
        assert ROUTING_ALGORITHMS[key].__name__ == expected_class_name
