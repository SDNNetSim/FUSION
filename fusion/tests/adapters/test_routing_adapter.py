"""
Tests for RoutingAdapter.

Phase: P2.4 - Legacy Adapters
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from fusion.core.adapters.routing_adapter import RoutingAdapter, SDNPropsProxy
from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState


@pytest.fixture
def simple_topology() -> nx.Graph:
    """Create a simple test topology."""
    G = nx.Graph()
    G.add_edge("A", "B", length=100.0)
    G.add_edge("B", "C", length=150.0)
    G.add_edge("A", "C", length=300.0)
    return G


@pytest.fixture
def config() -> SimulationConfig:
    """Create minimal SimulationConfig for testing."""
    return SimulationConfig(
        network_name="test_network",
        band_list=("c",),
        band_slots={"c": 320},
        cores_per_link=1,
        guard_slots=1,
        num_requests=1000,
        erlang=50.0,
        holding_time=1.0,
        k_paths=3,
        modulation_formats={
            "QPSK": {"max_reach_km": 5000, "bits_per_symbol": 2},
            "16-QAM": {"max_reach_km": 2000, "bits_per_symbol": 4},
        },
        route_method="k_shortest_path",
        allocation_method="first_fit",
    )


@pytest.fixture
def network_state(simple_topology: nx.Graph, config: SimulationConfig) -> NetworkState:
    """Create NetworkState from topology."""
    return NetworkState(simple_topology, config)


class TestSDNPropsProxy:
    """Tests for SDNPropsProxy."""

    def test_from_network_state(
        self, network_state: NetworkState, simple_topology: nx.Graph
    ) -> None:
        """Test creating proxy from NetworkState."""
        proxy = SDNPropsProxy.from_network_state(
            network_state=network_state,
            source="A",
            destination="C",
            bandwidth=100.0,
        )

        assert proxy.source == "A"
        assert proxy.destination == "C"
        assert proxy.bandwidth == 100.0
        assert proxy.topology is network_state.topology
        assert proxy.network_spectrum_dict is not None
        assert proxy.lightpath_status_dict is not None


class TestRoutingAdapter:
    """Tests for RoutingAdapter."""

    def test_init(self, config: SimulationConfig) -> None:
        """Test adapter initialization."""
        adapter = RoutingAdapter(config)
        assert adapter._config is config
        assert adapter._engine_props is not None

    def test_find_routes_empty_network(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test finding routes in empty network."""
        adapter = RoutingAdapter(config)

        # Mock the legacy Routing class to avoid full dependency
        with patch("fusion.core.routing.Routing") as MockRouting:
            mock_routing = MagicMock()
            mock_routing.route_props.paths_matrix = [["A", "B", "C"]]
            mock_routing.route_props.weights_list = [250.0]
            mock_routing.route_props.modulation_formats_matrix = [["QPSK"]]
            MockRouting.return_value = mock_routing

            result = adapter.find_routes(
                source="A",
                destination="C",
                bandwidth_gbps=100,
                network_state=network_state,
            )

            # Verify mock was called
            assert MockRouting.called, "Mock Routing class was not called"
            mock_routing.get_route.assert_called_once()

            assert result.paths is not None
            assert len(result.paths) == 1, f"Expected 1 path, got {len(result.paths)}. Strategy: {result.strategy_name}"
            assert result.paths[0] == ("A", "B", "C")
            assert result.weights_km[0] == 250.0

    def test_find_routes_forced_path(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test finding routes with forced path."""
        adapter = RoutingAdapter(config)

        result = adapter.find_routes(
            source="A",
            destination="C",
            bandwidth_gbps=100,
            network_state=network_state,
            forced_path=["A", "B", "C"],
        )

        # Forced path should return immediately without calling legacy
        assert result.paths is not None
        assert result.paths[0] == ("A", "B", "C")
        assert result.strategy_name == "forced"

    def test_find_routes_handles_exception(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test that exceptions are handled gracefully."""
        adapter = RoutingAdapter(config)

        with patch(
            "fusion.core.routing.Routing",
            side_effect=Exception("Test error"),
        ):
            result = adapter.find_routes(
                source="A",
                destination="C",
                bandwidth_gbps=100,
                network_state=network_state,
            )

            # Should return empty result on error
            assert result.paths == ()

    def test_calculate_path_weight(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test path weight calculation."""
        adapter = RoutingAdapter(config)

        weight = adapter._calculate_path_weight(["A", "B", "C"], network_state)

        # A-B = 100, B-C = 150
        assert weight == 250.0

    def test_get_modulations_for_weight(
        self,
        config: SimulationConfig,
    ) -> None:
        """Test getting valid modulations for weight."""
        adapter = RoutingAdapter(config)

        # Short path - both modulations valid
        mods = adapter._get_modulations_for_weight(1000.0)
        assert "QPSK" in mods
        assert "16-QAM" in mods

        # Medium path - only QPSK valid
        mods = adapter._get_modulations_for_weight(3000.0)
        assert "QPSK" in mods
        assert "16-QAM" not in mods

        # Very long path - no modulations valid
        mods = adapter._get_modulations_for_weight(10000.0)
        assert len(mods) == 0
