"""
Tests for GroomingAdapter.

Phase: P2.4 - Legacy Adapters
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from fusion.core.adapters.grooming_adapter import (
    GroomingAdapter,
    SDNPropsProxyForGrooming,
)
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
        },
        route_method="k_shortest_path",
        allocation_method="first_fit",
        grooming_enabled=True,  # Enable grooming
    )


@pytest.fixture
def config_no_grooming() -> SimulationConfig:
    """Create SimulationConfig with grooming disabled."""
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
        modulation_formats={},
        route_method="k_shortest_path",
        allocation_method="first_fit",
        grooming_enabled=False,  # Grooming disabled
    )


@pytest.fixture
def network_state(simple_topology: nx.Graph, config: SimulationConfig) -> NetworkState:
    """Create NetworkState from topology."""
    return NetworkState(simple_topology, config)


@pytest.fixture
def mock_request() -> MagicMock:
    """Create mock request."""
    request = MagicMock()
    request.request_id = 1
    request.source = "A"
    request.destination = "C"
    request.bandwidth_gbps = 100
    request.arrive_time = 0.0
    return request


class TestSDNPropsProxyForGrooming:
    """Tests for SDNPropsProxyForGrooming."""

    def test_from_network_state(
        self, network_state: NetworkState
    ) -> None:
        """Test creating proxy from NetworkState."""
        proxy = SDNPropsProxyForGrooming.from_network_state(
            network_state=network_state,
            source="A",
            destination="C",
            bandwidth=100.0,
            request_id=1,
            arrive_time=0.0,
        )

        assert proxy.source == "A"
        assert proxy.destination == "C"
        assert proxy.bandwidth == 100.0
        assert proxy.request_id == 1
        assert proxy.arrive == 0.0
        assert proxy.network_spectrum_dict is not None
        assert proxy.lightpath_status_dict is not None

    def test_proxy_lists_initialized(
        self, network_state: NetworkState
    ) -> None:
        """Test that proxy lists are initialized empty."""
        proxy = SDNPropsProxyForGrooming.from_network_state(
            network_state=network_state,
            source="A",
            destination="C",
            bandwidth=100.0,
            request_id=1,
            arrive_time=0.0,
        )

        assert proxy.bandwidth_list == []
        assert proxy.lightpath_id_list == []
        assert proxy.was_groomed is False
        assert proxy.was_partially_groomed is False


class TestGroomingAdapter:
    """Tests for GroomingAdapter."""

    def test_init(self, config: SimulationConfig) -> None:
        """Test adapter initialization."""
        adapter = GroomingAdapter(config)
        assert adapter._config is config
        assert adapter._engine_props is not None

    def test_try_groom_disabled(
        self,
        config_no_grooming: SimulationConfig,
        network_state: NetworkState,
        mock_request: MagicMock,
    ) -> None:
        """Test grooming when disabled in config."""
        # Create network state with no-grooming config
        ns = NetworkState(network_state.topology, config_no_grooming)
        adapter = GroomingAdapter(config_no_grooming)

        result = adapter.try_groom(
            request=mock_request,
            network_state=ns,
        )

        assert result.fully_groomed is False
        assert result.partially_groomed is False
        assert result.lightpaths_used == ()

    def test_try_groom_fully_groomed(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
        mock_request: MagicMock,
    ) -> None:
        """Test successful full grooming."""
        adapter = GroomingAdapter(config)

        with patch(
            "fusion.core.grooming.Grooming"
        ) as MockGrooming:
            mock_grooming = MagicMock()
            mock_grooming.handle_grooming.return_value = True
            MockGrooming.return_value = mock_grooming

            result = adapter.try_groom(
                request=mock_request,
                network_state=network_state,
            )

            # Check that grooming was called
            mock_grooming.handle_grooming.assert_called_once_with("arrival")

    def test_try_groom_not_groomed(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
        mock_request: MagicMock,
    ) -> None:
        """Test when no grooming is possible."""
        adapter = GroomingAdapter(config)

        with patch(
            "fusion.core.grooming.Grooming"
        ) as MockGrooming:
            mock_grooming = MagicMock()
            mock_grooming.handle_grooming.return_value = False
            MockGrooming.return_value = mock_grooming

            result = adapter.try_groom(
                request=mock_request,
                network_state=network_state,
            )

            # No grooming happened
            assert result.fully_groomed is False
            assert result.remaining_bandwidth_gbps == 100  # Original bandwidth

    def test_try_groom_handles_exception(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
        mock_request: MagicMock,
    ) -> None:
        """Test that exceptions are handled gracefully."""
        adapter = GroomingAdapter(config)

        with patch(
            "fusion.core.grooming.Grooming",
            side_effect=Exception("Test error"),
        ):
            result = adapter.try_groom(
                request=mock_request,
                network_state=network_state,
            )

            assert result.fully_groomed is False
            assert result.partially_groomed is False

    def test_rollback_groom(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
        mock_request: MagicMock,
    ) -> None:
        """Test rollback grooming."""
        adapter = GroomingAdapter(config)

        with patch(
            "fusion.core.grooming.Grooming"
        ) as MockGrooming:
            mock_grooming = MagicMock()
            MockGrooming.return_value = mock_grooming

            adapter.rollback_groom(
                request=mock_request,
                lightpath_ids=[1, 2],
                network_state=network_state,
            )

            # Check that grooming release was called
            mock_grooming.handle_grooming.assert_called_once_with("release")

    def test_rollback_groom_handles_exception(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
        mock_request: MagicMock,
    ) -> None:
        """Test that rollback exceptions are handled gracefully."""
        adapter = GroomingAdapter(config)

        with patch(
            "fusion.core.grooming.Grooming",
            side_effect=Exception("Test error"),
        ):
            # Should not raise
            adapter.rollback_groom(
                request=mock_request,
                lightpath_ids=[1, 2],
                network_state=network_state,
            )
