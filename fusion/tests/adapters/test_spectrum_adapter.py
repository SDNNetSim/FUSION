"""
Tests for SpectrumAdapter.

Phase: P2.4 - Legacy Adapters
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from fusion.core.adapters.spectrum_adapter import (
    SDNPropsProxyForSpectrum,
    SpectrumAdapter,
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
    )


@pytest.fixture
def network_state(simple_topology: nx.Graph, config: SimulationConfig) -> NetworkState:
    """Create NetworkState from topology."""
    return NetworkState(simple_topology, config)


class TestSDNPropsProxyForSpectrum:
    """Tests for SDNPropsProxyForSpectrum."""

    def test_from_network_state(
        self, network_state: NetworkState
    ) -> None:
        """Test creating proxy from NetworkState."""
        proxy = SDNPropsProxyForSpectrum.from_network_state(
            network_state=network_state,
            source="A",
            destination="C",
            bandwidth=100.0,
        )

        assert proxy.source == "A"
        assert proxy.destination == "C"
        assert proxy.bandwidth == 100.0
        assert proxy.network_spectrum_dict is not None


class TestSpectrumAdapter:
    """Tests for SpectrumAdapter."""

    def test_init(self, config: SimulationConfig) -> None:
        """Test adapter initialization."""
        adapter = SpectrumAdapter(config)
        assert adapter._config is config
        assert adapter._engine_props is not None

    def test_find_spectrum_empty_path(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test finding spectrum with empty path."""
        adapter = SpectrumAdapter(config)

        result = adapter.find_spectrum(
            path=[],
            modulation="QPSK",
            bandwidth_gbps=100,
            network_state=network_state,
        )

        assert result.is_free is False

    def test_find_spectrum_single_node_path(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test finding spectrum with single node path."""
        adapter = SpectrumAdapter(config)

        result = adapter.find_spectrum(
            path=["A"],
            modulation="QPSK",
            bandwidth_gbps=100,
            network_state=network_state,
        )

        assert result.is_free is False

    def test_find_spectrum_success(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test successful spectrum finding."""
        adapter = SpectrumAdapter(config)

        # Mock legacy SpectrumAssignment
        with patch(
            "fusion.core.spectrum_assignment.SpectrumAssignment"
        ) as MockSpectrum:
            mock_spectrum = MagicMock()
            mock_spectrum.spectrum_props.is_free = True
            mock_spectrum.spectrum_props.start_slot = 0
            mock_spectrum.spectrum_props.end_slot = 8
            mock_spectrum.spectrum_props.core_number = 0
            mock_spectrum.spectrum_props.current_band = "c"
            mock_spectrum.spectrum_props.modulation = "QPSK"
            mock_spectrum.spectrum_props.slots_needed = 8
            MockSpectrum.return_value = mock_spectrum

            result = adapter.find_spectrum(
                path=["A", "B", "C"],
                modulation="QPSK",
                bandwidth_gbps=100,
                network_state=network_state,
            )

            assert result.is_free is True
            assert result.start_slot == 0
            assert result.end_slot == 8
            assert result.core == 0
            assert result.band == "c"
            assert result.modulation == "QPSK"

    def test_find_spectrum_congestion(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test spectrum finding when no spectrum available."""
        adapter = SpectrumAdapter(config)

        with patch(
            "fusion.core.spectrum_assignment.SpectrumAssignment"
        ) as MockSpectrum:
            mock_spectrum = MagicMock()
            mock_spectrum.spectrum_props.is_free = False
            mock_spectrum.spectrum_props.slots_needed = 8
            MockSpectrum.return_value = mock_spectrum

            result = adapter.find_spectrum(
                path=["A", "B", "C"],
                modulation="QPSK",
                bandwidth_gbps=100,
                network_state=network_state,
            )

            assert result.is_free is False

    def test_find_spectrum_handles_exception(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test that exceptions are handled gracefully."""
        adapter = SpectrumAdapter(config)

        with patch(
            "fusion.core.spectrum_assignment.SpectrumAssignment",
            side_effect=Exception("Test error"),
        ):
            result = adapter.find_spectrum(
                path=["A", "B", "C"],
                modulation="QPSK",
                bandwidth_gbps=100,
                network_state=network_state,
            )

            assert result.is_free is False

    def test_calculate_slots_needed(
        self,
        config: SimulationConfig,
    ) -> None:
        """Test slots calculation."""
        adapter = SpectrumAdapter(config)

        # QPSK with 100 Gbps at 12.5 GHz per slot
        slots = adapter._calculate_slots_needed("QPSK", 100.0)
        assert slots >= 1

    def test_find_protected_spectrum_invalid_paths(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test protected spectrum with invalid paths."""
        adapter = SpectrumAdapter(config)

        # Empty primary path
        result = adapter.find_protected_spectrum(
            primary_path=[],
            backup_path=["A", "C"],
            modulation="QPSK",
            bandwidth_gbps=100,
            network_state=network_state,
        )
        assert result.is_free is False

        # Empty backup path
        result = adapter.find_protected_spectrum(
            primary_path=["A", "B", "C"],
            backup_path=[],
            modulation="QPSK",
            bandwidth_gbps=100,
            network_state=network_state,
        )
        assert result.is_free is False
