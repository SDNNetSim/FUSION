"""
Tests for SNRAdapter.

Phase: P2.4 - Legacy Adapters
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from fusion.core.adapters.snr_adapter import (
    SDNPropsProxyForSNR,
    SNRAdapter,
    SpectrumPropsProxyForSNR,
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
        snr_type="snr_e2e",  # SNR enabled
    )


@pytest.fixture
def config_no_snr() -> SimulationConfig:
    """Create SimulationConfig with SNR disabled."""
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
        snr_type=None,  # SNR disabled
    )


@pytest.fixture
def network_state(simple_topology: nx.Graph, config: SimulationConfig) -> NetworkState:
    """Create NetworkState from topology."""
    return NetworkState(simple_topology, config)


@pytest.fixture
def mock_lightpath() -> MagicMock:
    """Create mock lightpath."""
    lp = MagicMock()
    lp.lightpath_id = 1
    lp.source = "A"
    lp.destination = "C"
    lp.path = ["A", "B", "C"]
    lp.start_slot = 0
    lp.end_slot = 8
    lp.core = 0
    lp.band = "c"
    lp.modulation = "QPSK"
    lp.num_slots = 8
    lp.total_bandwidth_gbps = 100
    lp.path_weight_km = 250.0
    lp.is_degraded = False
    return lp


class TestSDNPropsProxyForSNR:
    """Tests for SDNPropsProxyForSNR."""

    def test_from_network_state(
        self, network_state: NetworkState
    ) -> None:
        """Test creating proxy from NetworkState."""
        proxy = SDNPropsProxyForSNR.from_network_state(
            network_state=network_state,
            source="A",
            destination="C",
            bandwidth=100.0,
            path_index=0,
        )

        assert proxy.source == "A"
        assert proxy.destination == "C"
        assert proxy.bandwidth == 100.0
        assert proxy.path_index == 0
        assert proxy.network_spectrum_dict is not None


class TestSpectrumPropsProxyForSNR:
    """Tests for SpectrumPropsProxyForSNR."""

    def test_creation(self) -> None:
        """Test proxy creation with all fields."""
        proxy = SpectrumPropsProxyForSNR(
            path_list=["A", "B", "C"],
            start_slot=0,
            end_slot=8,
            core_number=0,
            current_band="c",
            modulation="QPSK",
            slots_needed=8,
            is_free=True,
        )

        assert proxy.path_list == ["A", "B", "C"]
        assert proxy.start_slot == 0
        assert proxy.end_slot == 8
        assert proxy.core_number == 0
        assert proxy.current_band == "c"


class TestSNRAdapter:
    """Tests for SNRAdapter."""

    def test_init(self, config: SimulationConfig) -> None:
        """Test adapter initialization."""
        adapter = SNRAdapter(config)
        assert adapter._config is config
        assert adapter._engine_props is not None

    def test_validate_disabled(
        self,
        config_no_snr: SimulationConfig,
        network_state: NetworkState,
        mock_lightpath: MagicMock,
    ) -> None:
        """Test SNR validation when disabled."""
        # Create network state with no-snr config
        ns = NetworkState(network_state.topology, config_no_snr)
        adapter = SNRAdapter(config_no_snr)

        result = adapter.validate(
            lightpath=mock_lightpath,
            network_state=ns,
        )

        # Should return skipped result (passed=True)
        assert result.passed is True

    def test_validate_acceptable(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
        mock_lightpath: MagicMock,
    ) -> None:
        """Test successful SNR validation."""
        adapter = SNRAdapter(config)

        with patch(
            "fusion.core.snr_measurements.SnrMeasurements"
        ) as MockSNR:
            mock_snr = MagicMock()
            mock_snr.handle_snr.return_value = (True, 0.5, 100.0)  # acceptable
            MockSNR.return_value = mock_snr

            result = adapter.validate(
                lightpath=mock_lightpath,
                network_state=network_state,
            )

            assert result.passed is True

    def test_validate_not_acceptable(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
        mock_lightpath: MagicMock,
    ) -> None:
        """Test SNR validation failure."""
        adapter = SNRAdapter(config)

        with patch(
            "fusion.core.snr_measurements.SnrMeasurements"
        ) as MockSNR:
            mock_snr = MagicMock()
            mock_snr.handle_snr.return_value = (False, 0.8, 100.0)  # not acceptable
            MockSNR.return_value = mock_snr

            result = adapter.validate(
                lightpath=mock_lightpath,
                network_state=network_state,
            )

            assert result.passed is False

    def test_validate_handles_exception(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
        mock_lightpath: MagicMock,
    ) -> None:
        """Test that exceptions are handled gracefully."""
        adapter = SNRAdapter(config)

        with patch(
            "fusion.core.snr_measurements.SnrMeasurements",
            side_effect=Exception("Test error"),
        ):
            result = adapter.validate(
                lightpath=mock_lightpath,
                network_state=network_state,
            )

            # Should fail open (skip check)
            assert result.passed is True

    def test_recheck_affected_disabled(
        self,
        config_no_snr: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test SNR recheck when disabled."""
        ns = NetworkState(network_state.topology, config_no_snr)
        adapter = SNRAdapter(config_no_snr)

        result = adapter.recheck_affected(
            new_lightpath_id=1,
            network_state=ns,
        )

        assert result.all_pass is True
        assert result.checked_count == 0

    def test_recheck_affected_lightpath_not_found(
        self,
        config: SimulationConfig,
        network_state: NetworkState,
    ) -> None:
        """Test SNR recheck when new lightpath doesn't exist."""
        adapter = SNRAdapter(config)

        result = adapter.recheck_affected(
            new_lightpath_id=999,  # Non-existent
            network_state=network_state,
        )

        assert result.all_pass is True
        assert result.checked_count == 0
