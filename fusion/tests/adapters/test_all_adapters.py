"""
Integration tests for all adapters.

Verifies all adapters can be imported and instantiated.

Phase: P2.4 - Legacy Adapters
"""

from __future__ import annotations

import networkx as nx
import pytest

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
    """Create SimulationConfig for testing."""
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


class TestAllAdaptersImport:
    """Test that all adapters can be imported."""

    def test_import_routing_adapter(self) -> None:
        from fusion.core.adapters import RoutingAdapter
        assert RoutingAdapter is not None

    def test_import_spectrum_adapter(self) -> None:
        from fusion.core.adapters import SpectrumAdapter
        assert SpectrumAdapter is not None

    def test_import_grooming_adapter(self) -> None:
        from fusion.core.adapters import GroomingAdapter
        assert GroomingAdapter is not None

    def test_import_snr_adapter(self) -> None:
        from fusion.core.adapters import SNRAdapter
        assert SNRAdapter is not None


class TestAllAdaptersInstantiation:
    """Test that all adapters can be instantiated."""

    def test_instantiate_routing_adapter(self, config: SimulationConfig) -> None:
        from fusion.core.adapters import RoutingAdapter
        adapter = RoutingAdapter(config)
        assert adapter is not None

    def test_instantiate_spectrum_adapter(self, config: SimulationConfig) -> None:
        from fusion.core.adapters import SpectrumAdapter
        adapter = SpectrumAdapter(config)
        assert adapter is not None

    def test_instantiate_grooming_adapter(self, config: SimulationConfig) -> None:
        from fusion.core.adapters import GroomingAdapter
        adapter = GroomingAdapter(config)
        assert adapter is not None

    def test_instantiate_snr_adapter(self, config: SimulationConfig) -> None:
        from fusion.core.adapters import SNRAdapter
        adapter = SNRAdapter(config)
        assert adapter is not None


class TestAllAdaptersProtocol:
    """Test that all adapters implement their protocols."""

    def test_routing_implements_protocol(self, config: SimulationConfig) -> None:
        from fusion.core.adapters import RoutingAdapter
        from fusion.interfaces import RoutingPipeline
        adapter = RoutingAdapter(config)
        assert isinstance(adapter, RoutingPipeline)

    def test_spectrum_implements_protocol(self, config: SimulationConfig) -> None:
        from fusion.core.adapters import SpectrumAdapter
        from fusion.interfaces import SpectrumPipeline
        adapter = SpectrumAdapter(config)
        assert isinstance(adapter, SpectrumPipeline)

    def test_grooming_implements_protocol(self, config: SimulationConfig) -> None:
        from fusion.core.adapters import GroomingAdapter
        from fusion.interfaces import GroomingPipeline
        adapter = GroomingAdapter(config)
        assert isinstance(adapter, GroomingPipeline)

    def test_snr_implements_protocol(self, config: SimulationConfig) -> None:
        from fusion.core.adapters import SNRAdapter
        from fusion.interfaces import SNRPipeline
        adapter = SNRAdapter(config)
        assert isinstance(adapter, SNRPipeline)


class TestAllAdaptersHaveAdapterDocstring:
    """Test that all adapters are marked as temporary migration layers."""

    def test_routing_adapter_marked_as_adapter(self) -> None:
        from fusion.core.adapters import RoutingAdapter
        docstring = RoutingAdapter.__doc__ or ""
        assert "ADAPTER" in docstring

    def test_spectrum_adapter_marked_as_adapter(self) -> None:
        from fusion.core.adapters import SpectrumAdapter
        docstring = SpectrumAdapter.__doc__ or ""
        assert "ADAPTER" in docstring

    def test_grooming_adapter_marked_as_adapter(self) -> None:
        from fusion.core.adapters import GroomingAdapter
        docstring = GroomingAdapter.__doc__ or ""
        assert "ADAPTER" in docstring

    def test_snr_adapter_marked_as_adapter(self) -> None:
        from fusion.core.adapters import SNRAdapter
        docstring = SNRAdapter.__doc__ or ""
        assert "ADAPTER" in docstring
