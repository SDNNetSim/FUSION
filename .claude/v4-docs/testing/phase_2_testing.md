# Phase 2 Testing Strategy

## Overview

Phase 2 introduces `NetworkState` as the authoritative data source and pipeline interfaces. Testing must verify:

1. NetworkState correctly manages spectrum and lightpaths
2. Legacy compatibility properties return expected formats
3. State transitions maintain consistency
4. Pipeline adapters produce same results as legacy code

## Test File Structure

```
fusion/tests/
    domain/
        test_network_state.py       # NetworkState unit tests
        test_link_spectrum.py       # LinkSpectrum unit tests
    interfaces/
        test_pipeline_protocols.py  # Protocol compliance tests
    adapters/
        test_routing_adapter.py     # Routing adapter tests
        test_spectrum_adapter.py    # Spectrum adapter tests
        test_grooming_adapter.py    # Grooming adapter tests
        test_snr_adapter.py         # SNR adapter tests
    integration/
        test_state_consistency.py   # End-to-end state tests
        test_legacy_parity.py       # Legacy format comparison
```

## NetworkState Unit Tests

### test_network_state.py

```python
import pytest
import numpy as np
import networkx as nx

from fusion.domain.network_state import NetworkState, LinkSpectrum
from fusion.domain.config import SimulationConfig
from fusion.domain.lightpath import Lightpath


class TestNetworkStateInitialization:
    """Tests for NetworkState initialization."""

    def test_init_creates_spectrum_for_all_links(self, simple_topology, simple_config):
        """NetworkState should create spectrum for every link (both directions)."""
        state = NetworkState(simple_topology, simple_config)

        for u, v in simple_topology.edges():
            assert (u, v) in state._spectrum
            assert (v, u) in state._spectrum

    def test_init_spectrum_has_correct_shape(self, simple_config):
        """Spectrum matrices should match config dimensions."""
        topology = nx.path_graph(3, create_using=nx.Graph())
        nx.relabel_nodes(topology, {0: "A", 1: "B", 2: "C"}, copy=False)

        state = NetworkState(topology, simple_config)

        for link_spectrum in state._spectrum.values():
            for band in simple_config.band_list:
                matrix = link_spectrum.cores_matrix[band]
                assert matrix.shape == (simple_config.cores_per_link, simple_config.band_slots[band])

    def test_init_spectrum_is_empty(self, simple_topology, simple_config):
        """All spectrum should be free initially."""
        state = NetworkState(simple_topology, simple_config)

        for link_spectrum in state._spectrum.values():
            for band, matrix in link_spectrum.cores_matrix.items():
                assert np.all(matrix == 0)

    def test_init_no_lightpaths(self, simple_topology, simple_config):
        """No lightpaths should exist initially."""
        state = NetworkState(simple_topology, simple_config)

        assert len(state._lightpaths) == 0
        assert state._next_lightpath_id == 1


class TestSpectrumAvailability:
    """Tests for spectrum availability checking."""

    def test_is_spectrum_available_empty_network(self, simple_state):
        """All spectrum should be available in empty network."""
        assert simple_state.is_spectrum_available(["A", "B", "C"], 0, 10, 0, "c")

    def test_is_spectrum_available_after_allocation(self, simple_state):
        """Allocated spectrum should not be available."""
        simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        assert not simple_state.is_spectrum_available(["A", "B", "C"], 0, 10, 0, "c")
        assert not simple_state.is_spectrum_available(["A", "B"], 0, 10, 0, "c")
        assert not simple_state.is_spectrum_available(["B", "C"], 5, 15, 0, "c")

    def test_is_spectrum_available_different_core(self, simple_state):
        """Different cores should be independent."""
        simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        # Same slots, different core - should be available
        assert simple_state.is_spectrum_available(["A", "B", "C"], 0, 10, 1, "c")

    def test_is_spectrum_available_different_band(self, simple_state_multiband):
        """Different bands should be independent."""
        simple_state_multiband.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        # Same slots, different band - should be available
        assert simple_state_multiband.is_spectrum_available(["A", "B", "C"], 0, 10, 0, "l")


class TestLightpathCreation:
    """Tests for lightpath creation."""

    def test_create_lightpath_returns_lightpath(self, simple_state):
        """create_lightpath should return a Lightpath object."""
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        assert isinstance(lp, Lightpath)
        assert lp.path == ["A", "B", "C"]
        assert lp.start_slot == 0
        assert lp.end_slot == 10

    def test_create_lightpath_allocates_spectrum(self, simple_state):
        """create_lightpath should allocate spectrum on all path links."""
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        # Check both directions
        for link in [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]:
            spectrum = simple_state._spectrum[link].cores_matrix["c"][0]
            assert np.all(spectrum[0:10] == lp.lightpath_id)

    def test_create_lightpath_increments_id(self, simple_state):
        """Each lightpath should get a unique incrementing ID."""
        lp1 = simple_state.create_lightpath(
            path=["A", "B"], start_slot=0, end_slot=5, core=0, band="c",
            modulation="QPSK", bandwidth_gbps=50, path_weight_km=50.0,
        )
        lp2 = simple_state.create_lightpath(
            path=["B", "C"], start_slot=0, end_slot=5, core=0, band="c",
            modulation="QPSK", bandwidth_gbps=50, path_weight_km=50.0,
        )

        assert lp1.lightpath_id == 1
        assert lp2.lightpath_id == 2

    def test_create_lightpath_stores_in_dict(self, simple_state):
        """Lightpath should be retrievable after creation."""
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        assert simple_state.get_lightpath(lp.lightpath_id) is lp


class TestLightpathRelease:
    """Tests for lightpath release."""

    def test_release_lightpath_frees_spectrum(self, simple_state):
        """release_lightpath should free all allocated spectrum."""
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        simple_state.release_lightpath(lp.lightpath_id)

        # Spectrum should be free again
        assert simple_state.is_spectrum_available(["A", "B", "C"], 0, 10, 0, "c")

    def test_release_lightpath_removes_from_dict(self, simple_state):
        """Lightpath should not be retrievable after release."""
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        simple_state.release_lightpath(lp.lightpath_id)

        assert simple_state.get_lightpath(lp.lightpath_id) is None

    def test_release_nonexistent_lightpath_no_error(self, simple_state):
        """Releasing non-existent lightpath should not raise."""
        simple_state.release_lightpath(999)  # Should not raise


class TestLightpathQueries:
    """Tests for lightpath query methods."""

    def test_get_lightpaths_between(self, simple_state):
        """get_lightpaths_between should find lightpaths by endpoints."""
        lp1 = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )
        lp2 = simple_state.create_lightpath(
            path=["A", "D", "C"],
            start_slot=10,
            end_slot=20,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=150.0,
        )

        # Both should be found (same endpoints, different paths)
        lps = simple_state.get_lightpaths_between("A", "C")
        assert len(lps) == 2
        assert lp1 in lps
        assert lp2 in lps

        # Order of endpoints should not matter
        lps_reversed = simple_state.get_lightpaths_between("C", "A")
        assert len(lps_reversed) == 2

    def test_get_lightpaths_with_capacity(self, simple_state):
        """get_lightpaths_with_capacity should filter by bandwidth."""
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )
        # Simulate partial usage
        lp.remaining_bandwidth_gbps = 30

        lps_50 = simple_state.get_lightpaths_with_capacity("A", "C", min_bw=50)
        assert len(lps_50) == 0

        lps_20 = simple_state.get_lightpaths_with_capacity("A", "C", min_bw=20)
        assert len(lps_20) == 1
```

## Snapshot Tests

Snapshot tests verify that state changes are correctly tracked:

```python
class TestStateSnapshots:
    """Tests verifying state consistency across operations."""

    def test_create_release_restores_state(self, simple_state):
        """Create then release should restore initial state."""
        # Capture initial state
        initial_spectrum = self._capture_spectrum_snapshot(simple_state)

        # Create and release
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )
        simple_state.release_lightpath(lp.lightpath_id)

        # State should match initial
        final_spectrum = self._capture_spectrum_snapshot(simple_state)
        self._assert_spectrum_equal(initial_spectrum, final_spectrum)

    def test_multiple_create_release_restores_state(self, simple_state):
        """Multiple create/release cycles should restore initial state."""
        initial_spectrum = self._capture_spectrum_snapshot(simple_state)

        for i in range(10):
            lp = simple_state.create_lightpath(
                path=["A", "B", "C"],
                start_slot=i * 10,
                end_slot=(i + 1) * 10,
                core=0,
                band="c",
                modulation="QPSK",
                bandwidth_gbps=100,
                path_weight_km=100.0,
            )
            simple_state.release_lightpath(lp.lightpath_id)

        final_spectrum = self._capture_spectrum_snapshot(simple_state)
        self._assert_spectrum_equal(initial_spectrum, final_spectrum)

    def _capture_spectrum_snapshot(self, state):
        """Capture deep copy of spectrum state."""
        return {
            link: {
                band: matrix.copy()
                for band, matrix in ls.cores_matrix.items()
            }
            for link, ls in state._spectrum.items()
        }

    def _assert_spectrum_equal(self, snap1, snap2):
        """Assert two spectrum snapshots are equal."""
        assert snap1.keys() == snap2.keys()
        for link in snap1:
            for band in snap1[link]:
                np.testing.assert_array_equal(snap1[link][band], snap2[link][band])
```

## Legacy Parity Tests

These tests verify legacy properties return correct formats:

```python
class TestLegacyNetworkSpectrumDict:
    """Tests for network_spectrum_dict legacy property."""

    def test_format_matches_legacy(self, simple_state, legacy_sdn_props):
        """Legacy property should match expected format."""
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        legacy_dict = simple_state.network_spectrum_dict

        # Check structure
        assert ("A", "B") in legacy_dict
        assert "cores_matrix" in legacy_dict[("A", "B")]
        assert "c" in legacy_dict[("A", "B")]["cores_matrix"]

        # Check values
        spectrum = legacy_dict[("A", "B")]["cores_matrix"]["c"][0]
        assert np.all(spectrum[0:10] == lp.lightpath_id)

    def test_returns_copy_not_reference(self, simple_state):
        """Legacy property should return copy, not reference."""
        dict1 = simple_state.network_spectrum_dict
        dict2 = simple_state.network_spectrum_dict

        assert dict1 is not dict2


class TestLegacyLightpathStatusDict:
    """Tests for lightpath_status_dict legacy property."""

    def test_format_matches_legacy(self, simple_state):
        """Legacy property should match expected format."""
        lp = simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        legacy_dict = simple_state.lightpath_status_dict

        # Key should be sorted tuple
        key = ("A", "C")  # sorted
        assert key in legacy_dict

        # Lightpath ID should be nested key
        assert lp.lightpath_id in legacy_dict[key]

        # Check all expected fields
        lp_info = legacy_dict[key][lp.lightpath_id]
        assert lp_info["path"] == ["A", "B", "C"]
        assert lp_info["core"] == 0
        assert lp_info["band"] == "c"
        assert lp_info["start_slot"] == 0
        assert lp_info["end_slot"] == 10
        assert lp_info["mod_format"] == "QPSK"
        assert lp_info["lightpath_bandwidth"] == 100
        assert lp_info["remaining_bandwidth"] == 100
        assert "requests_dict" in lp_info

    def test_key_is_sorted(self, simple_state):
        """Endpoint key should be sorted regardless of path direction."""
        # Create lightpath C -> A (reversed)
        lp = simple_state.create_lightpath(
            path=["C", "B", "A"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        legacy_dict = simple_state.lightpath_status_dict

        # Key should still be sorted as (A, C)
        assert ("A", "C") in legacy_dict
        assert ("C", "A") not in legacy_dict
```

## Adapter Tests

```python
class TestRoutingAdapter:
    """Tests for RoutingAdapter."""

    def test_implements_protocol(self):
        """RoutingAdapter should implement RoutingPipeline protocol."""
        from fusion.interfaces.pipelines import RoutingPipeline

        adapter = RoutingAdapter(simple_config)
        assert isinstance(adapter, RoutingPipeline)

    def test_find_routes_returns_route_result(self, simple_state, simple_config):
        """find_routes should return RouteResult."""
        adapter = RoutingAdapter(simple_config)

        result = adapter.find_routes("A", "C", 100, simple_state)

        assert isinstance(result, RouteResult)
        assert len(result.paths) > 0

    def test_find_routes_matches_legacy(self, simple_state, simple_config, legacy_routing):
        """Adapter should return same routes as legacy code."""
        adapter = RoutingAdapter(simple_config)

        # Get results from adapter
        adapter_result = adapter.find_routes("A", "C", 100, simple_state)

        # Get results from legacy
        legacy_result = legacy_routing.get_route("A", "C")

        # Compare
        assert adapter_result.paths == legacy_result.paths_matrix
        assert adapter_result.weights_km == legacy_result.weights_list
```

## Validation Tests

```python
class TestStateValidation:
    """Tests for state validation and invariants."""

    def test_spectrum_cannot_be_double_allocated(self, simple_state):
        """Allocating already-used spectrum should fail."""
        simple_state.create_lightpath(
            path=["A", "B", "C"],
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        with pytest.raises(ValueError, match="Spectrum not available"):
            simple_state.create_lightpath(
                path=["A", "B", "C"],
                start_slot=5,
                end_slot=15,  # Overlaps with 0-10
                core=0,
                band="c",
                modulation="QPSK",
                bandwidth_gbps=100,
                path_weight_km=100.0,
            )

    def test_lightpath_id_uniqueness(self, simple_state):
        """Lightpath IDs should be unique."""
        ids = set()
        for i in range(100):
            lp = simple_state.create_lightpath(
                path=["A", "B"],
                start_slot=i,
                end_slot=i + 1,
                core=0,
                band="c",
                modulation="QPSK",
                bandwidth_gbps=10,
                path_weight_km=50.0,
            )
            assert lp.lightpath_id not in ids
            ids.add(lp.lightpath_id)
```

## Test Fixtures

```python
# conftest.py

import pytest
import networkx as nx

from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState


@pytest.fixture
def simple_config():
    """Simple single-band configuration."""
    return SimulationConfig(
        network_name="test",
        cores_per_link=7,
        band_list=("c",),
        band_slots={"c": 320},
        guard_slots=1,
        num_requests=100,
        erlang=100,
        holding_time=1,
        route_method="k_shortest_path",
        k_paths=3,
        allocation_method="first_fit",
        grooming_enabled=False,
        slicing_enabled=False,
        max_slices=1,
        snr_enabled=False,
        snr_type=None,
        snr_recheck=False,
        can_partially_serve=False,
        modulation_formats={},
        mod_per_bw={},
        snr_thresholds={},
    )


@pytest.fixture
def simple_topology():
    """Simple 4-node topology: A-B-C with A-D-C alternate path."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=50)
    G.add_edge("B", "C", weight=50)
    G.add_edge("A", "D", weight=75)
    G.add_edge("D", "C", weight=75)
    return G


@pytest.fixture
def simple_state(simple_topology, simple_config):
    """NetworkState with simple topology."""
    return NetworkState(simple_topology, simple_config)


@pytest.fixture
def simple_config_multiband():
    """Multi-band configuration."""
    return SimulationConfig(
        network_name="test",
        cores_per_link=7,
        band_list=("c", "l"),
        band_slots={"c": 320, "l": 320},
        # ... other fields
    )


@pytest.fixture
def simple_state_multiband(simple_topology, simple_config_multiband):
    """NetworkState with multi-band config."""
    return NetworkState(simple_topology, simple_config_multiband)
```

## Running Tests

```bash
# Run all Phase 2 tests
pytest fusion/tests/domain/test_network_state.py -v
pytest fusion/tests/domain/test_link_spectrum.py -v
pytest fusion/tests/adapters/ -v

# Run with coverage
pytest fusion/tests/domain/ fusion/tests/adapters/ --cov=fusion/domain --cov=fusion/core/adapters

# Run legacy parity tests
pytest fusion/tests/integration/test_legacy_parity.py -v

# Type checking
mypy fusion/domain/ fusion/interfaces/ fusion/core/adapters/
```

## Coverage Targets

| Module | Target Coverage |
|--------|-----------------|
| `fusion/domain/network_state.py` | 90% |
| `fusion/domain/link_spectrum.py` | 95% |
| `fusion/interfaces/pipelines.py` | 100% (just types) |
| `fusion/core/adapters/*.py` | 80% |
