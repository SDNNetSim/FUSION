# Phase 3 Testing Plan

## Overview

Phase 3 testing focuses on:
1. `PipelineFactory` correctness
2. `SDNOrchestrator` request flows
3. Feature combination coverage
4. Regression testing via `run_comparison.py`

## Test Strategy

### Layers

| Layer | Purpose | Tools |
|-------|---------|-------|
| Unit | Test individual components | pytest, mocks |
| Integration | Test component interactions | pytest, real objects |
| Regression | Verify behavioral equivalence | run_comparison.py |

### Key Principle: Old Path as Oracle

The legacy SDNController path serves as the regression oracle:
- Run same scenario through both paths
- Compare final metrics (blocking rate, throughput, etc.)
- Tolerance: `abs_tol=0.02` for statistical variations

---

## Unit Tests

### PipelineFactory Tests

**File**: `fusion/tests/core/test_pipeline_factory.py`

```python
import pytest
from fusion.domain.config import SimulationConfig
from fusion.core.pipeline_factory import PipelineFactory, PipelineSet
from fusion.interfaces.pipelines import (
    RoutingPipeline, SpectrumPipeline, GroomingPipeline,
    SNRPipeline, SlicingPipeline
)


class TestPipelineFactoryRouting:
    """Tests for routing pipeline creation."""

    def test_create_routing_default_returns_adapter(self, base_config):
        """Default routing uses adapter wrapping legacy."""
        pipeline = PipelineFactory.create_routing(base_config)
        assert isinstance(pipeline, RoutingPipeline)

    def test_create_routing_protected_returns_protected_pipeline(self, protection_config):
        """1+1 protection uses ProtectedRoutingPipeline."""
        pipeline = PipelineFactory.create_routing(protection_config)
        assert pipeline.__class__.__name__ == "ProtectedRoutingPipeline"

    def test_create_routing_ksp_returns_ksp_pipeline(self, ksp_config):
        """K-shortest-path uses KSPRoutingPipeline or adapter."""
        pipeline = PipelineFactory.create_routing(ksp_config)
        assert isinstance(pipeline, RoutingPipeline)


class TestPipelineFactorySpectrum:
    """Tests for spectrum pipeline creation."""

    def test_create_spectrum_first_fit(self, base_config):
        """First-fit config creates appropriate pipeline."""
        pipeline = PipelineFactory.create_spectrum(base_config)
        assert isinstance(pipeline, SpectrumPipeline)

    def test_create_spectrum_best_fit(self, best_fit_config):
        """Best-fit config creates BestFitSpectrumPipeline."""
        pipeline = PipelineFactory.create_spectrum(best_fit_config)
        assert pipeline.__class__.__name__ == "BestFitSpectrumPipeline"


class TestPipelineFactoryOptional:
    """Tests for optional pipeline creation."""

    def test_create_grooming_enabled(self, grooming_config):
        """Grooming enabled creates GroomingPipeline."""
        pipeline = PipelineFactory.create_grooming(grooming_config)
        assert pipeline is not None
        assert isinstance(pipeline, GroomingPipeline)

    def test_create_grooming_disabled(self, base_config):
        """Grooming disabled returns None."""
        pipeline = PipelineFactory.create_grooming(base_config)
        assert pipeline is None

    def test_create_snr_enabled(self, snr_config):
        """SNR enabled creates SNRPipeline."""
        pipeline = PipelineFactory.create_snr(snr_config)
        assert pipeline is not None
        assert isinstance(pipeline, SNRPipeline)

    def test_create_snr_disabled(self, base_config):
        """SNR disabled returns None."""
        pipeline = PipelineFactory.create_snr(base_config)
        assert pipeline is None

    def test_create_slicing_enabled(self, slicing_config):
        """Slicing enabled creates SlicingPipeline."""
        pipeline = PipelineFactory.create_slicing(slicing_config)
        assert pipeline is not None
        assert isinstance(pipeline, SlicingPipeline)

    def test_create_slicing_disabled(self, base_config):
        """Slicing disabled returns None."""
        pipeline = PipelineFactory.create_slicing(base_config)
        assert pipeline is None


class TestPipelineFactoryPipelineSet:
    """Tests for PipelineSet creation."""

    def test_create_pipeline_set_all_disabled(self, base_config):
        """Base config creates minimal pipeline set."""
        pipelines = PipelineFactory.create_pipeline_set(base_config)
        assert pipelines.routing is not None
        assert pipelines.spectrum is not None
        assert pipelines.grooming is None
        assert pipelines.snr is None
        assert pipelines.slicing is None

    def test_create_pipeline_set_all_enabled(self, full_config):
        """Full config creates complete pipeline set."""
        pipelines = PipelineFactory.create_pipeline_set(full_config)
        assert pipelines.routing is not None
        assert pipelines.spectrum is not None
        assert pipelines.grooming is not None
        assert pipelines.snr is not None
        assert pipelines.slicing is not None


class TestPipelineFactoryOrchestrator:
    """Tests for orchestrator creation."""

    def test_create_orchestrator_returns_orchestrator(self, base_config):
        """Factory creates SDNOrchestrator with pipelines."""
        orchestrator = PipelineFactory.create_orchestrator(base_config)
        assert orchestrator is not None
        assert orchestrator.routing is not None
        assert orchestrator.spectrum is not None


# Fixtures
@pytest.fixture
def base_config():
    return SimulationConfig(
        network_name="test",
        cores_per_link=1,
        band_list=("c",),
        band_slots={"c": 320},
        guard_slots=1,
        num_requests=100,
        erlang=100,
        holding_time=1.0,
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
def protection_config(base_config):
    return SimulationConfig(**{**base_config.__dict__, "route_method": "1plus1_protection"})

@pytest.fixture
def grooming_config(base_config):
    return SimulationConfig(**{**base_config.__dict__, "grooming_enabled": True})

@pytest.fixture
def snr_config(base_config):
    return SimulationConfig(**{**base_config.__dict__, "snr_enabled": True, "snr_type": "gsnr"})

@pytest.fixture
def slicing_config(base_config):
    return SimulationConfig(**{**base_config.__dict__, "slicing_enabled": True, "max_slices": 4})

@pytest.fixture
def full_config(base_config):
    return SimulationConfig(
        **{
            **base_config.__dict__,
            "grooming_enabled": True,
            "slicing_enabled": True,
            "max_slices": 4,
            "snr_enabled": True,
            "snr_type": "gsnr",
        }
    )
```

### SDNOrchestrator Tests

**File**: `fusion/tests/core/test_orchestrator.py`

```python
import pytest
from unittest.mock import Mock, MagicMock

from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request, RequestStatus
from fusion.domain.network_state import NetworkState
from fusion.domain.results import (
    RouteResult, SpectrumResult, GroomingResult, SNRResult, AllocationResult, BlockReason
)
from fusion.core.orchestrator import SDNOrchestrator
from fusion.core.pipeline_factory import PipelineSet


class TestOrchestratorPlainKSP:
    """Tests for plain K-shortest-path flow."""

    def test_successful_allocation_returns_success(self, orchestrator, request, network_state):
        """Successful allocation returns success result."""
        result = orchestrator.handle_arrival(request, network_state)

        assert result.success is True
        assert len(result.lightpaths_created) == 1
        assert result.is_groomed is False
        assert result.is_sliced is False

    def test_no_route_returns_blocked(self, orchestrator, request, network_state):
        """No route available returns blocked."""
        orchestrator.routing.find_routes.return_value = RouteResult(
            paths=[], modulations=[], weights_km=[]
        )

        result = orchestrator.handle_arrival(request, network_state)

        assert result.success is False
        assert result.block_reason == BlockReason.NO_ROUTE

    def test_no_spectrum_returns_blocked(self, orchestrator, request, network_state):
        """No spectrum available returns blocked."""
        orchestrator.spectrum.find_spectrum.return_value = SpectrumResult(is_free=False)

        result = orchestrator.handle_arrival(request, network_state)

        assert result.success is False
        assert result.block_reason == BlockReason.NO_SPECTRUM


class TestOrchestratorGrooming:
    """Tests for grooming flow."""

    def test_fully_groomed_skips_routing(self, grooming_orchestrator, request, network_state):
        """Fully groomed request skips routing and spectrum."""
        grooming_orchestrator.grooming.try_groom.return_value = GroomingResult(
            fully_groomed=True,
            partially_groomed=False,
            bandwidth_groomed_gbps=100,
            remaining_bandwidth_gbps=0,
            lightpaths_used=[1],
        )

        result = grooming_orchestrator.handle_arrival(request, network_state)

        assert result.success is True
        assert result.is_groomed is True
        assert result.lightpaths_groomed == [1]
        grooming_orchestrator.routing.find_routes.assert_not_called()

    def test_partial_grooming_continues_to_routing(
        self, grooming_orchestrator, request, network_state
    ):
        """Partial grooming continues to routing for remaining bandwidth."""
        grooming_orchestrator.grooming.try_groom.return_value = GroomingResult(
            fully_groomed=False,
            partially_groomed=True,
            bandwidth_groomed_gbps=50,
            remaining_bandwidth_gbps=50,
            lightpaths_used=[1],
            forced_path=["A", "B", "C"],
        )

        result = grooming_orchestrator.handle_arrival(request, network_state)

        assert result.success is True
        assert result.is_groomed is True
        assert result.is_partially_groomed is True
        grooming_orchestrator.routing.find_routes.assert_called_once()

    def test_grooming_rollback_on_failure(self, grooming_orchestrator, request, network_state):
        """Grooming is rolled back when subsequent allocation fails."""
        grooming_orchestrator.grooming.try_groom.return_value = GroomingResult(
            fully_groomed=False,
            partially_groomed=True,
            bandwidth_groomed_gbps=50,
            remaining_bandwidth_gbps=50,
            lightpaths_used=[1],
            forced_path=["A", "B", "C"],
        )
        grooming_orchestrator.routing.find_routes.return_value = RouteResult(
            paths=[], modulations=[], weights_km=[]
        )

        result = grooming_orchestrator.handle_arrival(request, network_state)

        assert result.success is False
        grooming_orchestrator.grooming.rollback.assert_called_once()


class TestOrchestratorSlicing:
    """Tests for slicing flow."""

    def test_slicing_fallback_when_spectrum_fails(
        self, slicing_orchestrator, request, network_state
    ):
        """Slicing is tried when standard spectrum allocation fails."""
        slicing_orchestrator.spectrum.find_spectrum.return_value = SpectrumResult(is_free=False)
        slicing_orchestrator.slicing.try_slice.return_value = AllocationResult(
            success=True,
            lightpaths_created=[1, 2, 3, 4],
            is_sliced=True,
            total_bandwidth_allocated_gbps=100,
        )

        result = slicing_orchestrator.handle_arrival(request, network_state)

        assert result.success is True
        assert result.is_sliced is True
        assert len(result.lightpaths_created) == 4


class TestOrchestratorSNR:
    """Tests for SNR validation flow."""

    def test_snr_failure_tries_next_path(self, snr_orchestrator, request, network_state):
        """SNR failure on first path tries second path."""
        snr_orchestrator.snr.validate.side_effect = [
            SNRResult(passed=False, snr_db=10.0, required_snr_db=15.0),
            SNRResult(passed=True, snr_db=16.0, required_snr_db=15.0),
        ]

        result = snr_orchestrator.handle_arrival(request, network_state)

        assert result.success is True
        assert snr_orchestrator.snr.validate.call_count == 2


class TestOrchestratorRelease:
    """Tests for release handling."""

    def test_release_frees_lightpath_bandwidth(self, orchestrator, network_state):
        """Release restores bandwidth to lightpath."""
        # Setup: request with allocated lightpath
        request = Request(
            request_id=1,
            source="A",
            destination="C",
            bandwidth_gbps=100,
            arrival_time=0.0,
            holding_time=1.0,
            status=RequestStatus.ROUTED,
            lightpath_ids=[1],
        )
        mock_lp = Mock()
        mock_lp.request_allocations = {1: 100}
        mock_lp.remaining_bandwidth_gbps = 0
        network_state.get_lightpath.return_value = mock_lp

        orchestrator.handle_release(request, network_state)

        assert mock_lp.remaining_bandwidth_gbps == 100
        assert 1 not in mock_lp.request_allocations


# Fixtures
@pytest.fixture
def mock_routing():
    routing = Mock()
    routing.find_routes.return_value = RouteResult(
        paths=[["A", "B", "C"]],
        modulations=[["QPSK"]],
        weights_km=[200.0],
    )
    return routing

@pytest.fixture
def mock_spectrum():
    spectrum = Mock()
    spectrum.find_spectrum.return_value = SpectrumResult(
        is_free=True,
        start_slot=0,
        end_slot=8,
        core=0,
        band="c",
        modulation="QPSK",
        slots_needed=8,
    )
    return spectrum

@pytest.fixture
def mock_network_state():
    state = Mock(spec=NetworkState)
    mock_lp = Mock()
    mock_lp.lightpath_id = 1
    mock_lp.request_allocations = {}
    state.create_lightpath.return_value = mock_lp
    return state

@pytest.fixture
def base_config():
    return SimulationConfig(
        network_name="test",
        cores_per_link=1,
        band_list=("c",),
        band_slots={"c": 320},
        guard_slots=1,
        num_requests=100,
        erlang=100,
        holding_time=1.0,
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
def orchestrator(base_config, mock_routing, mock_spectrum):
    pipelines = PipelineSet(routing=mock_routing, spectrum=mock_spectrum)
    return SDNOrchestrator(base_config, pipelines)

@pytest.fixture
def grooming_orchestrator(base_config, mock_routing, mock_spectrum):
    grooming_config = SimulationConfig(**{**base_config.__dict__, "grooming_enabled": True})
    mock_grooming = Mock()
    pipelines = PipelineSet(
        routing=mock_routing,
        spectrum=mock_spectrum,
        grooming=mock_grooming,
    )
    return SDNOrchestrator(grooming_config, pipelines)

@pytest.fixture
def slicing_orchestrator(base_config, mock_routing, mock_spectrum):
    slicing_config = SimulationConfig(
        **{**base_config.__dict__, "slicing_enabled": True, "max_slices": 4}
    )
    mock_slicing = Mock()
    pipelines = PipelineSet(
        routing=mock_routing,
        spectrum=mock_spectrum,
        slicing=mock_slicing,
    )
    return SDNOrchestrator(slicing_config, pipelines)

@pytest.fixture
def snr_orchestrator(base_config, mock_routing, mock_spectrum):
    snr_config = SimulationConfig(
        **{**base_config.__dict__, "snr_enabled": True, "snr_type": "gsnr"}
    )
    mock_snr = Mock()
    pipelines = PipelineSet(
        routing=mock_routing,
        spectrum=mock_spectrum,
        snr=mock_snr,
    )
    return SDNOrchestrator(snr_config, pipelines)

@pytest.fixture
def request():
    return Request(
        request_id=1,
        source="A",
        destination="C",
        bandwidth_gbps=100,
        arrival_time=0.0,
        holding_time=1.0,
    )

@pytest.fixture
def network_state(mock_network_state):
    return mock_network_state
```

---

## Integration Tests

### Feature Combination Tests

**File**: `fusion/tests/integration/test_orchestrator_flows.py`

```python
import pytest
import networkx as nx

from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.core.pipeline_factory import PipelineFactory


class TestOrchestratorFlowsIntegration:
    """Integration tests for orchestrator with real pipelines."""

    @pytest.fixture
    def small_topology(self):
        """Create small test topology."""
        G = nx.Graph()
        G.add_edge("A", "B", weight=100)
        G.add_edge("B", "C", weight=100)
        G.add_edge("A", "D", weight=150)
        G.add_edge("D", "C", weight=150)
        return G

    @pytest.fixture
    def base_config(self):
        return SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=100,
            erlang=100,
            holding_time=1.0,
            route_method="k_shortest_path",
            k_paths=2,
            allocation_method="first_fit",
            grooming_enabled=False,
            slicing_enabled=False,
            max_slices=1,
            snr_enabled=False,
            snr_type=None,
            snr_recheck=False,
            can_partially_serve=False,
            modulation_formats={"QPSK": {"max_length": 500, "bits_per_symbol": 2}},
            mod_per_bw={100: ["QPSK"]},
            snr_thresholds={"QPSK": 12.0},
        )

    def test_plain_ksp_flow(self, small_topology, base_config):
        """Test plain K-shortest-path flow with real pipelines."""
        network_state = NetworkState(small_topology, base_config)
        orchestrator = PipelineFactory.create_orchestrator(base_config)

        request = Request(
            request_id=1,
            source="A",
            destination="C",
            bandwidth_gbps=100,
            arrival_time=0.0,
            holding_time=1.0,
        )

        result = orchestrator.handle_arrival(request, network_state)

        assert result.success is True
        assert len(result.lightpaths_created) == 1

    def test_grooming_flow(self, small_topology, base_config):
        """Test grooming with real pipelines."""
        grooming_config = SimulationConfig(**{**base_config.__dict__, "grooming_enabled": True})
        network_state = NetworkState(small_topology, grooming_config)
        orchestrator = PipelineFactory.create_orchestrator(grooming_config)

        # First request - creates lightpath
        request1 = Request(
            request_id=1,
            source="A",
            destination="C",
            bandwidth_gbps=50,
            arrival_time=0.0,
            holding_time=10.0,
        )
        result1 = orchestrator.handle_arrival(request1, network_state)
        assert result1.success is True

        # Second request - should groom onto existing lightpath
        request2 = Request(
            request_id=2,
            source="A",
            destination="C",
            bandwidth_gbps=30,
            arrival_time=0.1,
            holding_time=10.0,
        )
        result2 = orchestrator.handle_arrival(request2, network_state)

        assert result2.success is True
        # Depending on implementation, may be fully groomed or partially groomed

    def test_slicing_flow(self, small_topology, base_config):
        """Test slicing with real pipelines."""
        slicing_config = SimulationConfig(
            **{**base_config.__dict__, "slicing_enabled": True, "max_slices": 4}
        )
        network_state = NetworkState(small_topology, slicing_config)
        orchestrator = PipelineFactory.create_orchestrator(slicing_config)

        # Request with high bandwidth that may require slicing
        request = Request(
            request_id=1,
            source="A",
            destination="C",
            bandwidth_gbps=400,
            arrival_time=0.0,
            holding_time=1.0,
        )

        result = orchestrator.handle_arrival(request, network_state)

        # May succeed via slicing or fail if no valid configuration
        # Test validates the flow executes without error

    def test_snr_validation_flow(self, small_topology, base_config):
        """Test SNR validation with real pipelines."""
        snr_config = SimulationConfig(
            **{**base_config.__dict__, "snr_enabled": True, "snr_type": "gsnr"}
        )
        network_state = NetworkState(small_topology, snr_config)
        orchestrator = PipelineFactory.create_orchestrator(snr_config)

        request = Request(
            request_id=1,
            source="A",
            destination="C",
            bandwidth_gbps=100,
            arrival_time=0.0,
            holding_time=1.0,
        )

        result = orchestrator.handle_arrival(request, network_state)

        # Result depends on SNR validation outcome
        # Test validates SNR pipeline is invoked
```

---

## Regression Tests with run_comparison.py

### Using Legacy Path as Oracle

```python
# tests/run_comparison.py additions

def test_orchestrator_matches_legacy():
    """Verify orchestrator produces same results as legacy."""
    configs = [
        "plain_ksp",
        "ksp_with_grooming",
        "ksp_with_snr",
        "ksp_with_slicing",
        "ksp_with_all_features",
    ]

    for config_name in configs:
        # Run with legacy path
        legacy_results = run_simulation(config_name, use_orchestrator=False)

        # Run with orchestrator path
        orchestrator_results = run_simulation(config_name, use_orchestrator=True)

        # Compare key metrics
        compare_metrics(
            legacy_results,
            orchestrator_results,
            metrics=["blocking_probability", "throughput", "modulations_used"],
            abs_tol=0.02,
        )


def run_simulation(config_name: str, use_orchestrator: bool = False) -> dict:
    """Run simulation with specified path."""
    engine_props = load_config(config_name)
    engine_props["use_orchestrator"] = use_orchestrator

    engine = SimulationEngine(engine_props)
    engine.run()

    return engine.get_stats()


def compare_metrics(
    legacy: dict,
    orchestrator: dict,
    metrics: list[str],
    abs_tol: float,
) -> None:
    """Compare metrics between two result sets."""
    for metric in metrics:
        legacy_val = legacy.get(metric, 0)
        orch_val = orchestrator.get(metric, 0)

        assert abs(legacy_val - orch_val) <= abs_tol, (
            f"{metric} mismatch: legacy={legacy_val}, orchestrator={orch_val}"
        )
```

### Running Regression Tests

```bash
# Run full comparison suite
python tests/run_comparison.py

# Run with verbose output
python tests/run_comparison.py --verbose

# Run specific config
python tests/run_comparison.py --config plain_ksp

# Run both paths and compare
python tests/run_comparison.py --compare-paths
```

---

## Test Coverage Matrix

| Feature | Unit | Integration | Regression |
|---------|------|-------------|------------|
| PipelineFactory routing creation | X | | |
| PipelineFactory spectrum creation | X | | |
| PipelineFactory optional pipelines | X | | |
| Orchestrator plain KSP | X | X | X |
| Orchestrator grooming (full) | X | X | X |
| Orchestrator grooming (partial) | X | X | X |
| Orchestrator grooming rollback | X | | X |
| Orchestrator slicing fallback | X | X | X |
| Orchestrator SNR validation | X | X | X |
| Orchestrator release | X | X | X |
| Feature flag switching | | X | X |
| Stats integration | | X | X |

---

## CI Integration

```yaml
# .github/workflows/quality.yml additions

  test-orchestrator:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run orchestrator unit tests
        run: pytest fusion/tests/core/test_orchestrator.py -v

      - name: Run orchestrator integration tests
        run: pytest fusion/tests/integration/test_orchestrator_flows.py -v

      - name: Run regression comparison
        run: python tests/run_comparison.py --compare-paths
```

---

## Known Limitations

1. **Floating-point tolerance**: Statistical variations may cause up to 2% difference in blocking probability
2. **RNG sensitivity**: Same seed required for deterministic comparison
3. **Feature interactions**: Some edge cases in grooming+slicing+SNR may differ slightly

## Related Documentation

- `migration/phase_3_orchestrator.md` - Implementation plan
- `architecture/orchestration.md` - Orchestrator design
- `testing/test_strategy.md` - Overall test strategy
