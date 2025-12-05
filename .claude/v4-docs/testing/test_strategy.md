# V4 Testing Strategy

This document defines the overall testing philosophy, structure, and practices for the FUSION V4 architecture.

## Overview

Testing in V4 follows a layered approach with clear boundaries between test types:

| Layer | Purpose | Speed | Isolation |
|-------|---------|-------|-----------|
| Unit | Test individual components | Fast (ms) | Full mocking |
| Integration | Test component interactions | Medium (s) | Partial mocking |
| Regression | Verify behavioral equivalence | Slow (min) | No mocking |

## Test Philosophy

### 1. Test Behavior, Not Implementation

```python
# GOOD: Tests the behavior
def test_routing_returns_paths_between_endpoints():
    result = strategy.find_routes("A", "D", 100, network_state)
    assert result.paths[0][0] == "A"
    assert result.paths[0][-1] == "D"

# BAD: Tests implementation details
def test_routing_uses_dijkstra():
    strategy.find_routes("A", "D", 100, network_state)
    assert strategy._dijkstra_called == True  # Implementation detail
```

### 2. Arrange-Act-Assert (AAA) Pattern

```python
def test_lightpath_creation_allocates_spectrum():
    # Arrange
    network_state = NetworkState(topology, config)
    path = ["A", "B", "C"]

    # Act
    lightpath = network_state.create_lightpath(
        path=path, start_slot=0, end_slot=8, core=0, band="c",
        modulation="QPSK", bandwidth_gbps=100, path_weight_km=200.0,
    )

    # Assert
    assert not network_state.is_spectrum_available(path, 0, 8, 0, "c")
```

### 3. Test Naming Convention

```
test_<what>_<when>_<expected>
```

Examples:
- `test_find_routes_when_no_path_exists_returns_empty_result`
- `test_create_lightpath_when_spectrum_occupied_raises_error`
- `test_grooming_when_fully_satisfied_skips_routing`

### 4. One Assertion Per Concept

```python
# GOOD: One concept per test
def test_lightpath_has_correct_id():
    lp = network_state.create_lightpath(...)
    assert lp.lightpath_id == 1

def test_lightpath_has_correct_path():
    lp = network_state.create_lightpath(...)
    assert lp.path == ["A", "B", "C"]

# ACCEPTABLE: Related assertions grouped
def test_lightpath_properties():
    lp = network_state.create_lightpath(...)
    assert lp.lightpath_id == 1
    assert lp.path == ["A", "B", "C"]
    assert lp.bandwidth_gbps == 100
```

---

## Unit Testing

### Scope

Unit tests verify individual components in isolation:

- Domain objects (`Request`, `Lightpath`, `SimulationConfig`)
- `NetworkState` methods
- Individual pipeline implementations
- Result object construction

### Structure

```
fusion/tests/
    domain/
        test_request.py
        test_lightpath.py
        test_network_state.py
        test_link_spectrum.py
        test_config.py
    routing/
        strategies/
            test_ksp.py
            test_load_balanced.py
            test_protection.py
    pipelines/
        test_spectrum_pipeline.py
        test_grooming_pipeline.py
        test_snr_pipeline.py
        test_slicing_pipeline.py
    core/
        test_pipeline_factory.py
        test_orchestrator.py
    adapters/
        test_routing_adapter.py
        test_spectrum_adapter.py
```

### Mocking Strategy

Use mocks for external dependencies:

```python
from unittest.mock import Mock

def test_orchestrator_calls_routing():
    # Mock the routing pipeline
    mock_routing = Mock()
    mock_routing.find_routes.return_value = RouteResult(
        paths=[["A", "B", "C"]],
        weights_km=[200.0],
        modulations=[["QPSK"]],
    )

    # Mock spectrum pipeline
    mock_spectrum = Mock()
    mock_spectrum.find_spectrum.return_value = SpectrumResult(is_free=True, ...)

    # Create orchestrator with mocks
    pipelines = PipelineSet(routing=mock_routing, spectrum=mock_spectrum)
    orchestrator = SDNOrchestrator(config, pipelines)

    # Act
    result = orchestrator.handle_arrival(request, network_state)

    # Assert
    mock_routing.find_routes.assert_called_once()
    assert result.success
```

### Fixtures

Use pytest fixtures for common setup:

```python
# conftest.py

@pytest.fixture
def simple_config():
    return SimulationConfig(
        network_name="test",
        cores_per_link=1,
        band_list=("c",),
        band_slots={"c": 320},
        # ... minimal required fields
    )

@pytest.fixture
def diamond_topology():
    G = nx.Graph()
    G.add_edge("A", "B", weight=100)
    G.add_edge("A", "C", weight=100)
    G.add_edge("B", "D", weight=100)
    G.add_edge("C", "D", weight=100)
    return G

@pytest.fixture
def network_state(diamond_topology, simple_config):
    return NetworkState(diamond_topology, simple_config)
```

---

## Integration Testing

### Scope

Integration tests verify component interactions:

- Orchestrator with real pipelines
- Pipeline chains (routing -> spectrum -> lightpath creation)
- State consistency across operations

### Structure

```
fusion/tests/integration/
    test_orchestrator_flows.py
    test_state_consistency.py
    test_legacy_parity.py
    test_feature_combinations.py
```

### Example: Orchestrator Flow Integration

```python
class TestOrchestratorFlowsIntegration:
    """Integration tests with real pipelines."""

    @pytest.fixture
    def real_orchestrator(self, diamond_topology, base_config):
        network_state = NetworkState(diamond_topology, base_config)
        orchestrator = PipelineFactory.create_orchestrator(base_config)
        return orchestrator, network_state

    def test_plain_ksp_end_to_end(self, real_orchestrator):
        orchestrator, network_state = real_orchestrator

        request = Request(
            request_id=1, source="A", destination="D",
            bandwidth_gbps=100, arrival_time=0.0, holding_time=1.0,
        )

        result = orchestrator.handle_arrival(request, network_state)

        assert result.success
        assert len(result.lightpaths_created) == 1

        # Verify state changed
        lp = network_state.get_lightpath(result.lightpaths_created[0])
        assert lp is not None
        assert lp.path[0] == "A"
        assert lp.path[-1] == "D"
```

### State Consistency Tests

```python
class TestStateConsistency:
    """Verify state invariants hold across operations."""

    def test_create_release_restores_state(self, network_state):
        # Capture initial state
        initial = self._snapshot(network_state)

        # Create and release
        lp = network_state.create_lightpath(...)
        network_state.release_lightpath(lp.lightpath_id)

        # Verify restoration
        final = self._snapshot(network_state)
        assert initial == final

    def test_no_double_allocation(self, network_state):
        lp1 = network_state.create_lightpath(
            path=["A", "B"], start_slot=0, end_slot=8, ...
        )

        with pytest.raises(ValueError, match="Spectrum not available"):
            network_state.create_lightpath(
                path=["A", "B"], start_slot=4, end_slot=12, ...  # Overlaps
            )
```

---

## Regression Testing

### Purpose

Regression tests ensure the new V4 architecture produces identical results to the legacy implementation. This is critical during migration.

### Oracle: Legacy Path

The legacy `SDNController` serves as the "oracle" - the source of expected behavior:

```python
def test_v4_matches_legacy():
    # Run with legacy path
    legacy_results = run_simulation(config, use_orchestrator=False)

    # Run with V4 path
    v4_results = run_simulation(config, use_orchestrator=True)

    # Compare key metrics
    assert abs(legacy_results["blocking_prob"] - v4_results["blocking_prob"]) < 0.02
    assert abs(legacy_results["throughput"] - v4_results["throughput"]) < 0.02
```

### run_comparison.py

The primary regression tool:

```bash
# Run all configs
python tests/run_comparison.py --all-configs

# Compare specific config
python tests/run_comparison.py --config plain_ksp

# Dual-path comparison
python tests/run_comparison.py --compare-paths

# With specific seed
python tests/run_comparison.py --compare-paths --seed 42

# Verbose output
python tests/run_comparison.py --compare-paths --verbose
```

### Tolerance Settings

Statistical simulations have inherent variability. Acceptable tolerances:

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Blocking probability | 2% | Statistical variation |
| Throughput | 2% | Depends on blocking |
| Modulation distribution | 5% | Sensitive to path choices |
| Average path length | 1% | Deterministic |

### Configuration Matrix

Test across all feature combinations:

| Config | Grooming | Slicing | SNR | Protection |
|--------|----------|---------|-----|------------|
| plain_ksp | | | | |
| ksp_grooming | X | | | |
| ksp_snr | | | X | |
| ksp_slicing | | X | | |
| ksp_grooming_snr | X | | X | |
| ksp_slicing_snr | | X | X | |
| ksp_all_features | X | X | X | |
| protected_ksp | | | | X |
| protected_snr | | | X | X |

---

## Coverage Targets

### By Module

| Module | Target | Rationale |
|--------|--------|-----------|
| `fusion/domain/` | 90% | Core data structures |
| `fusion/interfaces/` | 100% | Just type definitions |
| `fusion/routing/strategies/` | 85% | Algorithm implementations |
| `fusion/pipelines/` | 85% | Pipeline implementations |
| `fusion/core/orchestrator.py` | 90% | Critical coordination |
| `fusion/core/adapters/` | 80% | Bridge to legacy |

### Measuring Coverage

```bash
# Run with coverage
pytest fusion/tests/ --cov=fusion --cov-report=html

# Check specific module
pytest fusion/tests/domain/ --cov=fusion/domain --cov-report=term-missing

# Fail if below threshold
pytest --cov=fusion --cov-fail-under=80
```

---

## Test Evolution by Phase

### Phase 1: Core Model

Focus: Domain objects and basic types

```
Tests Added:
- test_request.py
- test_lightpath.py
- test_simulation_config.py
- test_result_objects.py
```

### Phase 2: NetworkState

Focus: State management and legacy parity

```
Tests Added:
- test_network_state.py
- test_link_spectrum.py
- test_adapters/*.py
- test_legacy_parity.py
```

### Phase 3: Orchestrator

Focus: Request flows and feature combinations

```
Tests Added:
- test_pipeline_factory.py
- test_orchestrator.py
- test_orchestrator_flows.py
- run_comparison.py enhancements
```

### Phases 4-6: RL/ML and Legacy Removal

Focus: Environment integration and cleanup

```
Tests Added:
- test_rl_environment.py
- test_ml_protection.py
Tests Removed:
- Legacy compatibility tests
```

---

## CI Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run unit tests
        run: pytest fusion/tests/ -v --cov=fusion --cov-fail-under=80

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run integration tests
        run: pytest fusion/tests/integration/ -v

  regression-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run regression comparison
        run: python tests/run_comparison.py --compare-paths --seed 42
```

---

## Debugging Test Failures

### Common Issues

#### 1. Flaky Tests (Random Failures)

**Symptom**: Test passes sometimes, fails other times

**Solution**: Fix RNG seed
```python
@pytest.fixture
def seeded_rng():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
```

#### 2. State Leakage

**Symptom**: Tests fail when run together, pass individually

**Solution**: Use fixtures that create fresh state
```python
@pytest.fixture
def fresh_network_state(topology, config):
    return NetworkState(topology, config)  # New instance each test
```

#### 3. Tolerance Failures

**Symptom**: Regression test fails with small differences

**Solution**: Adjust tolerance or investigate root cause
```python
# Check if difference is consistent
for seed in range(10):
    legacy = run_simulation(config, seed=seed, use_orchestrator=False)
    v4 = run_simulation(config, seed=seed, use_orchestrator=True)
    print(f"Seed {seed}: diff={abs(legacy['bp'] - v4['bp']):.4f}")
```

### Debug Commands

```bash
# Run single test with output
pytest fusion/tests/core/test_orchestrator.py::test_name -v -s

# Run with debugger on failure
pytest fusion/tests/ --pdb

# Show slowest tests
pytest fusion/tests/ --durations=10

# Run only failed tests from last run
pytest fusion/tests/ --lf
```

---

## Related Documentation

- [Testing: Phase 2 Testing](./phase_2_testing.md)
- [Testing: Phase 3 Testing](./phase_3_testing.md)
- [Migration: Phase 2 Checklist](../migration/phase_2_checklist.md)
- [Migration: Phase 3 Checklist](../migration/phase_3_checklist.md)
