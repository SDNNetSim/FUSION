# Phase 6: Quality Assurance

## 50 - Testing Requirements

**Section Reference**: Section 5 - Testing Requirements

**Purpose**: Define comprehensive testing strategy for survivability v1 including unit tests, integration tests, and regression tests to ensure code quality and backward compatibility.

---

## 1. Module-Level Unit Tests (Required)

Each new module MUST have:
- `tests/` subdirectory with `__init__.py` and `README.md`
- Unit tests for all public functions and classes
- **Test coverage ≥ 80%** (aim for 90% on critical modules)
- Mocks for external dependencies (topology, file I/O, etc.)
- Parametrized tests for multiple scenarios
- Clear test names following `test_<what>_<when>_<expected>` pattern

### Example Test Structure

```python
# fusion/modules/failures/tests/test_failure_manager.py

import pytest
import networkx as nx
from fusion.modules.failures import FailureManager


@pytest.fixture
def sample_topology():
    """Create sample topology for testing."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)])
    return G


def test_link_failure_blocks_path_when_link_failed(sample_topology):
    """Test that path using failed link is marked infeasible."""
    manager = FailureManager({}, sample_topology)
    manager.inject_failure('link', t_fail=10.0, t_repair=20.0, link_id=(1, 2))

    assert not manager.is_path_feasible([0, 1, 2, 3])
    assert manager.is_path_feasible([0, 4, 3])
```

---

## 2. Integration Tests (Required)

### End-to-End Survivability Test

**Location**: `tests/integration/test_survivability_pipeline.py`

```python
"""
End-to-end survivability integration test.
"""

def test_survivability_pipeline_complete():
    """
    Test complete survivability workflow:
    1. Run simulation with failures enabled
    2. Verify recovery metrics computed correctly
    3. Check dataset logging produces valid JSONL
    4. Validate RL policy inference runs without errors
    """
    config = {
        'num_requests': 1000,
        'failure_settings': {'failure_type': 'link', ...},
        'offline_rl_settings': {'policy_type': 'ksp_ff'},
        'dataset_logging': {'log_offline_dataset': True}
    }

    engine = SimulationEngine(config)
    stats = engine.run()

    # Verify recovery metrics
    assert len(stats.recovery_times_ms) > 0
    assert stats.get_recovery_stats()['mean_ms'] > 0

    # Verify dataset logged
    assert Path(config['dataset_logging']['dataset_output_path']).exists()

    # Verify no errors
    assert stats.compute_blocking_probability() >= 0
```

---

## 3. Regression Tests (Required)

### Backward Compatibility

**Location**: `tests/regression/test_backward_compat.py`

```python
"""
Regression tests for backward compatibility.
"""

def test_existing_simulations_unchanged():
    """Test that existing simulations (no failures) produce identical results."""
    # Legacy config without survivability features
    legacy_config = load_config('legacy_experiment.ini')

    results_before = run_reference_simulation(legacy_config)
    results_after = SimulationEngine(legacy_config).run()

    assert results_before['bp_overall'] == pytest.approx(results_after.compute_blocking_probability())


def test_legacy_configurations_still_work():
    """Test that legacy configuration files still work."""
    config = load_config('legacy_config.ini')
    engine = SimulationEngine(config)
    assert engine is not None


def test_existing_routing_algorithms_unaffected():
    """Test that existing routing/spectrum algorithms unaffected."""
    pass  # Verify KSP, First-Fit, etc. produce same results
```

---

## 4. Test Coverage Requirements

### Module-Specific Targets

| Module | Coverage Target | Critical Functions |
|--------|----------------|-------------------|
| `failures/` | 90% | `inject_failure`, `is_path_feasible` |
| `rl/policies/` | 85% | `select_path`, action masking |
| `routing/one_plus_one` | 85% | `find_disjoint_paths` |
| `routing/k_path_cache` | 80% | `get_path_features` |
| `reporting/dataset_logger` | 85% | `log_transition` |

### Running Coverage

```bash
# Module-level coverage
pytest fusion/modules/failures/tests/ --cov=fusion.modules.failures --cov-report=html

# Overall survivability coverage
pytest tests/ -k survivability --cov=fusion --cov-report=term-missing
```

---

## 5. Testing Best Practices

### AAA Pattern (Arrange-Act-Assert)

```python
def test_action_mask_filters_infeasible_paths():
    # Arrange
    k_paths = [[0, 1, 2], [0, 3, 2]]
    features = [
        {'failure_mask': 1, 'min_residual_slots': 10},  # Failed
        {'failure_mask': 0, 'min_residual_slots': 5}    # OK
    ]

    # Act
    mask = compute_action_mask(k_paths, features, slots_needed=4)

    # Assert
    assert mask == [False, True]
```

### Parametrized Tests

```python
@pytest.mark.parametrize("failure_type,expected_links", [
    ('link', 1),
    ('srlg', 3),
    ('geo', 5)
])
def test_failure_types_affect_correct_links(failure_type, expected_links):
    manager = FailureManager({}, topology)
    event = manager.inject_failure(failure_type, ...)
    assert len(event['failed_links']) == expected_links
```

---

## 6. Acceptance Criteria

- [x] All modules have ≥ 80% test coverage
- [x] End-to-end integration test passes
- [x] Regression tests verify backward compatibility
- [x] All tests pass with `pytest tests/`
- [x] No test warnings or deprecation notices
- [x] Tests run in < 5 minutes (unit + integration)

---

**Related Documents**:
- [51-documentation.md](51-documentation.md) (Documentation standards)
- [52-performance.md](52-performance.md) (Performance testing)
