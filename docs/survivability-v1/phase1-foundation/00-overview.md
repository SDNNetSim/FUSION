# Phase 1: Foundation & Setup

## 00 - Project Context & Integration Points

**Purpose**: Understand FUSION's existing architecture and how survivability features will integrate.

---

## 0.1 Existing FUSION Architecture

The FUSION simulator is built around a modular architecture with clear separation of concerns:

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Simulation Engine** | `fusion/core/simulation.py` | Main simulation loop, event handling (`SimulationEngine` class) |
| **SDN Controller** | `fusion/core/sdn_controller.py` | Routing and spectrum assignment orchestration (`SDNController` class) |
| **Properties** | `fusion/core/properties.py` | Configuration objects (RoutingProps, SpectrumProps, SDNProps, etc.) |
| **Routing Module** | `fusion/modules/routing/` | Registry-based routing algorithms (KSP, KSPLC, etc.) |
| **Spectrum Module** | `fusion/modules/spectrum/` | Spectrum assignment algorithms (First-Fit, Best-Fit, Last-Fit) |
| **Statistics** | `fusion/reporting/statistics.py` | Metrics collection (`SimulationStatistics`, `GroomingStatistics`) |
| **Configuration** | `fusion/configs/` | INI/YAML/JSON configuration with validation |
| **RL Infrastructure** | `fusion/modules/rl/` | Stable-Baselines3 integration, agents, environments |

### Key Integration Points for Survivability

The survivability features will integrate at these points:

1. **SimulationEngine** - Add `FailureManager` and `DatasetLogger` instances
2. **SDNController** - Integrate K-path cache, policy selection, action masking
3. **Properties (SDNProps)** - Extend with protection paths, timing parameters
4. **Routing Module** - Add `OnePlusOneProtection` and `KPathCache`
5. **Statistics** - Add recovery time tracking, fragmentation metrics
6. **Configuration** - Add failure, protection, and RL policy settings

---

## 0.2 FUSION Coding Standards (Must Follow)

All new code MUST adhere to these standards:

### Naming Conventions

- **Functions/Variables**: `snake_case`
  ```python
  def compute_path_features(path: list[int]) -> dict[str, Any]:
      failure_mask = 0
      min_residual_slots = 100
  ```

- **Classes**: `PascalCase`
  ```python
  class FailureManager:
      pass

  class OnePlusOneProtection(AbstractRouter):
      pass
  ```

- **Constants**: `SCREAMING_SNAKE_CASE`
  ```python
  MAX_K_PATHS = 5
  DEFAULT_PROTECTION_SWITCHOVER_MS = 50.0
  ```

### Type Hints

**Required on all functions, parameters, and returns**:

```python
def inject_failure(
    self,
    failure_type: str,
    t_fail: float,
    t_repair: float,
    **kwargs: Any
) -> dict[str, Any]:
    pass
```

### Docstrings

**Sphinx format required** (`:param:`, `:type:`, `:return:`, `:raises:`):

```python
def is_path_feasible(self, path: list[int]) -> bool:
    """
    Check if path is feasible given active failures.

    :param path: List of node IDs forming the path
    :type path: list[int]
    :return: True if path has no failed links
    :rtype: bool
    """
    pass
```

### File Organization

- **File Size**: < 500 lines per file (split if exceeded)
- **Function Size**: < 50 lines per function
- **Module Structure**: Each module has:
  - `__init__.py` (with `__all__` exports)
  - `README.md` (module documentation)
  - `registry.py` (if multi-component)
  - `tests/` subdirectory

### Error Handling

- **Custom exceptions**: Define specific exception classes
- **No bare `except`**: Always specify exception types

```python
# Good
try:
    result = process_failure()
except FailureConfigError as e:
    logger.error(f"Configuration error: {e}")
    raise

# Bad
try:
    result = process_failure()
except:  # Don't do this!
    pass
```

### Path Handling

- **Use `pathlib.Path`**: No string path manipulation
- **Get paths from config**: No hardcoded paths

```python
from pathlib import Path

# Good
output_path = Path(engine_props.get('dataset_output_path'))
output_path.parent.mkdir(parents=True, exist_ok=True)

# Bad
output_path = "/home/user/datasets/output.jsonl"  # Hardcoded!
```

---

## 0.3 FUSION Testing Standards (Must Follow)

### Test Location

- **Module-level tests**: `fusion/modules/<module>/tests/`
- **Test naming**: `test_<what>_<when>_<expected>`

```python
def test_link_failure_blocks_path():
    """Test that a path using a failed link is marked infeasible."""
    pass

def test_failure_repair_restores_path():
    """Test that path becomes feasible after repair time."""
    pass
```

### Test Structure (AAA Pattern)

```python
def test_action_mask_computation():
    # Arrange
    k_paths = [[0, 1, 2], [0, 3, 2]]
    k_path_features = [
        {'failure_mask': 0, 'min_residual_slots': 10},
        {'failure_mask': 1, 'min_residual_slots': 8}
    ]
    slots_needed = 5

    # Act
    action_mask = compute_action_mask(k_paths, k_path_features, slots_needed)

    # Assert
    assert action_mask == [True, False]  # Second path failed
```

### Mocking

**Mock external dependencies in unit tests**:

```python
from unittest.mock import Mock, patch

def test_failure_manager_with_mocked_topology():
    # Mock the topology
    mock_topology = Mock(spec=nx.Graph)
    mock_topology.edges.return_value = [(0, 1), (1, 2)]

    # Test with mock
    manager = FailureManager(engine_props, mock_topology)
    assert manager.topology is mock_topology
```

### Fixtures

**Use pytest fixtures for test data**:

```python
import pytest

@pytest.fixture
def sample_topology():
    """Provide a sample topology for tests."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
    return G

def test_geo_failure_with_fixture(sample_topology):
    result = fail_geo(sample_topology, center_node=1, hop_radius=1, t_fail=10.0, t_repair=20.0)
    assert len(result['failed_links']) > 0
```

### Coverage

**Run tests with coverage**:

```bash
pytest --cov=fusion.modules.failures fusion/modules/failures/tests/ -v
```

**Coverage targets**: 80-90% for all new modules

---

## 0.4 Key Architectural Principles

### 1. Minimal Invasiveness

**Extend, don't replace**:
- Add new modules without modifying core functionality
- Use composition over inheritance where possible
- Preserve backward compatibility

### 2. Registry Pattern

**For multi-component modules**:

```python
# fusion/modules/failures/registry.py
FAILURE_TYPES = {
    'link': fail_link,
    'node': fail_node,
    'srlg': fail_srlg,
    'geo': fail_geo
}

def get_failure_handler(failure_type: str) -> Callable:
    """Get failure handler by type."""
    if failure_type not in FAILURE_TYPES:
        raise FailureConfigError(f"Unknown failure type: {failure_type}")
    return FAILURE_TYPES[failure_type]
```

### 3. Properties-Based Configuration

**Use FUSION's properties objects**:

```python
# Extend existing SDNProps
class SDNProps:
    def __init__(self):
        # Existing attributes...

        # New survivability attributes
        self.protection_mode: str | None = None
        self.primary_path: list[int] | None = None
        self.backup_path: list[int] | None = None
        self.active_path: str = "primary"
        self.protection_switchover_ms: float = 50.0
```

### 4. Determinism First

**All randomness must be seedable**:

```python
def seed_all_rngs(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

---

## 0.5 Dependencies

### Internal Dependencies

New modules will depend on existing FUSION components:

```
fusion/modules/failures/
  ├── networkx (graph operations)
  ├── fusion/core/properties (configuration)
  └── fusion/interfaces/* (if applicable)

fusion/modules/rl/policies/
  ├── torch (model inference)
  ├── fusion/interfaces/router (policy interface)
  └── fusion/modules/routing (K-path cache)
```

### External Dependencies

Already included in FUSION (no new dependencies needed):

- **NetworkX**: Graph operations and shortest path algorithms
- **PyTorch**: RL model inference
- **NumPy**: Numerical operations and statistics
- **Pandas**: Data processing (optional, for batch results)

---

## 0.6 Integration Checklist

Before implementing each module, verify:

- [ ] Understand where it fits in FUSION's architecture
- [ ] Identify which existing classes/modules it extends
- [ ] Know which properties objects need extension
- [ ] Plan integration points in SimulationEngine/SDNController
- [ ] Design registry structure (if multi-component)
- [ ] Define custom exceptions
- [ ] Plan test fixtures and mocks
- [ ] Check file size limits (< 500 lines)

---

## Next Steps

After understanding the project context:

1. **Review** [01-scope-boundaries.md](01-scope-boundaries.md) to understand what's in/out of scope
2. **Study** [02-module-summary.md](02-module-summary.md) for module-level overview
3. **Learn** [03-version-control.md](03-version-control.md) for branching strategy

Then proceed to **Phase 2** for implementation.

---

**Related Documents**:
- FUSION [CODING_STANDARDS.md](../../../CODING_STANDARDS.md)
- FUSION [TESTING_STANDARDS.md](../../../TESTING_STANDARDS.md)
- [02-module-summary.md](02-module-summary.md) (Module overview)
