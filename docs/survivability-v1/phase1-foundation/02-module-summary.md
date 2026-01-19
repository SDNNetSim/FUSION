# Phase 1: Foundation & Setup

## 02 - Module-by-Module Summary

**Purpose**: High-level overview of all modules to be created or extended for survivability support.

---

## New Modules to Create

### 1. Failures Module

**Location**: `fusion/modules/failures/`

**Purpose**: Injection and tracking of network failures (link, node, SRLG, geographic).

**Structure**:
```
fusion/modules/failures/
├── __init__.py              # Public API exports
├── README.md                # Module documentation
├── registry.py              # Failure injection registry
├── errors.py                # Custom exceptions (FailureConfigError, etc.)
├── failure_manager.py       # Main FailureManager class
├── failure_types.py         # LinkFailure, NodeFailure, SRLGFailure, GeoFailure
├── utils.py                 # Helper functions
└── tests/
    ├── __init__.py
    ├── README.md
    ├── test_failure_manager.py
    ├── test_failure_types.py
    └── fixtures/            # Test data files
```

**Key Classes**:
- `FailureManager`: Manages active failures, tracks history, checks path feasibility
- `fail_link()`, `fail_node()`, `fail_srlg()`, `fail_geo()`: Failure injection functions

**Dependencies**:
- NetworkX (graph operations)
- `fusion/core/properties`

**Estimated Effort**: 1.5-2 days

**Details**: See [10-failure-module.md](../phase2-infrastructure/10-failure-module.md)

---

### 2. RL Policies Module

**Location**: `fusion/modules/rl/policies/`

**Purpose**: Path selection policies (heuristic baselines and RL inference).

**Structure**:
```
fusion/modules/rl/policies/
├── __init__.py
├── README.md
├── base.py                  # PathPolicy interface
├── ksp_ff_policy.py        # Baseline KSP-FF policy
├── one_plus_one_policy.py  # Baseline 1+1 policy
├── bc_policy.py            # Behavior Cloning policy
├── iql_policy.py           # IQL (Implicit Q-Learning) policy
├── action_masking.py       # Action masking utilities
└── tests/
    ├── __init__.py
    ├── README.md
    ├── test_bc_policy.py
    ├── test_iql_policy.py
    └── test_action_masking.py
```

**Key Classes**:
- `PathPolicy`: Abstract interface for path selection
- `KSPFFPolicy`, `OnePlusOnePolicy`: Baseline policies
- `BCPolicy`, `IQLPolicy`: RL policies (offline inference)
- `compute_action_mask()`, `apply_fallback_policy()`: Masking utilities

**Dependencies**:
- PyTorch (model inference)
- `fusion/interfaces/router`
- `fusion/modules/routing` (K-path cache)

**Estimated Effort**: 1.5 days (policies) + 0.5 days (masking) = 2 days

**Details**: See [30-rl-policies.md](../phase4-rl-integration/30-rl-policies.md)

---

### 3. Dataset Logger

**Location**: `fusion/reporting/dataset_logger.py`

**Purpose**: Log offline RL training data to JSON Lines format.

**Key Class**:
- `DatasetLogger`: Logs (state, action, reward, next_state, action_mask, meta) tuples

**Schema** (JSONL):
```json
{
  "t": 12345,
  "seed": 42,
  "state": {...},
  "action": 0,
  "reward": 1.0,
  "next_state": null,
  "action_mask": [true, true, false, true],
  "meta": {...}
}
```

**Dependencies**:
- pathlib (file I/O)
- json (JSONL serialization)

**Estimated Effort**: 1 day

**Details**: See [31-dataset-logging.md](../phase4-rl-integration/31-dataset-logging.md)

---

## Modules to Extend

### 4. Routing Module

**Location**: `fusion/modules/routing/`

**New Files**:
- `one_plus_one_protection.py`: 1+1 disjoint protection routing
- `k_path_cache.py`: K-path pre-computation and feature extraction

**Extensions**:

#### `one_plus_one_protection.py`

**Key Class**: `OnePlusOneProtection(AbstractRouter)`
- Computes link-disjoint primary and backup paths
- Uses Suurballe's algorithm or two runs of Yen's K-shortest paths
- Stores backup in `sdn_props`

**Estimated Effort**: 1.5-2 days

**Details**: See [20-protection.md](../phase3-protection/20-protection.md)

---

#### `k_path_cache.py`

**Key Class**: `KPathCache`
- Pre-computes K shortest paths for all (src, dst) pairs
- Caches paths at simulation startup
- Extracts path features (hops, residual, fragmentation, failure mask)

**Estimated Effort**: 1 day

**Details**: See [11-k-path-cache.md](../phase2-infrastructure/11-k-path-cache.md)

---

### 5. Core Properties

**Location**: `fusion/core/properties.py`

**Extensions to `SDNProps`**:

```python
class SDNProps:
    # ... existing attributes ...

    # 1+1 Protection attributes
    self.protection_mode: str | None = None  # "none" or "1plus1"
    self.primary_path: list[int] | None = None
    self.backup_path: list[int] | None = None
    self.is_protected: bool = False
    self.active_path: str = "primary"  # "primary" or "backup"
    self.protection_switchover_ms: float = 50.0
    self.restoration_latency_ms: float = 100.0
```

**Estimated Effort**: 0.5 days (part of protection module)

**Details**: See [20-protection.md](../phase3-protection/20-protection.md)

---

### 6. SDN Controller

**Location**: `fusion/core/sdn_controller.py`

**Extensions**:
1. **K-path cache integration**:
   - Get K candidate paths for each request
   - Extract path features

2. **Action masking**:
   - Compute feasibility mask based on failures and spectrum

3. **Policy selection**:
   - Use policy to select path index
   - Apply fallback if all paths masked

4. **Failure checking**:
   - Check path feasibility via `failure_manager.is_path_feasible()`

**Estimated Effort**: 1-1.5 days (part of integration)

**Details**: See [30-rl-policies.md](../phase4-rl-integration/30-rl-policies.md), [10-failure-module.md](../phase2-infrastructure/10-failure-module.md)

---

### 7. Simulation Engine

**Location**: `fusion/core/simulation.py`

**Extensions**:
1. **FailureManager integration**:
   - Add `self.failure_manager: FailureManager | None`
   - Initialize after topology is loaded
   - Call `failure_manager.repair_failures(current_time)` in main loop

2. **DatasetLogger integration**:
   - Add `self.dataset_logger: DatasetLogger | None`
   - Log transitions after each routing decision

3. **K-path cache initialization**:
   - Add `self.k_path_cache: KPathCache | None`
   - Pre-compute paths after topology load

4. **Seed management**:
   - Call `seed_all_rngs(seed)` at startup

**Estimated Effort**: 1.5-2 days (integration across modules)

**Details**: See Phase 2-4 implementation docs

---

### 8. Statistics Module

**Location**: `fusion/reporting/statistics.py`

**Extensions to `SimulationStatistics`**:

```python
class SimulationStatistics:
    def __init__(self, engine_props: dict[str, Any]) -> None:
        # ... existing initialization ...

        # Survivability metrics
        self.recovery_times_ms: list[float] = []
        self.fragmentation_scores: list[float] = []
        self.decision_times_ms: list[float] = []
        self.failure_window_bp_list: list[float] = []

    def record_recovery_event(...) -> None:
        """Record recovery event details."""
        pass

    def compute_fragmentation_proxy(...) -> float:
        """Compute fragmentation score for a path."""
        pass

    def get_recovery_stats() -> dict[str, float]:
        """Get mean and P95 recovery times."""
        pass

    def to_csv_row() -> dict[str, Any]:
        """Export all metrics to CSV row."""
        pass
```

**Estimated Effort**: 1 day

**Details**: See [40-metrics-reporting.md](../phase5-metrics/40-metrics-reporting.md)

---

### 9. Configuration System

**Location**: `fusion/configs/`

**New Files**:
- `templates/survivability_experiment.ini`: Template configuration
- `schemas/survivability.json`: Schema validation

**New Sections**:
- `[failure_settings]`: Failure type, timing, targets
- `[protection_settings]`: Protection mode, latencies
- `[offline_rl_settings]`: Policy type, model paths, device
- `[dataset_logging]`: Enable logging, output path, epsilon
- `[recovery_timing]`: Switchover/restoration latencies
- `[reporting]`: CSV export, seed aggregation

**Estimated Effort**: 0.5 days

**Details**: See [12-configuration.md](../phase2-infrastructure/12-configuration.md)

---

## Module Dependencies

### Dependency Graph

```
fusion/modules/failures/
  ├── depends on: networkx, fusion/core/properties
  └── used by: fusion/core/sdn_controller, fusion/modules/routing/k_path_cache

fusion/modules/rl/policies/
  ├── depends on: torch, fusion/interfaces/router
  └── used by: fusion/core/sdn_controller

fusion/reporting/dataset_logger.py
  ├── depends on: pathlib, json
  └── used by: fusion/core/simulation

fusion/modules/routing/one_plus_one_protection.py
  ├── depends on: networkx, fusion/interfaces/router
  └── used by: fusion/core/sdn_controller

fusion/modules/routing/k_path_cache.py
  ├── depends on: networkx, fusion/modules/failures
  └── used by: fusion/core/sdn_controller, fusion/modules/rl/policies
```

### Implementation Order

Based on dependencies, implement in this order:

1. **Phase 2: Core Infrastructure** (can be parallelized)
   - Failures module (no dependencies on other survivability modules)
   - Configuration system (independent)
   - K-path cache (depends on failures module)
   - Determinism/seeds (independent)

2. **Phase 3: Protection** (depends on Phase 2)
   - 1+1 Protection routing (depends on failures for feasibility checks)
   - Recovery timing (depends on protection module)

3. **Phase 4: RL Integration** (depends on Phases 2-3)
   - RL policies (depends on K-path cache)
   - Dataset logging (depends on policies)

4. **Phase 5: Metrics** (depends on Phases 2-4)
   - Metrics & reporting (depends on all previous phases)

5. **Phases 6-7: Quality & Management** (ongoing)
   - Testing (throughout development)
   - Documentation (throughout development)

---

## Module Size Estimates

| Module | Files | LOC (approx) | Tests (LOC) | Total |
|--------|-------|--------------|-------------|-------|
| Failures | 5 main + 4 test | ~800 | ~600 | ~1400 |
| RL Policies | 6 main + 3 test | ~600 | ~400 | ~1000 |
| Dataset Logger | 1 main + 1 test | ~150 | ~100 | ~250 |
| 1+1 Protection | 1 main + 1 test | ~300 | ~200 | ~500 |
| K-Path Cache | 1 main + 1 test | ~350 | ~250 | ~600 |
| Config Extensions | 2 files | ~200 | ~50 | ~250 |
| Properties Extensions | In existing | ~50 | ~50 | ~100 |
| SDN Controller Ext. | In existing | ~150 | ~100 | ~250 |
| SimEngine Ext. | In existing | ~100 | ~100 | ~200 |
| Statistics Ext. | In existing | ~200 | ~150 | ~350 |
| **Total** | | **~2900** | **~2000** | **~4900** |

**Note**: All individual files adhere to < 500 lines limit. Some modules span multiple files.

---

## Testing Strategy by Module

| Module | Unit Tests | Integration Tests | Mocking Strategy |
|--------|------------|-------------------|------------------|
| **Failures** | Per failure type | End-to-end failure + recovery | Mock topology |
| **RL Policies** | Per policy class | Policy + masking integration | Mock models, mock K-paths |
| **Dataset Logger** | Write, close, format | Full sim with logging | Mock file system |
| **1+1 Protection** | Disjoint path finding | Protection + failure scenario | Mock topology |
| **K-Path Cache** | Feature extraction | Cache + policy integration | Mock topology, mock spectrum |
| **Config** | Schema validation | Load config + run sim | None (use real config) |
| **Statistics** | Metric computation | Full sim with metrics | Mock data |

---

## Summary Table

| Category | Count | Details |
|----------|-------|---------|
| **New Modules** | 3 | Failures, RL Policies, Dataset Logger |
| **Extended Modules** | 6 | Routing, Properties, SDN Controller, SimEngine, Statistics, Config |
| **New Files** | ~20 | Including tests |
| **Total LOC** | ~4900 | ~2900 main + ~2000 test |
| **Estimated Effort** | 13-17 days | See [60-work-breakdown.md](../phase7-management/60-work-breakdown.md) |

---

## Next Steps

After understanding the module structure:

1. **Review** [03-version-control.md](03-version-control.md) for Git workflow
2. **Proceed** to **Phase 2** to implement core infrastructure
3. **Refer** to detailed module docs as you implement

---

**Related Documents**:
- [00-overview.md](00-overview.md) (Project context)
- [01-scope-boundaries.md](01-scope-boundaries.md) (What's in/out of scope)
- [60-work-breakdown.md](../phase7-management/60-work-breakdown.md) (Detailed effort estimates)
