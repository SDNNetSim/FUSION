# Phase 5: ML Control + Protection Integration

**Version:** v5-final-docs (revised)
**Scope:** Phase 5 (P5.1 through P5.6)
**Prerequisite:** Phase 4 complete (RL integration verified)

---

## Overview

Phase 5 introduces a unified control policy abstraction and 1+1 dedicated path protection into the FUSION architecture. This phase enables flexible decision-making strategies (heuristic, ML, RL) under a common interface while adding production-grade network survivability capabilities.

### Core Principles

1. **Protocol-Based Design**: `ControlPolicy` uses Python's structural typing (`Protocol`) for flexibility
2. **Selection, Not Routing**: Policies choose among pre-computed `PathOption` objects; they do not compute routes
3. **Protection as Strategy**: 1+1 protection is a specialized routing strategy, not a separate pipeline stage
4. **Backwards Compatible**: Existing behavior preserved with safe defaults (heuristic policy, protection disabled)
5. **RL Unchanged**: Existing RL code continues to work through the RLPolicy adapter wrapper
6. **Deployment Focus**: ML policies deploy pre-trained models; online training is handled by RL (Phase 4)

---

## Goals

1. **ControlPolicy Protocol**: Unified interface for heuristics, RL, and ML path selection
2. **Heuristic Policies**: Baseline policies (first-fit, shortest, least-congested, random, load-balanced, modulation-aware)
3. **RL Policy Adapter**: Wrapper enabling existing SB3 models to implement ControlPolicy
4. **ML Policy Support**: Pre-trained model deployment with PyTorch, ONNX, and sklearn support
5. **Composite Policies**: FallbackPolicy, TiebreakingPolicy for policy composition
6. **Policy Registry**: Extensible registry pattern for custom policies
7. **Protection Pipeline**: 1+1 dedicated path protection with disjoint paths and failure switchover
8. **Domain Model Extensions**: Lightpath, RouteResult, AllocationResult protection fields
9. **NetworkState Extensions**: Protected lightpath creation and link queries
10. **Orchestrator Integration**: Wire policies and protection into SDNOrchestrator
11. **Configuration Extension**: Add [policy], [protection], [heuristic], [reward] INI sections

---

## Sub-Phases

| Sub-Phase | Description | Primary Deliverables |
|-----------|-------------|----------------------|
| [P5.1](P5.1_control_policy_protocol/P5.1.index.md) | ControlPolicy Protocol + RLPolicy | `fusion/interfaces/control_policy.py`, `fusion/policies/rl_policy.py` |
| [P5.2](P5.2_heuristic_policies/P5.2.index.md) | Heuristic + Composite Policies | `fusion/policies/heuristic_policy.py`, `fusion/policies/composite_policy.py` |
| [P5.3](P5.3_ml_policy_support/P5.3.index.md) | ML Policy Support | `fusion/policies/ml_policy.py` |
| [P5.4](P5.4_protection_pipeline/P5.4.index.md) | Protection Pipeline | `fusion/pipelines/protection_pipeline.py`, domain extensions |
| [P5.5](P5.5_orchestrator_integration/P5.5.index.md) | Orchestrator Integration | `fusion/core/orchestrator.py` updates |
| [P5.6](P5.6_configuration_extension/P5.6.index.md) | Configuration Extension | Config additions |

---

## Dependencies Between Sub-Phases

```
P5.1 (ControlPolicy + RLPolicy) ----+
                                    |
P5.2 (Heuristics + Composites) -----+---> P5.5 (Orchestrator) ---> P5.6 (Config)
                                    |
P5.3 (ML Policy) -------------------+
                                    |
P5.4 (Protection + Domain) ---------+
```

- **P5.1** must complete before P5.2, P5.3, P5.4
- **P5.2, P5.3, P5.4** can run in parallel after P5.1
- **P5.5** requires P5.1-P5.4 complete
- **P5.6** requires P5.5 complete

---

## Key Interfaces

### ControlPolicy Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ControlPolicy(Protocol):
    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select action (path index 0 to k-1) or -1 for invalid."""
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """Update policy based on experience (no-op for heuristics)."""
        ...

    def get_name(self) -> str:
        """Return policy name for logging and metrics."""
        ...
```

### PathOption (Extended for Protection)

```python
@dataclass(frozen=True)
class PathOption:
    path_index: int
    path: tuple[str, ...]
    weight_km: float
    is_feasible: bool
    modulation: str | None
    slots_needed: int | None
    congestion: float
    # Protection fields (Phase 5)
    backup_path: tuple[str, ...] | None = None
    backup_feasible: bool | None = None
    backup_weight_km: float | None = None
    backup_modulation: str | None = None
    is_protected: bool = False

    @property
    def both_paths_feasible(self) -> bool:
        """True if both primary and backup paths are feasible."""
        if not self.is_protected:
            return self.is_feasible
        return self.is_feasible and (self.backup_feasible or False)
```

### Policy Types Summary

| Category | Policy | Selection Criterion |
|----------|--------|---------------------|
| **Heuristic** | `FirstFeasiblePolicy` | First path with `is_feasible=True` |
| **Heuristic** | `ShortestFeasiblePolicy` | Feasible path with minimum `weight_km` |
| **Heuristic** | `LeastCongestedPolicy` | Feasible path with minimum `congestion` |
| **Heuristic** | `RandomFeasiblePolicy` | Random selection among feasible paths |
| **Heuristic** | `LoadBalancedPolicy` | Weighted combination of length and congestion |
| **Heuristic** | `ModulationAwarePolicy` | Feasible path with highest modulation efficiency |
| **Composite** | `FallbackPolicy` | Chain policies, use first valid action |
| **Composite** | `TiebreakingPolicy` | Primary policy, secondary for ties |
| **RL** | `RLPolicy` | Wraps SB3 model, builds observation, applies mask |
| **ML** | `MLControlPolicy` | Pre-trained NN/classifier with fallback |

---

## Configuration Sections

```ini
[policy]
type = first_feasible           # first_feasible, shortest_feasible, least_congested,
                                # random, load_balanced, modulation_aware, ml, rl
model_path =                    # Path to model for ML/RL policy
device = cpu                    # cpu or cuda
fallback_type = first_feasible  # Fallback policy for ML errors

[protection]
enabled = false                 # Enable 1+1 protection
disjointness = link             # link or node disjoint
switchover_time_ms = 50         # Protection switchover latency

[heuristic]
load_balance_alpha = 0.5        # Weight for load_balanced (0=length, 1=congestion)
random_seed = 42                # Seed for random_feasible policy

[reward]
success = 1.0                   # Reward for successful allocation
failure = -1.0                  # Penalty for blocked request
efficiency_bonus = 0.1          # Bonus multiplier for efficient allocations
```

---

## New File Locations

```
fusion/
  interfaces/
    control_policy.py           # ControlPolicy protocol
  policies/
    __init__.py                 # Package exports
    heuristic_policy.py         # HeuristicPolicy base + implementations
    composite_policy.py         # FallbackPolicy, TiebreakingPolicy
    ml_policy.py                # MLControlPolicy
    rl_policy.py                # RLPolicy wrapper for SB3 models
    policy_factory.py           # PolicyFactory
    registry.py                 # Policy registry
  pipelines/
    protection_pipeline.py      # ProtectionPipeline
    disjoint_path_finder.py     # DisjointPathFinder helper
  domain/
    # Extensions to existing files:
    lightpath.py                # Add protection fields
    results.py                  # Add protection fields to RouteResult, AllocationResult
    network_state.py            # Add create_protected_lightpath, get_lightpaths_on_link
  tests/
    interfaces/
      test_control_policy.py
    policies/
      test_heuristic_policy.py
      test_composite_policy.py
      test_ml_policy.py
      test_rl_policy.py
      test_policy_factory.py
    pipelines/
      test_protection_pipeline.py
```

---

## Domain Model Extensions

### Lightpath (protection fields)

```python
@dataclass
class Lightpath:
    # Existing fields...

    # Protection fields (Phase 5)
    backup_path: list[str] | None = None
    backup_spectrum_start: int | None = None
    backup_spectrum_end: int | None = None
    is_protected: bool = False
    active_path: str = "primary"  # "primary" or "backup"

    @property
    def current_path(self) -> list[str]:
        """Return currently active path."""
        if self.is_protected and self.active_path == "backup":
            return self.backup_path or self.path
        return self.path
```

### RouteResult (protection fields)

```python
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]
    modulations: list[list[str | None]]
    weights_km: list[float]
    # Protection fields (Phase 5)
    backup_paths: list[list[str]] | None = None
    backup_modulations: list[list[str | None]] | None = None
    backup_weights_km: list[float] | None = None
    strategy_name: str = ""
```

### AllocationResult (protection fields)

```python
@dataclass(frozen=True)
class AllocationResult:
    success: bool
    lightpath_id: str | None = None
    path: list[str] | None = None
    spectrum_start: int | None = None
    spectrum_end: int | None = None
    modulation: str | None = None
    block_reason: BlockReason | None = None
    # Protection fields (Phase 5)
    backup_path: list[str] | None = None
    is_protected: bool = False
```

---

## Quality Requirements

| Module | Target Coverage |
|--------|-----------------|
| `interfaces/control_policy.py` | 100% |
| `policies/heuristic_policy.py` | 95% |
| `policies/composite_policy.py` | 95% |
| `policies/ml_policy.py` | 90% |
| `policies/rl_policy.py` | 90% |
| `pipelines/protection_pipeline.py` | 90% |

---

## Exit Criteria

- [ ] ControlPolicy protocol defined with select_action, update, get_name
- [ ] PathOption extended with backup fields and both_paths_feasible property
- [ ] All 6 heuristic policies implemented and tested
- [ ] RLPolicy wraps SB3 models through ControlPolicy interface
- [ ] MLControlPolicy supports PyTorch, sklearn, ONNX with fallback
- [ ] FallbackPolicy and TiebreakingPolicy implemented
- [ ] Policy registry with registration capability
- [ ] ProtectionPipeline finds disjoint paths correctly
- [ ] DisjointPathFinder supports link and node disjoint modes
- [ ] Protected spectrum allocation on both paths
- [ ] Failure switchover mechanism functional
- [ ] Lightpath, RouteResult, AllocationResult protection extensions complete
- [ ] NetworkState.create_protected_lightpath and get_lightpaths_on_link implemented
- [ ] PolicyFactory creates all policy types from config
- [ ] SDNOrchestrator integrates with policies
- [ ] Configuration sections documented with defaults
- [ ] All tests pass with target coverage
- [ ] No regressions in Phase 4 RL functionality

---

## What Phase 5 Does NOT Include

Phase 5 is strictly scoped. The following are explicitly **out of scope**:

1. **New routing algorithms** - Policies select among existing routes, not compute new ones
2. **Online ML training** - ML policies deploy pre-trained models; training is external
3. **Multi-failure scenarios** - Protection handles single failures; cascading failures are future work
4. **Restoration (dynamic re-routing)** - Only 1+1 protection; restoration is Phase 6+
5. **Shared protection (1:1)** - Only 1+1 dedicated protection implemented
6. **Legacy code removal** - Phase 6 handles deprecation and cleanup
7. **GUI changes** - The `fusion/gui` module is deprecated and not modified

---

## Reference Documents

### V4 Architecture Docs
- `.claude/v4-docs/architecture/control_policy.md` - ControlPolicy specification
- `.claude/v4-docs/architecture/ml_policies.md` - ML policy design
- `.claude/v4-docs/architecture/heuristic_policies.md` - Heuristic policies
- `.claude/v4-docs/architecture/protection_pipeline.md` - Protection architecture

### V4 Decision Records
- `.claude/v4-docs/decisions/0010-control-policy-protocol.md` - Protocol rationale
- `.claude/v4-docs/decisions/0011-ml-control-policy.md` - ML policy design
- `.claude/v4-docs/decisions/0012-protection-pipeline.md` - Protection design

### V4 Migration
- `.claude/v4-docs/migration/phase_5_ml_protection.md` - Phase 5 objectives
- `.claude/v4-docs/migration/phase_5_checklist.md` - Implementation checklist

### Earlier V5 Phases
- `.claude/v5-final-docs/phase-4-rl-integration/` - RLSimulationAdapter, PathOption
