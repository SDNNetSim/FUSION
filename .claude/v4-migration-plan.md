# FUSION V4 Architecture Migration Plan

## Overview

This plan creates a **documentation-first, micro-phase migration** from V3, structured as multiple markdown files. The output will be a complete directory tree under `.claude/v4-docs/` for reference, with selected docs promoted to `docs/` as the migration progresses.

## Key Decisions

Based on user input:
1. **Tests**: Consolidate under `fusion/tests/` (scalable for open source)
2. **RL Env**: Create new `UnifiedSimEnv` (avoid breaking existing workflows)
3. **Docs Output**: Place working docs in `.claude/v4-docs/`
4. **Legacy**: Delete when no longer needed (git tracks history)

## Deliverables

The following files will be created in the `.claude/v4-docs/` directory:

```
.claude/v4-docs/
├── architecture/
│   ├── overview.md                 # High-level architecture diagram
│   ├── domain_model.md             # SimulationConfig, Request, Lightpath, NetworkState
│   ├── routing_strategies.md       # Strategy pattern + RouteResult spec
│   ├── result_objects.md           # All result types: RouteResult, SpectrumResult, etc.
│   ├── network_state.md            # NetworkState single source of truth
│   ├── pipelines.md                # Pipeline protocols and implementations
│   ├── orchestration.md            # SDNOrchestrator design rules
│   ├── stats_and_metrics.md        # StatsCollector architecture
│   └── configuration.md            # SimulationConfig and feature flags
├── migration/
│   ├── overview.md                 # Phase summary and validation stack
│   ├── phase_1_core_model.md       # P1.1-P1.5 micro-phases
│   ├── phase_2_network_state.md    # P2.1-P2.4 micro-phases
│   ├── phase_3_orchestrator.md     # P3.1-P3.5 micro-phases
│   ├── phase_4_rl_integration.md   # P4.1-P4.4 micro-phases
│   ├── phase_5_ml_protection.md    # P5.1-P5.3 micro-phases
│   └── phase_6_legacy_removal.md   # P6.1-P6.3 micro-phases
├── tutorials/
│   ├── getting_started.md          # Quick start guide
│   ├── writing_a_new_pipeline.md   # Pipeline development tutorial
│   └── adding_a_routing_strategy.md # Strategy development tutorial
├── decisions/
│   ├── 0001-architecture-overview.md
│   ├── 0002-networkstate-single-source.md
│   ├── 0003-result-objects.md
│   └── 0004-routing-strategy-pattern.md
└── testing/
    └── test_strategy.md            # Test evolution strategy
```

---

## Critical Files to Reference/Modify

### Existing Files (Read-Only Reference)
- `fusion/core/metrics.py` - Current `SimStats` class (1459 lines)
- `fusion/core/properties.py` - `StatsProps`, `SDNProps` classes
- `fusion/modules/routing/registry.py` - Existing routing registry pattern
- `fusion/modules/routing/one_plus_one_protection.py` - 1+1 protection impl
- `fusion/modules/rl/gymnasium_envs/general_sim_env.py` - Current RL env
- `tests/run_comparison.py` - Comparison test framework
- `docs/conf.py` - Sphinx config (MyST enabled)
- `.github/workflows/quality.yml` - CI checks (ruff, mypy, pytest, bandit)

### New Files to Create (Phase 1-6)

**Phase 1: Domain Model**
- `fusion/domain/__init__.py`
- `fusion/domain/config.py` - SimulationConfig
- `fusion/domain/request.py` - Request, RequestStatus
- `fusion/domain/lightpath.py` - Lightpath
- `fusion/domain/results.py` - All result types

**Phase 2: NetworkState**
- `fusion/domain/network_state.py`
- `fusion/interfaces/__init__.py`
- `fusion/interfaces/pipelines.py`
- `fusion/interfaces/routing_strategy.py`
- `fusion/legacy/adapters/routing_adapter.py`

**Phase 3: Orchestrator**
- `fusion/core/orchestrator.py` - SDNOrchestrator
- `fusion/core/pipeline_factory.py` - PipelineFactory, PipelineSet
- `fusion/pipelines/*.py` - Pipeline implementations
- `fusion/stats/collector.py` - StatsCollector

**Phase 4-6**: RL adapter, ML policies, legacy removal

---

## Section 1: Routing Strategies and RouteResult

### RoutingStrategy Protocol

The existing `AbstractRoutingAlgorithm` in `fusion/interfaces/router.py` will be extended with a `RoutingStrategy` protocol that:

1. Uses the existing registry pattern from `fusion/modules/routing/registry.py`
2. Returns a standardized `RouteResult` instead of modifying `sdn_props`
3. Supports multiple strategies: KSP, load-balanced, 1+1 protection, etc.

### RouteResult Specification

```python
@dataclass(frozen=True)
class RouteResult:
    # Primary paths
    paths: list[list[str]]           # [[node1, node2, ...], ...]
    weights_km: list[float]          # Path lengths
    modulations: list[list[str | None]]  # Valid mods per path

    # Backup paths (for protection)
    backup_paths: list[list[str]] | None = None
    backup_weights_km: list[float] | None = None
    backup_modulations: list[list[str | None]] | None = None

    # Metadata
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Invariants:
    # - len(paths) == len(weights_km) == len(modulations)
    # - If backup_paths: len(backup_paths) == len(paths)
```

### Integration with Existing Code

The existing `OnePlusOneProtection.route()` method will be adapted to return `RouteResult` instead of modifying `sdn_props.routes_matrix` etc.

---

## Section 2: Result Objects

### Complete Result Type List

| Type | Created By | Consumed By | Purpose |
|------|-----------|-------------|---------|
| `RouteResult` | RoutingPipeline | Orchestrator, SpectrumPipeline | Candidate paths |
| `SpectrumResult` | SpectrumPipeline | Orchestrator, NetworkState | Slot assignments |
| `GroomingResult` | GroomingPipeline | Orchestrator | Grooming outcome |
| `SlicingResult` | SlicingPipeline | Orchestrator | Slicing outcome |
| `SNRResult` | SNRPipeline | Orchestrator | SNR validation |
| `AllocationResult` | Orchestrator | Engine, StatsCollector | Final outcome |

### Feasibility Authority

- `RouteResult.is_empty` - No route found
- `SpectrumResult.is_free` - Spectrum available
- `SNRResult.passed` - SNR threshold met
- `AllocationResult.success` - **Final authority** (orchestrator combines all)

---

## Section 3: Stats Architecture

### Current State

- `SimStats` class in `fusion/core/metrics.py` (1459 lines)
- `StatsProps` in `fusion/core/properties.py` - holds all counters
- Stats updated from multiple places (engine, metrics, helpers)

### New StatsCollector Design

```python
@dataclass
class StatsCollector:
    config: SimulationConfig

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    blocked_requests: int = 0

    # Blocking breakdown
    block_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Feature tracking
    groomed_requests: int = 0
    sliced_requests: int = 0
    protected_requests: int = 0

    # SNR tracking
    snr_values: list[float] = field(default_factory=list)

    def record_arrival(self, request: Request, result: AllocationResult, network_state: NetworkState) -> None:
        """Single entry point for stats update."""
        ...

    def to_comparison_format(self) -> dict:
        """Export in format expected by run_comparison.py."""
        ...
```

### Legacy Mapping

| Legacy (`StatsProps`) | New (`StatsCollector`) | Updated By |
|-----------------------|------------------------|------------|
| `simulation_blocking_list` | `blocking_probability` (computed) | `record_arrival()` |
| `block_reasons_dict` | `block_reasons` | `record_arrival()` |
| `modulations_used_dict` | `requests_per_modulation` | `record_arrival()` |
| `snr_list` | `snr_values` | `record_snr_result()` |

---

## Section 4: Final File Layout

```
fusion/
├── domain/                    # NEW: Core domain objects
│   ├── config.py              # SimulationConfig
│   ├── request.py             # Request, RequestStatus
│   ├── lightpath.py           # Lightpath
│   ├── network_state.py       # NetworkState (Phase 2)
│   └── results.py             # All result types
├── interfaces/                # NEW: Protocols
│   ├── pipelines.py           # Pipeline protocols
│   └── routing_strategy.py    # RoutingStrategy protocol
├── routing/                   # NEW: Strategy implementations
│   └── strategies/            # KSP, disjoint, etc.
├── pipelines/                 # NEW: Pipeline implementations
│   ├── routing_pipeline.py
│   ├── spectrum_pipeline.py
│   └── ...
├── core/
│   ├── simulation.py          # Engine (modified Phase 3)
│   ├── orchestrator.py        # NEW: SDNOrchestrator
│   └── pipeline_factory.py    # NEW: PipelineFactory
├── stats/                     # NEW: Statistics
│   └── collector.py           # StatsCollector
├── modules/
│   ├── routing/               # EXISTING: Adapters wrap these
│   ├── rl/                    # MODIFIED Phase 4
│   └── failures/              # EXISTING: Protection support
├── legacy/                    # TEMPORARY: Adapters during migration
│   └── adapters/              # Phase 2 adapters, deleted in Phase 6
└── tests/                     # Mirrors new structure
    ├── domain/
    ├── pipelines/
    └── integration/
```

---

## Section 5: Micro-Phase Plan

### Phase 1: Core Domain Model

| ID | Name | Files Changed | Files Created | Docs Updated |
|----|------|---------------|---------------|--------------|
| P1.1 | Domain Scaffolding | - | `fusion/domain/__init__.py`, `config.py` | `architecture/domain_model.md` |
| P1.2 | Request Wrapper | - | `fusion/domain/request.py` | `architecture/domain_model.md` |
| P1.3 | Lightpath Wrapper | - | `fusion/domain/lightpath.py` | `architecture/domain_model.md` |
| P1.4 | Result Objects | - | `fusion/domain/results.py` | `architecture/result_objects.md` |
| P1.5 | StatsCollector Skeleton | - | `fusion/stats/__init__.py`, `collector.py` | `architecture/stats_and_metrics.md` |

**Verification per micro-phase:**
```bash
pytest fusion/tests/domain/ -v
ruff check fusion/domain/ fusion/stats/
mypy fusion/domain/ fusion/stats/
cd docs && make html
```

### Phase 2: NetworkState

| ID | Name | Files Changed | Files Created |
|----|------|---------------|---------------|
| P2.1 | NetworkState Core | - | `fusion/domain/network_state.py` |
| P2.2 | Legacy Compat Props | `network_state.py` | - |
| P2.3 | Pipeline Protocols | - | `fusion/interfaces/pipelines.py`, `routing_strategy.py` |
| P2.4 | Legacy Adapters | - | `fusion/legacy/adapters/*.py` |

### Phase 3: Orchestrator

| ID | Name | Files Changed | Files Created |
|----|------|---------------|---------------|
| P3.1 | PipelineFactory | - | `fusion/core/pipeline_factory.py` |
| P3.2 | SDNOrchestrator Core | - | `fusion/core/orchestrator.py` |
| P3.3 | Feature Flag | `fusion/core/simulation.py` | - |
| P3.4 | StatsCollector Integration | `simulation.py`, `collector.py` | - |
| P3.5 | run_comparison Verification | `tests/run_comparison.py` (if needed) | - |

### Phases 4-6: RL, ML, Legacy Removal

(Similar structure with micro-phases)

---

## Section 6: Test Strategy

### Test Layout

```
fusion/tests/
├── domain/
│   ├── test_config.py
│   ├── test_request.py
│   ├── test_lightpath.py
│   ├── test_network_state.py
│   └── test_results.py
├── routing/strategies/
│   ├── test_ksp.py
│   └── test_disjoint.py
├── pipelines/
│   ├── test_routing_pipeline.py
│   └── test_spectrum_pipeline.py
├── core/
│   ├── test_orchestrator.py
│   └── test_pipeline_factory.py
├── stats/
│   └── test_collector.py
└── integration/
    ├── test_full_simulation.py
    └── test_comparison.py
```

### Evolution Strategy

1. **Phase 1**: Existing tests unchanged; add `tests/domain/`
2. **Phase 2**: Add `tests/routing/strategies/`; existing routing tests serve as regression
3. **Phase 3**: Add `tests/core/test_orchestrator.py`; run_comparison validates both paths
4. **Phase 6**: Remove tests for deprecated code

---

## Section 7: Documentation Rules

1. **No pipeline without docs**: Entry in `docs/architecture/pipelines.md` + tutorial
2. **SDNOrchestrator changes require ADR**: If behavior changes, create ADR
3. **Feature flags must be documented**: In `docs/architecture/configuration.md`
4. **New strategies follow template**: `docs/tutorials/adding_a_routing_strategy.md`

---

## Section 8: Resolved Decisions

All key decisions have been resolved:

1. **Tests**: Consolidate under `fusion/tests/` (scalable for open source)
2. **RL environment**: Create new `UnifiedSimEnv` (avoid breaking existing workflows)
3. **Legacy code**: Delete when no longer needed, use git for history
4. **Docs**: Working docs in `.claude/v4-docs/`, promote to `docs/` as migration progresses
5. **run_comparison.py**: Keep existing `abs_tol=0.02` tolerance during migration

---

## Implementation Order

1. **First**: Create docs structure (docs/architecture/, docs/migration/, etc.)
2. **Second**: P1.1-P1.5 (domain objects)
3. **Third**: P2.1-P2.4 (NetworkState + adapters)
4. **Fourth**: P3.1-P3.5 (Orchestrator + feature flag)
5. **Fifth**: Validate with run_comparison.py
6. **Then**: P4-P6 (RL, ML, cleanup)

Each micro-phase follows: **Code -> Tests -> Docs -> Verify**
