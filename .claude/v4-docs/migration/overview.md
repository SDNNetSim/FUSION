# Migration Overview

This document provides a high-level summary of the V4 architecture migration.

## Migration Phases

| Phase | Name | Description | Key Deliverables |
|-------|------|-------------|------------------|
| 1 | Core Domain Model | Introduce foundational dataclasses | `SimulationConfig`, `Request`, `Lightpath`, result types |
| 2 | NetworkState | Single source of truth for network state | `NetworkState`, pipeline protocols, legacy adapters |
| 3 | Orchestrator | Pipeline coordination layer | `SDNOrchestrator`, `PipelineFactory`, feature flag |
| 4 | RL Integration | Unified RL environment | `RLSimulationAdapter`, `UnifiedSimEnv` |
| 5 | ML + Protection | ML policies and 1+1 protection | `ControlPolicy`, `ProtectionPipeline` |
| 6 | Legacy Removal | Clean up deprecated code | Remove legacy properties, old SDNController |

## Validation Stack

Each phase must pass the complete validation stack before proceeding:

```bash
# 1. Unit tests for new code
pytest fusion/tests/<phase_module>/ -v

# 2. Linting
ruff check fusion/<phase_module>/

# 3. Type checking
mypy fusion/<phase_module>/

# 4. Documentation build
cd docs && make html

# 5. Regression tests (existing behavior unchanged)
pytest tests/ -v

# 6. Comparison validation (Phase 3+)
python tests/run_comparison.py
```

## Phase Dependencies

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
              │            │           │
              │            │           ▼
              │            │       Phase 5
              │            │           │
              │            ▼           ▼
              │       Phase 6 (after 4+5)
              │
              └──► Adapters wrap existing code
```

## Key Principles

### 1. Additive First

Each phase adds new code without modifying existing behavior. Legacy code continues to work.

### 2. Feature Flag Transition

Phase 3 introduces a feature flag (`use_orchestrator`) allowing both old and new paths to run in parallel.

### 3. Verification Before Proceeding

Exit criteria must be met before starting the next phase. `run_comparison.py` is the ultimate arbiter.

### 4. Legacy Removal Last

Legacy code is only removed in Phase 6, after all functionality is migrated and verified.

## File Layout After Migration

```
fusion/
├── domain/                    # Phase 1
│   ├── config.py
│   ├── request.py
│   ├── lightpath.py
│   ├── network_state.py       # Phase 2
│   └── results.py
├── interfaces/                # Phase 2
│   ├── pipelines.py
│   └── routing_strategy.py
├── pipelines/                 # Phase 3
│   ├── routing_pipeline.py
│   ├── spectrum_pipeline.py
│   ├── grooming_pipeline.py
│   ├── snr_pipeline.py
│   └── slicing_pipeline.py
├── core/
│   ├── simulation.py          # Modified Phase 3
│   ├── orchestrator.py        # Phase 3
│   └── pipeline_factory.py    # Phase 3
├── stats/                     # Phase 1
│   └── collector.py
├── modules/
│   └── rl/
│       ├── env_adapter.py     # Phase 4
│       └── gymnasium_envs/
│           └── unified_sim_env.py  # Phase 4
├── policies/                  # Phase 5
│   ├── heuristic_policy.py
│   └── ml_policy.py
└── legacy/                    # Phase 2, removed Phase 6
    └── adapters/
```

## Micro-Phase Structure

Each phase is broken into micro-phases (e.g., P1.1, P1.2) that:

1. Create specific files
2. Have clear verification steps
3. Can be completed in a single work session
4. Produce testable artifacts

See individual phase documents for micro-phase details:

- [Phase 1: Core Domain Model](phase_1_core_model.md)
- [Phase 2: NetworkState](phase_2_network_state.md) (TODO)
- [Phase 3: Orchestrator](phase_3_orchestrator.md) (TODO)
- [Phase 4: RL Integration](phase_4_rl_integration.md) (TODO)
- [Phase 5: ML + Protection](phase_5_ml_protection.md) (TODO)
- [Phase 6: Legacy Removal](phase_6_legacy_removal.md) (TODO)
