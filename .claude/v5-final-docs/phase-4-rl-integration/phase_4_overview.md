# Phase 4: RL Integration Migration - Overview

**Version:** v5.1 (Gap Analysis Complete)
**Status:** Ready for Implementation
**Dependencies:** Phase 1 (Core Domain), Phase 2 (State Management), Phase 3 (Orchestrator)
**Last Updated:** 2024 (Gap Analysis)

## Executive Summary

Phase 4 integrates reinforcement learning (RL) agents with the V4 simulation stack, eliminating the duplicated simulation logic in the legacy `GeneralSimEnv`. The new `UnifiedSimEnv` uses the same pipelines and orchestrator as non-RL simulation, ensuring consistent behavior and easier maintenance.

**Gap Analysis Status:** Complete. See `P4.0_gap_analysis.md` for identified gaps and resolutions.

## Core Principle: No Forked Simulator

The legacy RL environment (`GeneralSimEnv`) duplicates simulation logic via `mock_handle_arrival()`, which can diverge from the actual simulation. Phase 4 eliminates this duplication by:

1. Creating `RLSimulationAdapter` - a thin coordination layer
2. Routing all operations through `SDNOrchestrator`
3. Using the same `RoutingPipeline` and `SpectrumPipeline` instances
4. Ensuring action feasibility comes from real spectrum checks

## Supported RL Paths

Phase 4 covers migration for all existing RL paths in the codebase:

| Category | Algorithms | Migration Guide |
|----------|------------|-----------------|
| SB3 Online | PPO, A2C, DQN, QR-DQN | P4.3.e |
| Tabular | Q-Learning, Bandits | P4.3.e |
| Offline RL | BC, IQL | P4.1.f, P4.3.e |
| GNN-based | Graphormer, PathGNN | P4.2.f |

## Sub-Phases

### P4.1: RLSimulationAdapter Scaffolding

**Goal:** Create the adapter layer between RL environments and simulation pipelines.

**Key Deliverables:**
- `fusion/rl/adapter.py` - `RLSimulationAdapter` class
- `PathOption` dataclass for candidate paths
- Methods: `get_path_options()`, `apply_action()`, `compute_reward()`

**Critical Invariant:** Adapter uses SAME pipeline instances as orchestrator.

**Files:** 8 (1 index, 1 shared context, 6 micro-tasks including P4.1.f gap-filling)

### P4.2: UnifiedSimEnv Wiring

**Goal:** Create Gymnasium-compatible environment using V4 stack.

**Key Deliverables:**
- `fusion/rl/environments/unified_env.py` - `UnifiedSimEnv` class
- `fusion/rl/environments/wrappers.py` - `ActionMaskWrapper`
- Full Gymnasium compliance

**Critical Feature:** Action mask in `info["action_mask"]` for SB3 MaskablePPO.

**Files:** 8 (1 index, 1 shared context, 6 micro-tasks including P4.2.f gap-filling)

### P4.3: Migrate Existing RL Experiments

**Goal:** Enable gradual migration from legacy to unified environment.

**Key Deliverables:**
- Factory function `create_sim_env(config, env_type)`
- Feature flags (`USE_UNIFIED_ENV` env var)
- Deprecation warnings on `GeneralSimEnv`

**Critical Constraint:** Existing experiments must work unchanged.

**Files:** 7 (1 index, 1 shared context, 5 micro-tasks including P4.3.e gap-filling)

### P4.4: Parity Validation & Differences Documentation

**Goal:** Verify unified env produces comparable results; document differences.

**Key Deliverables:**
- Parity test suite (`test_rl_parity.py`)
- Comparison script (`compare_rl_envs.py`)
- Differences documentation (`rl_differences.md`)
- Rollback procedures

**Critical Metric:** Blocking probability within 5% of legacy.

**Files:** 7 (1 index, 1 shared context, 5 micro-tasks including P4.4.e gap-filling)

## File Inventory

```
.claude/v5-final-docs/phase-4-rl-integration/
├── phase_4_overview.md                          # This file
├── P4.0_gap_analysis.md                         # Gap analysis (NEW)
├── P4.1_rl_adapter/
│   ├── P4.1.index.md
│   ├── P4.1.shared_context_legacy_rl_paths.md
│   ├── P4.1.a_context_extraction_legacy_rl.md
│   ├── P4.1.b_design_pathoption.md
│   ├── P4.1.c_design_adapter_api.md
│   ├── P4.1.d_wiring_plan_pipelines.md
│   ├── P4.1.e_verification_plan.md
│   └── P4.1.f_offline_rl_compatibility.md       # Gap-filling (NEW)
├── P4.2_unified_env/
│   ├── P4.2.index.md
│   ├── P4.2.shared_context_env_requirements.md
│   ├── P4.2.a_context_extraction_general_sim_env.md
│   ├── P4.2.b_design_unified_env.md
│   ├── P4.2.c_wiring_plan_sim_engine.md
│   ├── P4.2.d_action_masking_design.md
│   ├── P4.2.e_verification_plan.md
│   └── P4.2.f_graph_observation_support.md      # Gap-filling (NEW)
├── P4.3_migrate_experiments/
│   ├── P4.3.index.md
│   ├── P4.3.shared_context_training_scripts.md
│   ├── P4.3.a_context_extraction_experiments.md
│   ├── P4.3.b_migration_plan_factory.md
│   ├── P4.3.c_deprecation_plan.md
│   ├── P4.3.d_verification_plan.md
│   └── P4.3.e_algorithm_migration_guide.md      # Gap-filling (NEW)
└── P4.4_parity_and_differences/
    ├── P4.4.index.md
    ├── P4.4.shared_context_parity_metrics.md
    ├── P4.4.a_design_parity_tests.md
    ├── P4.4.b_verification_plan_parity.md
    ├── P4.4.c_documentation_differences.md
    ├── P4.4.d_rollback_plan.md
    └── P4.4.e_per_algorithm_differences.md      # Gap-filling (NEW)
```

**Total Files:** 32 (4 index, 4 shared context, 22 micro-tasks, 1 overview, 1 gap analysis)

## Implementation Order

1. **P4.1.a-f** → Complete adapter, PathOption, and offline RL compatibility
2. **P4.2.a-f** → Complete UnifiedSimEnv with graph observation support
3. **P4.3.a-e** → Complete migration infrastructure with algorithm guides
4. **P4.4.a-e** → Complete validation and per-algorithm differences documentation

Each micro-task can be executed independently with its specified context files.

### Priority Order for Gap-Filling Tasks

High priority (blocks migration):
1. **P4.2.f** - Graph observation support (GNN users blocked)
2. **P4.1.f** - Offline RL compatibility (BC/IQL users blocked)
3. **P4.3.e** - Algorithm migration guide (all users need guidance)

Medium priority (improves experience):
4. **P4.4.e** - Per-algorithm differences (helps decision making)

## Key Code Locations (After Implementation)

```
fusion/
├── rl/
│   ├── __init__.py
│   ├── adapter.py              # RLSimulationAdapter, PathOption
│   ├── observation.py          # Observation building (optional)
│   ├── reward.py               # Reward computation (optional)
│   └── environments/
│       ├── __init__.py
│       ├── unified_env.py      # UnifiedSimEnv
│       └── wrappers.py         # ActionMaskWrapper
├── modules/rl/gymnasium_envs/
│   ├── __init__.py             # Factory function added
│   └── general_sim_env.py      # Deprecation warning added
└── tests/rl/
    ├── test_pathoption.py
    ├── test_adapter.py
    ├── test_unified_env.py
    ├── test_migration*.py
    └── test_rl_parity.py
```

## Dependencies on Earlier Phases

### From Phase 1 (Core Domain)
- `SimulationConfig` - configuration object
- `Request` - request dataclass
- `Lightpath` - lightpath dataclass
- Result types (`AllocationResult`, `RouteResult`, `SpectrumResult`)

### From Phase 2 (State Management)
- `NetworkState` - network state object with pipeline protocol
- Query methods: `get_link_utilization()`, `get_available_slots()`

### From Phase 3 (Orchestrator)
- `SDNOrchestrator` - orchestration with `handle_arrival()`
- `PipelineFactory` - pipeline creation
- `RoutingPipeline`, `SpectrumPipeline` - pipeline protocols
- `forced_path` parameter support in orchestrator

## Exit Criteria

### Core Requirements
- [ ] `RLSimulationAdapter` uses same pipeline instances as orchestrator
- [ ] `UnifiedSimEnv` passes `gymnasium.utils.env_checker`
- [ ] Action masks correctly reflect real spectrum feasibility
- [ ] Parity tests pass (blocking within 5% of legacy)
- [ ] Legacy `GeneralSimEnv` has deprecation warning
- [ ] Factory function allows gradual migration
- [ ] All RL tests pass with >= 80% coverage
- [ ] SB3 MaskablePPO integration works
- [ ] `run_comparison.py` still passes
- [ ] Code passes ruff and mypy

### Gap-Filling Requirements (NEW)
- [ ] Graph observations supported for GNN feature extractors
- [ ] Offline RL policies (BC, IQL) work via OfflinePolicyAdapter
- [ ] Configurable observation space (obs_1 through obs_8)
- [ ] Algorithm-specific migration guide for all 8+ algorithms
- [ ] Per-algorithm differences documented
- [ ] Disaster-aware features supported in PathOption
- [ ] Fragmentation tracking integrated (optional)

## Rollback Strategy

If Phase 4 causes issues:

1. Set `USE_UNIFIED_ENV=0` environment variable
2. Legacy `GeneralSimEnv` continues working unchanged
3. New code isolated in `fusion/rl/` directory
4. No modifications required to legacy RL code paths
5. All legacy paths preserved until Phase 6

## What Phase 4 Does NOT Include

- Removal of legacy `GeneralSimEnv` (deferred to later phase)
- New RL algorithms or training methods
- Changes to routing/spectrum/grooming logic
- Modifications to `fusion/modules/rl/` internals (except deprecation)

## Usage Example (After Implementation)

```python
# Using UnifiedSimEnv with SB3 MaskablePPO
from sb3_contrib import MaskablePPO
from fusion.core.config import SimulationConfig
from fusion.rl.environments import UnifiedSimEnv, ActionMaskWrapper

# Create environment
config = SimulationConfig.from_file("config.ini")
env = UnifiedSimEnv(config)
env = ActionMaskWrapper(env)

# Train
model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate
obs, info = env.reset()
while True:
    action, _ = model.predict(obs, action_masks=env.action_masks())
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Workflow for Executing Tasks

1. Pick a micro-task file (e.g., `P4.1.b_design_pathoption.md`)
2. Load the files listed in "Context to load before running this task"
3. Implement the outputs described in the task
4. Run verification steps
5. Mark task complete, move to next task

Each task is designed to be self-contained with 3-5 context files maximum.

## V3 Architecture Compliance

Phase 4 aligns with the ARCHITECTURE_REFACTOR_PLAN_V3.md principles:

### Unified Execution Path
- RL uses SAME `RoutingPipeline` and `SpectrumPipeline` instances as non-RL simulation
- No `mock_handle_arrival()` duplication
- Feasibility checks come from real spectrum availability

### Domain Object Integration
- Observations built from `Request`, `PathOption`, `NetworkState`
- Rewards computed from `AllocationResult`
- Action masks derived from `PathOption.is_feasible`

### Pipeline Protocol Compliance
- `RLSimulationAdapter` respects pipeline protocols
- Read-only queries through `get_path_options()`
- Mutations only through `SDNOrchestrator.handle_arrival()`

### Backward Compatibility
- Factory function for gradual migration
- Feature flags for environment selection
- Deprecation warnings guide users to new patterns
- Rollback to legacy env with single flag

## Cross-References

### Related V4 Documentation
- `v4-docs/architecture/rl_integration.md` - Original RL architecture design
- `v4-docs/architecture/ml_policies.md` - ML policy architecture
- `v4-docs/decisions/0004-rl-ml-integration.md` - RL/ML integration decisions
- `v4-docs/decisions/0009-rl-env-design.md` - RL environment design decisions
- `v4-docs/decisions/0011-ml-control-policy.md` - ML control policy decisions

### Related V5 Phases
- Phase 1 provides: `SimulationConfig`, `Request`, `Lightpath`, result types
- Phase 2 provides: `NetworkState`, pipeline protocols
- Phase 3 provides: `SDNOrchestrator`, `PipelineFactory`

## Quick Reference: Gap-Filling Documents

| Document | Purpose | Key Classes/Features |
|----------|---------|---------------------|
| P4.1.f | Offline RL compatibility | `DisasterState`, `OfflinePolicyAdapter`, `build_offline_state()` |
| P4.2.f | Graph observations | `PathEncoder`, graph obs space, configurable obs_1-obs_8 |
| P4.3.e | Algorithm migration | Per-algorithm guides (PPO, A2C, DQN, Q-Learning, BC, IQL) |
| P4.4.e | Per-algorithm differences | Compatibility matrix, retraining requirements, adapters |
