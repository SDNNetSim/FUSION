# Final V4 Migration Checklist

## Overview

This checklist consolidates all requirements for declaring V4 migration complete. Use this document to verify that all phases are finished and V4 is the sole supported simulation path.

---

## Phase 1: Core Domain Model

### Documentation

- [ ] `.claude/v4-docs/architecture/domain_model.md` complete
- [ ] `.claude/v4-docs/architecture/result_objects.md` complete
- [ ] `.claude/v4-docs/decisions/0001-frozen-dataclasses.md` accepted
- [ ] `.claude/v4-docs/decisions/0002-request-status-enum.md` accepted
- [ ] `.claude/v4-docs/decisions/0003-result-object-design.md` accepted
- [ ] `.claude/v4-docs/migration/phase_1_core_model.md` complete
- [ ] `.claude/v4-docs/migration/phase_1_checklist.md` all items checked

### Implementation

- [ ] `fusion/domain/__init__.py` exists and exports public API
- [ ] `fusion/domain/config.py` - SimulationConfig implemented
- [ ] `fusion/domain/request.py` - Request, RequestStatus implemented
- [ ] `fusion/domain/lightpath.py` - Lightpath implemented
- [ ] `fusion/domain/results.py` - All result types implemented

### Testing

- [ ] `fusion/tests/domain/` directory exists with tests
- [ ] Domain model unit tests pass
- [ ] Coverage >= 90% for `fusion/domain/`

---

## Phase 2: State Management

### Documentation

- [ ] `.claude/v4-docs/architecture/network_state.md` complete
- [ ] `.claude/v4-docs/architecture/pipeline_interfaces.md` complete
- [ ] `.claude/v4-docs/decisions/0006-networkstate-authority.md` accepted
- [ ] `.claude/v4-docs/migration/phase_2_state_management.md` complete
- [ ] `.claude/v4-docs/migration/phase_2_checklist.md` all items checked

### Implementation

- [ ] `fusion/domain/network_state.py` - NetworkState implemented
- [ ] `fusion/interfaces/pipelines.py` - Pipeline protocols defined
- [ ] `fusion/interfaces/routing_strategy.py` - RoutingStrategy protocol defined

### Testing

- [ ] NetworkState unit tests pass
- [ ] State consistency tests pass (create + release = original)
- [ ] Legacy compatibility properties work (until Phase 6)

---

## Phase 3: Orchestrator

### Documentation

- [ ] `.claude/v4-docs/architecture/orchestration.md` complete
- [ ] `.claude/v4-docs/architecture/pipelines.md` complete
- [ ] `.claude/v4-docs/architecture/pipeline_walkthroughs.md` complete
- [ ] `.claude/v4-docs/decisions/0007-orchestrator-design.md` accepted
- [ ] `.claude/v4-docs/decisions/0008-routing-strategy-pattern.md` accepted
- [ ] `.claude/v4-docs/migration/phase_3_orchestrator.md` complete
- [ ] `.claude/v4-docs/migration/phase_3_checklist.md` all items checked

### Implementation

- [ ] `fusion/core/orchestrator.py` - SDNOrchestrator implemented
- [ ] `fusion/core/pipeline_factory.py` - PipelineFactory, PipelineSet implemented
- [ ] `fusion/pipelines/routing_pipeline.py` - Routing pipelines implemented
- [ ] `fusion/pipelines/spectrum_pipeline.py` - Spectrum pipelines implemented
- [ ] `fusion/pipelines/grooming_pipeline.py` - Grooming pipeline implemented
- [ ] `fusion/pipelines/snr_pipeline.py` - SNR pipeline implemented
- [ ] `fusion/pipelines/slicing_pipeline.py` - Slicing pipeline implemented
- [ ] Feature flag `use_orchestrator` functional

### Testing

- [ ] Orchestrator unit tests pass
- [ ] Pipeline unit tests pass
- [ ] Integration tests pass
- [ ] `run_comparison.py` passes with V4 path enabled

---

## Phase 4: RL Integration

### Documentation

- [ ] `.claude/v4-docs/architecture/rl_integration.md` complete
- [ ] `.claude/v4-docs/architecture/adapter_pattern.md` complete
- [ ] `.claude/v4-docs/decisions/0009-rl-env-design.md` accepted
- [ ] `.claude/v4-docs/migration/phase_4_rl_integration.md` complete
- [ ] `.claude/v4-docs/migration/phase_4_checklist.md` all items checked
- [ ] `.claude/v4-docs/tutorials/using_rl_with_v4_simulator.md` complete

### Implementation

- [ ] `fusion/modules/rl/env_adapter.py` - RLSimulationAdapter implemented
- [ ] `fusion/modules/rl/gymnasium_envs/unified_sim_env.py` - UnifiedSimEnv implemented
- [ ] Old `mock_handle_arrival()` deprecated (deprecation warning added)

### Testing

- [ ] RL adapter unit tests pass
- [ ] UnifiedSimEnv unit tests pass
- [ ] RL training integration tests pass
- [ ] SB3 compatibility verified

---

## Phase 5: ML Control + Protection

### Documentation

- [ ] `.claude/v4-docs/architecture/control_policy.md` complete
- [ ] `.claude/v4-docs/architecture/ml_policies.md` complete
- [ ] `.claude/v4-docs/architecture/heuristic_policies.md` complete
- [ ] `.claude/v4-docs/architecture/protection_pipeline.md` complete
- [ ] `.claude/v4-docs/decisions/0010-control-policy-protocol.md` accepted
- [ ] `.claude/v4-docs/decisions/0011-ml-control-policy.md` accepted
- [ ] `.claude/v4-docs/decisions/0012-protection-pipeline.md` accepted
- [ ] `.claude/v4-docs/migration/phase_5_ml_protection.md` complete
- [ ] `.claude/v4-docs/migration/phase_5_checklist.md` all items checked
- [ ] `.claude/v4-docs/tutorials/implementing_a_new_policy.md` complete

### Implementation

- [ ] `fusion/interfaces/control_policy.py` - ControlPolicy protocol defined
- [ ] `fusion/policies/heuristic_policy.py` - Heuristic policies implemented
- [ ] `fusion/policies/ml_policy.py` - ML policy implemented
- [ ] `fusion/policies/policy_factory.py` - PolicyFactory implemented
- [ ] `fusion/pipelines/protection_pipeline.py` - Protection pipeline implemented
- [ ] Lightpath protection fields added
- [ ] RouteResult backup path fields added

### Testing

- [ ] Policy unit tests pass
- [ ] Protection pipeline unit tests pass
- [ ] Policy integration tests pass
- [ ] Protected routing tests pass

---

## Phase 6: Legacy Removal

### Documentation

- [ ] `.claude/v4-docs/migration/phase_6_legacy_removal.md` complete
- [ ] `.claude/v4-docs/decisions/0013-legacy-removal.md` accepted
- [ ] `.claude/v4-docs/testing/phase_6_testing.md` complete
- [ ] This checklist complete

### P6.1: Remove Unused Legacy Structures

- [ ] `network_spectrum_dict` property removed from NetworkState
- [ ] `lightpath_status_dict` property removed from NetworkState
- [ ] `SDNProps` class removed or deprecated
- [ ] `StatsProps` class removed or deprecated
- [ ] `RoutingProps` class removed or deprecated
- [ ] `mock_handle_arrival()` deleted
- [ ] All P6.1 tests pass

### P6.2: Replace Legacy Call Sites

- [ ] All adapter classes in `fusion/core/adapters/` deleted
- [ ] `to_legacy_dict()` methods removed from domain objects
- [ ] `to_engine_props()` removed (or kept with deprecation warning if CLI needs it)
- [ ] `use_orchestrator` feature flag removed
- [ ] SimulationEngine always uses SDNOrchestrator
- [ ] All P6.2 tests pass

### P6.3: Delete Deprecated Modules

- [ ] `fusion/core/sdn_controller.py` deleted or moved to legacy/
- [ ] `fusion/core/properties.py` deleted or moved to legacy/
- [ ] `fusion/core/adapters/` directory deleted
- [ ] `fusion/legacy/adapters/` directory deleted
- [ ] Old RL environment files deleted
- [ ] All documentation updated
- [ ] Sphinx docs build without errors
- [ ] All P6.3 tests pass

---

## Legacy Uses Verification

### engine_props-like dicts

- [ ] `grep -r "engine_props" fusion/ --include="*.py"` returns only:
  - [ ] `from_engine_props()` class methods (if kept for CLI)
  - [ ] Documentation/comments
- [ ] No code instantiates or passes `engine_props` dicts

### Old SDN Controller Paths

- [ ] `grep -r "SDNController" fusion/ --include="*.py"` returns nothing (or only legacy/ imports)
- [ ] `grep -r "sdn_obj" fusion/ --include="*.py"` returns nothing
- [ ] `grep -r "sdn_controller" fusion/ --include="*.py"` returns only import statements for legacy/

### Legacy Properties

- [ ] `grep -r "network_spectrum_dict" fusion/ --include="*.py"` returns nothing
- [ ] `grep -r "lightpath_status_dict" fusion/ --include="*.py"` returns nothing

---

## RL Flows

- [ ] All RL training uses `UnifiedSimEnv`
- [ ] All RL evaluation uses `RLSimulationAdapter`
- [ ] No code calls `mock_handle_arrival()`
- [ ] RL uses same pipelines as simulation (no duplicated logic)
- [ ] Action masking works correctly

---

## ML / Protection Flows

- [ ] All policy selection uses `ControlPolicy` protocol
- [ ] `PolicyFactory.create_policy()` works for all policy types:
  - [ ] `first_feasible`
  - [ ] `shortest_feasible`
  - [ ] `least_congested`
  - [ ] `load_balanced`
  - [ ] `random_feasible`
  - [ ] `ml` (with model path)
  - [ ] `rl` (with model path)
- [ ] Protection pipeline finds disjoint paths correctly
- [ ] Protected lightpaths allocate spectrum on both paths
- [ ] Failure switchover works

---

## Documentation

### Architecture Docs

- [ ] `.claude/v4-docs/architecture/overview.md` reflects V4-only architecture
- [ ] `.claude/v4-docs/architecture/domain_model.md` up to date
- [ ] `.claude/v4-docs/architecture/network_state.md` up to date
- [ ] `.claude/v4-docs/architecture/orchestration.md` up to date
- [ ] `.claude/v4-docs/architecture/pipelines.md` up to date
- [ ] `.claude/v4-docs/architecture/rl_integration.md` up to date
- [ ] `.claude/v4-docs/architecture/control_policy.md` up to date

### Migration Docs

- [ ] `.claude/v4-docs/migration/overview.md` marked complete
- [ ] All phase docs have "complete" status
- [ ] All phase checklists have all items checked

### Decision Records

- [ ] All ADRs (0001-0013) have "Accepted" status
- [ ] No pending or superseded ADRs without replacements

### Tutorials

- [ ] All tutorials work with V4 architecture
- [ ] No tutorials reference legacy code
- [ ] `.claude/v4-docs/tutorials/migrating_legacy_code_to_v4_domain_model.md` updated for post-migration use

### Sphinx Documentation

- [ ] `docs/conf.py` references V4 modules only
- [ ] `docs/architecture/` wired to V4 architecture docs
- [ ] Sphinx builds without warnings for deleted modules
- [ ] API docs regenerated for V4 modules

---

## CI Enforcement

- [ ] CI runs all V4 test suites
- [ ] CI fails if coverage below thresholds:
  - [ ] `fusion/domain/`: 90%
  - [ ] `fusion/pipelines/`: 85%
  - [ ] `fusion/core/orchestrator.py`: 90%
- [ ] CI runs regression tests against baseline
- [ ] CI runs snapshot comparison tests
- [ ] Phase 6 PRs require `destructive-migration` label
- [ ] Phase 6 PRs require 2 reviewer approvals

---

## Final Verification Commands

```bash
# Full test suite
pytest fusion/tests/ -v --cov=fusion --cov-report=term-missing

# Coverage check
pytest fusion/tests/ --cov=fusion --cov-fail-under=80

# Type check
mypy fusion/ --ignore-missing-imports

# Lint check
ruff check fusion/

# Regression test (baseline mode)
python tests/run_comparison.py --baseline-mode --all-configs --seed 42

# Snapshot comparison
python tests/compare_snapshots.py --baseline tests/snapshots/pre_phase6/ --tolerance 0.02

# Documentation build
cd docs && make html

# Verify no legacy imports in main code
grep -r "from fusion.core.sdn_controller" fusion/ --include="*.py" | grep -v legacy/ | wc -l  # Should be 0
grep -r "from fusion.core.properties" fusion/ --include="*.py" | grep -v legacy/ | wc -l  # Should be 0
```

---

## Sign-Off

### Technical Review

- [ ] All checklist items verified by technical lead
- [ ] All tests passing in CI
- [ ] No known regressions
- [ ] Performance within acceptable bounds (< 10% slower than legacy)

### Documentation Review

- [ ] All documentation reviewed for accuracy
- [ ] All tutorials tested and working
- [ ] README updated with V4 examples

### Final Approval

- [ ] Migration declared complete
- [ ] V4 is the sole supported simulation path
- [ ] Legacy code deleted or moved to clearly marked legacy directory

---

## Related Documents

- [Phase 6: Legacy Removal](./phase_6_legacy_removal.md)
- [Phase 6 Testing](../testing/phase_6_testing.md)
- [ADR-0013: Legacy Removal](../decisions/0013-legacy-removal.md)
- [Migration Overview](./overview.md)
- [Test Strategy](../testing/test_strategy.md)
