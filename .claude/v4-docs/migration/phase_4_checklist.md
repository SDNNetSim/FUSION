# Phase 4 Checklist: RL Integration

## Overview

This checklist tracks completion of Phase 4 (RL Integration) micro-phases. Each item must be checked off before proceeding to the next phase.

---

## Prerequisites

- [ ] Phase 1 complete (domain objects)
- [ ] Phase 2 complete (NetworkState, pipeline protocols)
- [ ] Phase 3 complete (SDNOrchestrator, PipelineFactory)
- [ ] `run_comparison.py` passing with orchestrator path

---

## P4.1: RLSimulationAdapter Scaffolding

### Files to Create

- [ ] `fusion/rl/__init__.py`
- [ ] `fusion/rl/adapter.py` (RLSimulationAdapter, PathOption)
- [ ] `fusion/rl/observation.py` (observation building utilities)
- [ ] `fusion/rl/reward.py` (reward computation utilities)

### Implementation Checklist

- [ ] `PathOption` dataclass defined with all fields
- [ ] `RLSimulationAdapter.__init__` stores references to orchestrator pipelines
- [ ] `RLSimulationAdapter.get_path_options()` method signature defined
- [ ] `RLSimulationAdapter.apply_action()` method signature defined
- [ ] `RLSimulationAdapter.compute_reward()` method signature defined
- [ ] No direct NetworkState storage in adapter

### Tests

- [ ] `fusion/tests/rl/__init__.py` created
- [ ] `fusion/tests/rl/test_adapter.py` created
- [ ] Test: adapter uses same pipeline instances as orchestrator
- [ ] Test: adapter does not create own pipelines

### Verification

```bash
pytest fusion/tests/rl/test_adapter.py -v
ruff check fusion/rl/
mypy fusion/rl/
```

- [ ] All tests pass
- [ ] No ruff errors
- [ ] No mypy errors

---

## P4.2: Wire RL Adapter to Pipelines

### Files to Create

- [ ] `fusion/rl/environments/__init__.py`
- [ ] `fusion/rl/environments/unified_env.py` (UnifiedSimEnv)
- [ ] `fusion/rl/environments/wrappers.py` (ActionMaskWrapper)

### Files to Modify

- [ ] `fusion/rl/adapter.py` (add observation/reward implementations)
- [ ] `fusion/core/simulation.py` (add `get_next_request()`, `process_releases_until()`)

### Implementation Checklist

- [ ] `UnifiedSimEnv` implements `gymnasium.Env`
- [ ] `observation_space` defined as `spaces.Dict`
- [ ] `action_space` defined as `spaces.Discrete(k_paths)`
- [ ] `reset()` returns (obs, info) with action_mask in info
- [ ] `step()` calls `adapter.apply_action()` which calls orchestrator
- [ ] `step()` returns (obs, reward, terminated, truncated, info)
- [ ] `_get_action_mask()` returns mask from PathOption feasibility
- [ ] `SimulationEngine.get_next_request()` method added
- [ ] `SimulationEngine.process_releases_until()` method added

### Tests

- [ ] `fusion/tests/rl/test_unified_env.py` created
- [ ] Test: env observation shape matches observation_space
- [ ] Test: env action_mask in info dict
- [ ] Test: env step calls orchestrator.handle_arrival
- [ ] Test: env is gymnasium compliant (check_env)

### Verification

```bash
pytest fusion/tests/rl/test_unified_env.py -v
python -c "from gymnasium.utils.env_checker import check_env; from fusion.rl.environments import UnifiedSimEnv; check_env(UnifiedSimEnv(config))"
```

- [ ] All tests pass
- [ ] Gymnasium env checker passes

---

## P4.3: Migrate Existing RL Experiments

### Files to Modify

- [ ] `fusion/modules/rl/gymnasium_envs/__init__.py` (add factory, re-exports)
- [ ] Add deprecation warning to `GeneralSimEnv.__init__`

### Implementation Checklist

- [ ] `create_sim_env()` factory function added
- [ ] `USE_UNIFIED_ENV` feature flag added
- [ ] Deprecation warning in `GeneralSimEnv` pointing to `UnifiedSimEnv`
- [ ] Re-export `UnifiedSimEnv` from `fusion.modules.rl.gymnasium_envs`

### Documentation Updates

- [ ] Update any RL example scripts to show both old and new usage
- [ ] Add migration notes to docstrings

### Verification

```bash
# Test legacy path still works
python -c "from fusion.modules.rl.gymnasium_envs import GeneralSimEnv; print('Legacy OK')"

# Test new path works
python -c "from fusion.rl.environments import UnifiedSimEnv; print('New OK')"

# Test factory
python -c "from fusion.modules.rl.gymnasium_envs import create_sim_env; print('Factory OK')"
```

- [ ] Legacy import works (with deprecation warning)
- [ ] New import works
- [ ] Factory function works

---

## P4.4: Validate Parity and Document Differences

### Files to Create

- [ ] `fusion/tests/rl/test_rl_parity.py`

### Tests

- [ ] Test: same seed produces same traffic sequence
- [ ] Test: same action produces same allocation outcome
- [ ] Test: blocking probability within 5% of legacy over 10 episodes
- [ ] Test: deterministic reproducibility with same seed

### Parity Validation

```bash
pytest fusion/tests/rl/test_rl_parity.py -v --timeout=300
```

- [ ] All parity tests pass

### Document Any Differences

If differences found:
- [ ] Document in `docs/migration/rl_differences.md`
- [ ] Explain cause of each difference
- [ ] Note impact on existing trained policies
- [ ] Provide migration guidance

---

## Additional Tests

### Pipeline Bypass Prevention

- [ ] `fusion/tests/rl/test_no_bypass.py` created
- [ ] Test: RL adapter does not mutate spectrum directly
- [ ] Test: RL env mutations go through orchestrator
- [ ] Test: observation building is read-only

### NetworkState Authority

- [ ] `fusion/tests/rl/test_state_authority.py` created
- [ ] Test: RL adapter does not store NetworkState
- [ ] Test: each call uses passed NetworkState

### Verification

```bash
pytest fusion/tests/rl/ -v --cov=fusion/rl --cov-report=term-missing
```

- [ ] All RL tests pass
- [ ] Coverage >= 80% for `fusion/rl/`

---

## Integration Tests

### With SB3

```bash
# Quick smoke test with MaskablePPO
python -c "
from sb3_contrib import MaskablePPO
from fusion.rl.environments import UnifiedSimEnv
from fusion.domain.config import SimulationConfig

config = SimulationConfig.from_engine_props({'network': 'NSFNET', 'num_requests': 10})
env = UnifiedSimEnv(config)
model = MaskablePPO('MultiInputPolicy', env, verbose=0)
model.learn(total_timesteps=100)
print('SB3 integration OK')
"
```

- [ ] SB3 MaskablePPO can instantiate with UnifiedSimEnv
- [ ] SB3 MaskablePPO can run learn() without error

### With run_comparison.py

```bash
# Ensure orchestrator path still works
python tests/run_comparison.py
```

- [ ] run_comparison.py still passes (RL changes don't break simulation)

---

## Documentation Checklist

### Created in This Phase

- [ ] `architecture/rl_integration.md`
- [ ] `migration/phase_4_rl_integration.md`
- [ ] `migration/phase_4_checklist.md` (this file)
- [ ] `decisions/0009-rl-env-design.md`
- [ ] `testing/phase_4_testing.md`
- [ ] `tutorials/using_rl_with_v4_simulator.md`

### Cross-References Updated

- [ ] `architecture/orchestration.md` references RL adapter
- [ ] `migration/overview.md` updated with Phase 4 status
- [ ] `testing/test_strategy.md` includes RL test categories

---

## Exit Criteria

All items below must be checked before Phase 4 is complete:

- [ ] `RLSimulationAdapter` uses same pipeline instances as orchestrator
- [ ] `UnifiedSimEnv` passes gymnasium env checker
- [ ] Action masks correctly reflect real spectrum feasibility
- [ ] Parity tests pass (blocking within 5% of legacy)
- [ ] Legacy `GeneralSimEnv` has deprecation warning
- [ ] Factory function allows gradual migration
- [ ] All RL tests pass with >= 80% coverage
- [ ] SB3 integration smoke test passes
- [ ] run_comparison.py still passes
- [ ] All documentation created and cross-referenced
- [ ] Code passes ruff, mypy

---

## Rollback Plan

If Phase 4 causes issues:

1. [ ] Set `USE_UNIFIED_ENV=False` environment variable
2. [ ] Legacy `GeneralSimEnv` continues to work
3. [ ] New code isolated in `fusion/rl/` directory
4. [ ] No modifications to legacy RL code paths required

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| Reviewer | | | |

---

## Notes

- Legacy RL removal is deferred to Phase 6
- Offline RL (BC, IQL, CQL) support is a future enhancement
- Multi-agent RL is out of scope for Phase 4
