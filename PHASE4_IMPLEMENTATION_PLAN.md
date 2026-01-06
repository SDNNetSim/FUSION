# Phase 4 Implementation Strategy - Incremental Approach

## Why Phase 4 Exists

### The Problem with Legacy RL

```
GeneralSimEnv (legacy RL)
       │
       ▼
mock_handle_arrival()  ← DUPLICATED simulation logic
       │
       ▼
Results (can diverge from real simulation)
```

The legacy RL environment has its **own copy** of the simulation logic via `mock_handle_arrival()`. This means:
- Bug fixes to the real simulation don't automatically apply to RL
- RL behavior can diverge from actual simulation behavior
- Two codebases to maintain

### The Solution: Unified Environment

```
UnifiedSimEnv (new RL)
       │
       ▼
RLSimulationAdapter
       │
       ▼
SDNOrchestrator  ← SAME orchestrator as regular simulation
       │
       ▼
Pipelines (routing, spectrum, etc.)  ← SAME pipelines
       │
       ▼
Results (guaranteed to match simulation)
```

**Core Principle:** UnifiedSimEnv uses the **exact same** orchestrator and pipelines as regular simulation. No forked simulator.

---

## Problem Statement

Previous phases caused 2 weeks of debugging because:
1. Large batches of changes were made at once
2. Bugs accumulated and were hard to isolate
3. Testing happened after implementation, not during

## Proposed Approach: Micro-Commits

Instead of "design all, implement all", do:
1. Implement ONE small thing
2. Write tests for it
3. Verify it works (pytest + ruff + mypy + run_comparison.py)
4. Commit
5. Move to next thing

**Key Rule: Each chunk is ONE commit. If bugs appear later, git bisect finds the exact commit.**

---

## Verification Strategy

### What `run_comparison.py` Validates

```
run_comparison.py tests:

Regular Simulation Path:
    SimulationEngine → SDNOrchestrator → Pipelines → Results
         vs
    SimulationEngine → Legacy SDNController → Results

    ✓ Validates orchestrator produces same results as legacy
```

**Important:** `run_comparison.py` validates the orchestrator/pipelines work correctly. Since UnifiedSimEnv uses the same orchestrator/pipelines, this is foundational validation.

### What `run_comparison.py` Does NOT Validate

- That `RLSimulationAdapter` calls the orchestrator correctly
- That `UnifiedSimEnv` calls the adapter correctly
- That observations are built correctly
- That rewards are computed correctly

These are NEW code paths that need their own verification.

### Honest Assessment: When Can We Verify RL Results?

| Chunks | What We Can Verify |
|--------|-------------------|
| 1-5 | **Unit tests only** - Adapter exists but nothing uses it yet. `run_comparison.py` confirms we didn't break orchestrator. |
| 6-8 | **Unit tests only** - Env exists but can't run full episode yet. |
| **9** | **FIRST REAL RL VERIFICATION** - Env passes gymnasium checker, can run manual episode, can compare with GeneralSimEnv |
| 10-11 | Can verify SB3 training runs without crashing |
| 12 | Can verify GNN observations work with feature extractors |
| **16-18** | **FULL PARITY** - Statistical comparison of UnifiedSimEnv vs GeneralSimEnv |

### The Verification Chain

1. **Orchestrator correct?** → `run_comparison.py` (run after every chunk)
2. **Adapter calls orchestrator correctly?** → Integration test at Chunk 5
3. **Env works end-to-end?** → Manual episode test at Chunk 9
4. **RL training works?** → SB3 smoke test at Chunk 11
5. **Results match legacy?** → Parity tests at Chunks 16-18

---

## Proposed Implementation Chunks

### Phase 4.1: RLSimulationAdapter (5 chunks)

**Chunk 1: Package + PathOption dataclass**
- Files: `fusion/rl/__init__.py`, `fusion/rl/path_option.py`
- Scope: Just the frozen dataclass with fields
- Tests: Can create, is frozen, has expected fields
- Verify: pytest, ruff, mypy, run_comparison.py pass
- Commit message: "feat(rl): add PathOption dataclass"

**Chunk 2: RLSimulationAdapter skeleton**
- Files: `fusion/rl/adapter.py`
- Scope: Class with `__init__` only, stores pipeline references
- Tests: `adapter.routing is orchestrator.routing` (identity check)
- Verify: pytest, run_comparison.py pass
- Commit message: "feat(rl): add RLSimulationAdapter skeleton"

**Chunk 3: get_path_options() method**
- Files: Modify `adapter.py`
- Scope: Add one method that calls routing pipeline
- Tests: Returns list of PathOption, correct count, correct fields
- Verify: pytest, run_comparison.py pass
- Commit message: "feat(rl): add get_path_options to adapter"

**Chunk 4: apply_action() method**
- Files: Modify `adapter.py`
- Scope: Add method that calls orchestrator.handle_arrival
- Tests: Verify allocation goes through orchestrator
- Verify: pytest, run_comparison.py pass
- Commit message: "feat(rl): add apply_action to adapter"

**Chunk 5: compute_reward() method**
- Files: Modify `adapter.py`
- Scope: Add reward computation
- Tests: Correct reward for success/block
- Commit message: "feat(rl): add compute_reward to adapter"

**>>> INTEGRATION CHECKPOINT 1 <<<**
After Chunk 5, verify adapter works with real orchestrator:
```python
orchestrator = PipelineFactory.create_orchestrator(config)
adapter = RLSimulationAdapter(orchestrator)
network_state = NetworkState(config)
options = adapter.get_path_options(request, network_state)
# Verify: paths returned match what legacy routing would return
```

### Phase 4.2: UnifiedSimEnv (7 chunks)

**Chunk 6: Environment skeleton**
- Files: `fusion/rl/environments/__init__.py`, `fusion/rl/environments/unified_env.py`
- Scope: Class with `__init__`, observation_space, action_space only
- Tests: Spaces are valid Gymnasium spaces
- Commit message: "feat(rl): add UnifiedSimEnv skeleton"

**Chunk 7: reset() method**
- Files: Modify `unified_env.py`
- Scope: Add reset(), returns observation and info
- Tests: Returns valid observation, info has action_mask
- Commit message: "feat(rl): add reset to UnifiedSimEnv"

**Chunk 8: step() method**
- Files: Modify `unified_env.py`
- Scope: Add step(), returns obs, reward, terminated, truncated, info
- Tests: Step returns valid types, terminated when episode ends
- Commit message: "feat(rl): add step to UnifiedSimEnv"

**Chunk 9: Pass gymnasium.utils.env_checker**
- Files: Modify `unified_env.py` as needed
- Scope: Fix any issues the checker finds
- Tests: `gymnasium.utils.env_checker(env)` passes
- Commit message: "feat(rl): UnifiedSimEnv passes gymnasium env_checker"

**>>> INTEGRATION CHECKPOINT 2 <<<**
After Chunk 9, run manual episode and compare with legacy:
```python
# Run both envs with same seed
legacy_env = GeneralSimEnv(config, seed=42)
unified_env = UnifiedSimEnv(config, seed=42)

# Run 100 steps on each, compare:
# - Number of blocks
# - Number of successes
# - Rewards received
# Should be identical or very close
```

**Chunk 10: ActionMaskWrapper**
- Files: `fusion/rl/environments/wrappers.py`
- Scope: Wrapper that exposes action_masks() method
- Tests: Works with mock env, masks correct shape
- Commit message: "feat(rl): add ActionMaskWrapper"

**Chunk 11: SB3 MaskablePPO integration test**
- Files: Test file only
- Scope: Integration test with SB3
- Tests: Can create MaskablePPO with env, can call predict, can train 1000 steps
- Commit message: "test(rl): add SB3 MaskablePPO integration test"

**>>> INTEGRATION CHECKPOINT 3 <<<**
After Chunk 11, verify RL training runs:
```python
env = UnifiedSimEnv(config)
env = ActionMaskWrapper(env)
model = MaskablePPO("MlpPolicy", env)
model.learn(total_timesteps=1000)  # Should not crash
```

**Chunk 12: Graph observation support for GNN**
- Files: Modify `unified_env.py`, add `fusion/rl/environments/graph_obs.py`
- Scope: Add graph observation space (x, edge_index, edge_attr, path_masks)
- Tests: Graph obs has correct shape, PathEncoder works
- Commit message: "feat(rl): add graph observation support for GNN"

**>>> INTEGRATION CHECKPOINT 4 <<<**
After Chunk 12, verify GNN feature extractors work:
```python
env = UnifiedSimEnv(config, obs_type="graph")
obs, info = env.reset()
# Verify obs has x, edge_index, edge_attr, path_masks
# Verify Graphormer/PathGNN can process it
```

### Phase 4.3: Migration Infrastructure (3 chunks)

**Chunk 13: Factory function**
- Files: Modify `fusion/modules/rl/gymnasium_envs/__init__.py`
- Scope: Add `create_sim_env(config, env_type="legacy"|"unified")`
- Tests: Returns correct env type based on parameter
- Commit message: "feat(rl): add create_sim_env factory function"

**Chunk 14: Environment variable toggle**
- Files: Modify factory function
- Scope: Check `USE_UNIFIED_ENV` env var
- Tests: Env var controls which env is returned
- Commit message: "feat(rl): add USE_UNIFIED_ENV toggle"

**Chunk 15: Deprecation warning on GeneralSimEnv**
- Files: Modify `general_sim_env.py`
- Scope: Add DeprecationWarning in __init__
- Tests: Warning is raised
- Commit message: "feat(rl): add deprecation warning to GeneralSimEnv"

### Phase 4.4: Parity Validation (3 chunks)

**Chunk 16: Basic parity test**
- Files: `fusion/tests/rl/test_rl_parity.py`
- Scope: Compare blocking probability legacy vs unified (1 config, 1000 requests)
- Tests: Blocking probability within 5% difference
- Commit message: "test(rl): add basic parity test"

**Chunk 17: Extended parity tests + multiple algorithms**
- Files: Modify parity test file
- Scope: Test PPO, A2C, DQN with both envs
- Tests: All algorithms produce comparable training curves
- Commit message: "test(rl): extend parity tests to multiple algorithms"

**Chunk 18: Comparison script**
- Files: `scripts/compare_rl_envs.py`
- Scope: CLI script to compare environments
- Tests: Script runs, outputs comparison report
- Commit message: "feat(rl): add compare_rl_envs.py script"

**>>> FINAL VERIFICATION <<<**
After Chunk 18:
```bash
python scripts/compare_rl_envs.py --episodes 100 --seed 42
# Should show: blocking rates match within 5%
```

---

## Workflow Per Chunk

```
1. Read the relevant Phase 4 doc section for context
2. Write the code (1-2 files max)
3. Write the tests
4. Run: pytest fusion/tests/rl/ -v
5. Run: ruff check fusion/rl/
6. Run: mypy fusion/rl/
7. Run: python tests/run_comparison.py  ← Ensures orchestrator still works
8. If at integration checkpoint: Run integration verification
9. Commit with descriptive message
10. Move to next chunk
```

## If Something Fails

**Unit test fails:** Fix the code in current chunk, re-run tests

**run_comparison.py fails:** You broke the orchestrator somehow. Check what you modified. The bug is in the current chunk.

**Integration checkpoint fails:** The new code doesn't correctly interface with existing code. Debug the integration, not the individual components.

**Parity test fails:** The entire chain works but produces different results than legacy. This is the hardest to debug - use the golden recording approach:
1. Record legacy env behavior step-by-step
2. Replay same steps on unified env
3. Find first divergence point

---

## User Decisions (Captured)

1. **Offline RL (BC/IQL, disaster features):** Defer - not tested yet, will use in future phase
2. **GNN support:** YES - need Graphormer, PathGNN, PathGNNCached support
3. **Algorithms:** Multiple - PPO, A2C, DQN, etc. need comprehensive testing

## What's Deferred

**NOT in this plan (future phase):**
- DisasterState dataclass
- build_offline_state() method
- OfflinePolicyAdapter for BC/IQL
- Disaster-aware features in PathOption

**Included:**
- Graph observation support for GNN (Chunk 12)
- Multiple algorithm testing (Chunk 17)

---

## Summary

**Total chunks: 18**

**Priority order:**
1. Chunks 1-5: RLSimulationAdapter (core functionality)
2. Chunks 6-12: UnifiedSimEnv (basic env + GNN support)
3. Chunks 13-15: Migration infrastructure
4. Chunks 16-18: Parity validation

**Estimated time per chunk:** 30-60 minutes (including tests)

**Total estimated time:** ~9-18 hours of focused work (spread over multiple days)

---

## Files to Create (New)

```
fusion/rl/
    __init__.py
    path_option.py
    adapter.py
    environments/
        __init__.py
        unified_env.py
        wrappers.py
        graph_obs.py          # For GNN support

fusion/tests/rl/
    __init__.py
    test_path_option.py
    test_adapter.py
    test_unified_env.py
    test_rl_parity.py

scripts/
    compare_rl_envs.py
```

## Files to Modify (Existing)

```
fusion/modules/rl/gymnasium_envs/__init__.py  # Add factory function
fusion/modules/rl/gymnasium_envs/general_sim_env.py  # Add deprecation warning
```

## Critical: What We Do NOT Touch

- `fusion/modules/rl/` internals (algorithms, policies, agents, etc.) - these stay as-is
- `fusion/core/` - orchestrator and pipelines stay as-is
- `fusion/pipelines/` - stay as-is
- Existing tests - they must continue to pass

---

## How to Work with Claude (Context Management)

### The Problem
Claude will lose context between sessions and may forget details discussed earlier.

### The Solution: Use This Plan File as Persistent Memory

**At the start of each session, tell Claude:**
```
Read PHASE4_IMPLEMENTATION_PLAN.md
and .claude/v5-final-docs/phase-4-rl-integration/phase_4_overview.md

We are implementing Phase 4 RL Integration using the incremental approach.
Currently on Chunk X. Last completed chunk was: [chunk name]
Last commit was: [commit hash or message]
```

### Per-Chunk Workflow

**Before starting a chunk:**
```
Read PHASE4_IMPLEMENTATION_PLAN.md
I want to implement Chunk X: [chunk name]
Read the relevant Phase 4 doc if needed: [specific doc path]
```

**After completing a chunk:**
1. Verify all tests pass (pytest + ruff + mypy + run_comparison.py)
2. If at integration checkpoint, run integration verification
3. Commit with the message from the plan
4. Update the progress tracking table below

### Progress Tracking (Update After Each Chunk)

| Chunk | Status | Commit Hash | Notes |
|-------|--------|-------------|-------|
| 1. PathOption | Not started | - | - |
| 2. Adapter skeleton | Not started | - | - |
| 3. get_path_options | Not started | - | - |
| 4. apply_action | Not started | - | - |
| 5. compute_reward | Not started | - | CHECKPOINT 1 |
| 6. Env skeleton | Not started | - | - |
| 7. reset() | Not started | - | - |
| 8. step() | Not started | - | - |
| 9. env_checker | Not started | - | CHECKPOINT 2 |
| 10. ActionMaskWrapper | Not started | - | - |
| 11. SB3 test | Not started | - | CHECKPOINT 3 |
| 12. GNN support | Not started | - | CHECKPOINT 4 |
| 13. Factory function | Not started | - | - |
| 14. Env var toggle | Not started | - | - |
| 15. Deprecation warning | Not started | - | - |
| 16. Basic parity | Not started | - | - |
| 17. Multi-algo parity | Not started | - | - |
| 18. Compare script | Not started | - | FINAL |

### Key Constraints to Remind Claude

1. **One chunk = one commit** - Don't batch multiple chunks
2. **Tests before commit** - pytest + ruff + mypy + run_comparison.py must pass
3. **Additive only** - Don't modify existing fusion/modules/rl/ code (except where specified)
4. **Identity check** - adapter.routing IS orchestrator.routing (same object, not copy)
5. **No forked simulator** - UnifiedSimEnv uses same pipelines as orchestrator

### Recovery If Claude Forgets

If Claude seems confused or forgets the approach:
```
STOP. Read PHASE4_IMPLEMENTATION_PLAN.md first.
We are doing incremental implementation - one small chunk at a time.
Each chunk must be tested and committed before moving to the next.
The goal is to avoid 2-week debugging sessions by catching issues immediately.
```

### What to Do If a Chunk Has Bugs

1. DON'T move to next chunk
2. Fix the bug in the current chunk
3. Re-run all tests
4. Amend the commit OR create a fix commit
5. THEN move to next chunk

### What to Do If Integration Checkpoint Fails

1. The bug is in one of the chunks since the last checkpoint
2. Review each chunk's changes
3. Add more targeted tests to isolate the issue
4. Fix and verify before continuing

---

## Reference Documents

- Phase 4 Overview: `.claude/v5-final-docs/phase-4-rl-integration/phase_4_overview.md`
- Gap Analysis: `.claude/v5-final-docs/phase-4-rl-integration/P4.0_gap_analysis.md`
- P4.1 Index (Adapter): `.claude/v5-final-docs/phase-4-rl-integration/P4.1_rl_adapter/P4.1.index.md`
- P4.2 Index (Env): `.claude/v5-final-docs/phase-4-rl-integration/P4.2_unified_env/P4.2.index.md`
- P4.3 Index (Migration): `.claude/v5-final-docs/phase-4-rl-integration/P4.3_migrate_experiments/P4.3.index.md`
- P4.4 Index (Parity): `.claude/v5-final-docs/phase-4-rl-integration/P4.4_parity_and_differences/P4.4.index.md`

---

## Quick Reference: Verification Commands

```bash
# After every chunk:
pytest fusion/tests/rl/ -v
ruff check fusion/rl/
mypy fusion/rl/
python tests/run_comparison.py

# At integration checkpoints (as specified above):
# Run the checkpoint-specific verification code

# Final parity check:
python scripts/compare_rl_envs.py --episodes 100 --seed 42
```
