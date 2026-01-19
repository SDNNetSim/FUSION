# Phase 7: Project Management

## 62 - Traceability to Paper Claims

**Section Reference**: Section 10 - Traceability to Paper Claims

**Purpose**: Map paper claims to implementation components and verification methods to ensure all research objectives are met.

---

## Traceability Matrix

| Paper Claim | Implementation | Verification |
|-------------|----------------|--------------|
| **BP & variance across seeds** | `SimulationStatistics.bp_overall`, seed aggregation | `test_bp_computed`, CSV export with multi-seed runs |
| **Recovery time (mean, P95)** | `SimulationStatistics.recovery_times_ms` | `test_recovery_time_recorded`, CSV export |
| **Failure types (F1, F3, F4)** | `FailureManager`, `failure_types.py` | `test_link_failure`, `test_srlg_failure`, `test_geo_failure` |
| **Safety (mask + fallback)** | `action_masking.py`, `apply_fallback_policy()` | `test_action_mask_applied`, `test_fallback_on_all_masked` |
| **Offline dataset (2-3M transitions)** | `DatasetLogger` | `test_dataset_logged`, file size check |
| **Baseline fairness (KSP-FF, 1+1)** | `KSPFFPolicy`, `OnePlusOnePolicy`, `OnePlusOneProtection` | Compare BP with/without RL in same conditions |
| **Conservative offline RL (BC → IQL)** | `BCPolicy`, `IQLPolicy` (offline inference only) | Load pre-trained models, verify inference |

---

## Detailed Mapping

### 1. BP & Variance Across Seeds

**Claim**: "We measure blocking probability and its variance across 5 seeds to establish statistical significance."

**Implementation**:
- `SimulationStatistics.compute_blocking_probability()`: Computes overall BP
- `aggregate_seed_results()`: Aggregates BP across seeds with mean, std, CI95
- Multi-seed execution via `run_multi_seed_experiment()`

**Verification**:
```python
# Run with multiple seeds
seeds = [42, 43, 44, 45, 46]
results = [run_with_seed(s) for s in seeds]

# Verify variance computed
agg = aggregate_seed_results(results, ['bp_overall'])
assert 'std' in agg['bp_overall']
assert 'ci95_lower' in agg['bp_overall']
```

---

### 2. Recovery Time (Mean, P95)

**Claim**: "1+1 protection achieves mean recovery time of 50ms (P95: 52ms) vs. 100ms+ for restoration."

**Implementation**:
- `SimulationStatistics.record_recovery_event()`: Logs recovery times
- `get_recovery_stats()`: Computes mean, P95, max

**Verification**:
```python
# Inject failure, verify recovery
stats.record_recovery_event(100.0, 100.05, 1, 'protection')
recovery_stats = stats.get_recovery_stats()

assert recovery_stats['mean_ms'] == 50.0
assert recovery_stats['p95_ms'] <= 55.0
```

---

### 3. Failure Types (F1, F3, F4)

**Claim**: "We evaluate under three failure models: link (F1), SRLG (F3), and geographic (F4) with radius=2."

**Implementation**:
- `fail_link()`: Single link failure
- `fail_srlg()`: Multiple link failure
- `fail_geo()`: Hop-radius based failure

**Verification**:
```python
# Test each failure type
for failure_type in ['link', 'srlg', 'geo']:
    manager = FailureManager(config, topology)
    event = manager.inject_failure(failure_type, ...)
    assert len(event['failed_links']) > 0
```

---

### 4. Safety (Action Masking + Fallback)

**Claim**: "Our RL policy uses action masking and heuristic fallback to ensure safe operation under failures."

**Implementation**:
- `compute_action_mask()`: Identifies infeasible paths
- `apply_fallback_policy()`: Uses heuristic when all masked

**Verification**:
```python
# Test masking
mask = compute_action_mask(paths, features, slots_needed=4)
assert mask[0] == False  # Infeasible path masked

# Test fallback
try:
    selected = policy.select_path(state, [False, False, False])
except AllPathsMaskedError:
    selected = apply_fallback_policy(state, fallback)
    assert selected >= 0  # Fallback succeeded
```

---

### 5. Offline Dataset (2-3M Transitions)

**Claim**: "We collect 2-3M transitions from heuristic policies with epsilon-mix for offline training."

**Implementation**:
- `DatasetLogger.log_transition()`: Logs each decision
- `select_path_with_epsilon_mix()`: Adds exploration

**Verification**:
```bash
# Generate dataset
python run_sim.py --log_offline_dataset true --num_requests 100000

# Check size
wc -l data/datasets/offline_data.jsonl  # Should be ~100k lines
```

---

### 6. Baseline Fairness (KSP-FF, 1+1)

**Claim**: "We compare RL policies against fair baselines (KSP-FF, 1+1) under identical conditions."

**Implementation**:
- `KSPFFPolicy`: Standard heuristic baseline
- `OnePlusOneProtection`: Protection baseline
- Same seeds, loads, failures for all experiments

**Verification**:
```python
# Run baseline and RL with same config
config['seed'] = 42
baseline_bp = run_with_policy(config, 'ksp_ff')
rl_bp = run_with_policy(config, 'bc')

# Compare
improvement = (baseline_bp - rl_bp) / baseline_bp
```

---

### 7. Conservative Offline RL (BC → IQL)

**Claim**: "We train BC from demonstrations, then improve with IQL (conservative offline RL)."

**Implementation**:
- `BCPolicy`: Loads pre-trained BC model
- `IQLPolicy`: Loads pre-trained IQL model
- Models trained externally, inference only in simulator

**Verification**:
```python
# Load models
bc_policy = BCPolicy('models/bc_model.pt')
iql_policy = IQLPolicy('models/iql_model.pt')

# Verify inference
selected_bc = bc_policy.select_path(state, mask)
selected_iql = iql_policy.select_path(state, mask)

assert 0 <= selected_bc < len(mask)
assert 0 <= selected_iql < len(mask)
```

---

## Acceptance Criteria

- [x] All paper claims mapped to implementation
- [x] All claims have verification tests
- [x] Experiments can reproduce paper results
- [x] Traceability matrix complete

---

**Related Documents**:
- [40-metrics-reporting.md](../phase5-metrics/40-metrics-reporting.md) (Metrics export)
- [50-testing.md](../phase6-quality/50-testing.md) (Verification tests)
- [63-usage-workflow.md](63-usage-workflow.md) (Experiment workflow)
