# Phase 6 Testing Strategy

## Overview

Phase 6 involves destructive changes: removing legacy code, deleting deprecated modules, and eliminating transitional shims. This testing strategy ensures these changes do not introduce regressions or break functionality.

## Required Test Suites

Before any Phase 6 PR merges, the following test suites must pass:

### 1. Domain Model Unit Tests

**Location**: `fusion/tests/domain/`

**Coverage Requirements**: >= 90%

**Key Tests**:
```
test_request.py
test_lightpath.py
test_network_state.py
test_link_spectrum.py
test_simulation_config.py
test_result_objects.py
```

**Verification Command**:
```bash
pytest fusion/tests/domain/ -v --cov=fusion/domain --cov-report=term-missing --cov-fail-under=90
```

### 2. NetworkState Unit Tests

**Location**: `fusion/tests/domain/test_network_state.py`

**Critical Tests**:
- Spectrum allocation/release
- Lightpath creation/deletion
- State consistency (create + release = original state)
- Multi-band, multi-core operations
- Protected lightpath management

**Verification Command**:
```bash
pytest fusion/tests/domain/test_network_state.py -v
```

### 3. Pipeline Unit Tests

**Location**: `fusion/tests/pipelines/`

**Key Tests**:
```
test_routing_pipeline.py
test_spectrum_pipeline.py
test_grooming_pipeline.py
test_snr_pipeline.py
test_slicing_pipeline.py
test_protection_pipeline.py
```

**Verification Command**:
```bash
pytest fusion/tests/pipelines/ -v --cov=fusion/pipelines --cov-fail-under=85
```

### 4. Orchestrator Unit Tests

**Location**: `fusion/tests/core/test_orchestrator.py`

**Critical Tests**:
- Plain KSP flow
- Grooming flow
- Slicing flow
- SNR validation flow
- Protection flow
- Combined feature flows
- Policy integration

**Verification Command**:
```bash
pytest fusion/tests/core/test_orchestrator.py -v --cov=fusion/core/orchestrator --cov-fail-under=90
```

### 5. RL/ML Integration Tests

**Location**: `fusion/tests/rl/`, `fusion/tests/policies/`

**Key Tests**:
```
test_rl_simulation_adapter.py
test_unified_sim_env.py
test_heuristic_policy.py
test_ml_policy.py
test_policy_factory.py
```

**Verification Command**:
```bash
pytest fusion/tests/rl/ fusion/tests/policies/ -v
```

### 6. End-to-End Simulation Tests

**Location**: `fusion/tests/integration/`

**Key Tests**:
```
test_full_simulation.py
test_orchestrator_flows.py
test_state_consistency.py
test_feature_combinations.py
```

**Verification Command**:
```bash
pytest fusion/tests/integration/ -v
```

### 7. Regression/Comparison Tests

**Location**: `tests/run_comparison.py`

**Purpose**: Verify V4 produces statistically equivalent results to a known-good baseline.

**Verification Command**:
```bash
# Against stored baseline (post-Phase-6 mode)
python tests/run_comparison.py --baseline-mode --all-configs --seed 42

# With tolerance checks
python tests/run_comparison.py --baseline-mode --tolerance 0.02
```

---

## Snapshot/Regression Testing for Subtle Behavior Changes

Phase 6 removals may introduce subtle behavior changes that unit tests miss. Use snapshot testing to detect these.

### Approach: Capture-and-Compare

1. **Before Phase 6**: Capture "golden" outputs for key scenarios
2. **During Phase 6**: Run same scenarios, compare against golden outputs
3. **After Phase 6**: Update golden outputs if changes are intentional and verified

### Snapshot Categories

| Category | What to Capture | Tolerance |
|----------|-----------------|-----------|
| Blocking Probability | BP per erlang level | 2% |
| Throughput | Total Gbps served | 2% |
| Modulation Distribution | % requests per modulation | 5% |
| Average Path Length | Weighted average hops | 1% |
| Spectrum Utilization | % slots used at end | 5% |
| Lightpath Count | Total lightpaths created | 5% |

### Snapshot File Format

```json
{
  "config": "plain_ksp",
  "seed": 42,
  "version": "pre-phase-6",
  "metrics": {
    "blocking_probability": 0.0523,
    "throughput_gbps": 45230.0,
    "modulation_distribution": {
      "QPSK": 0.45,
      "16-QAM": 0.35,
      "64-QAM": 0.20
    },
    "average_path_length_hops": 3.2,
    "spectrum_utilization": 0.67,
    "lightpath_count": 1523
  }
}
```

### Snapshot Commands

```bash
# Capture golden snapshots (run before Phase 6)
python tests/capture_snapshots.py --output tests/snapshots/pre_phase6/

# Compare against snapshots (run during/after Phase 6)
python tests/compare_snapshots.py --baseline tests/snapshots/pre_phase6/ --tolerance 0.02

# Update snapshots (after verifying changes are intentional)
python tests/capture_snapshots.py --output tests/snapshots/phase6/
```

### Snapshot Test Implementation

```python
# tests/compare_snapshots.py (conceptual)

import json
from pathlib import Path

def compare_snapshots(baseline_dir: Path, current_results: dict, tolerance: float) -> list[str]:
    """Compare current results against baseline snapshots."""
    failures = []

    for config_file in baseline_dir.glob("*.json"):
        config_name = config_file.stem
        baseline = json.loads(config_file.read_text())

        if config_name not in current_results:
            failures.append(f"Missing results for config: {config_name}")
            continue

        current = current_results[config_name]

        # Compare key metrics
        for metric in ["blocking_probability", "throughput_gbps"]:
            baseline_val = baseline["metrics"][metric]
            current_val = current["metrics"][metric]
            diff = abs(baseline_val - current_val) / max(baseline_val, 0.001)

            if diff > tolerance:
                failures.append(
                    f"{config_name}.{metric}: baseline={baseline_val:.4f}, "
                    f"current={current_val:.4f}, diff={diff:.2%}"
                )

    return failures
```

---

## CI Gating for Phase 6 PRs

### PR Labels

Phase 6 PRs must be labeled with:
- `destructive-migration`
- `phase-6`

### Required CI Checks

```yaml
# .github/workflows/phase6-gate.yml

name: Phase 6 Gate

on:
  pull_request:
    labels:
      - destructive-migration
      - phase-6

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Domain model tests
        run: pytest fusion/tests/domain/ -v --cov=fusion/domain --cov-fail-under=90

      - name: Pipeline tests
        run: pytest fusion/tests/pipelines/ -v --cov=fusion/pipelines --cov-fail-under=85

      - name: Orchestrator tests
        run: pytest fusion/tests/core/test_orchestrator.py -v --cov-fail-under=90

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Integration tests
        run: pytest fusion/tests/integration/ -v

      - name: RL integration tests
        run: pytest fusion/tests/rl/ fusion/tests/policies/ -v

  regression-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Regression comparison
        run: python tests/run_comparison.py --baseline-mode --all-configs --seed 42

      - name: Snapshot comparison
        run: python tests/compare_snapshots.py --baseline tests/snapshots/pre_phase6/ --tolerance 0.02

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Type check
        run: mypy fusion/ --ignore-missing-imports

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint check
        run: ruff check fusion/
```

### Branch Protection

For Phase 6, require:
- All CI checks passing
- At least 2 reviewer approvals
- No merge until all reviewers approve

---

## Test Removal Strategy

As legacy code is removed, associated tests must also be removed or updated.

### Tests to Delete

| Test File | Reason | When to Delete |
|-----------|--------|----------------|
| Tests for Props classes | Props classes removed | P6.1 |
| Tests for `mock_handle_arrival` | Function removed | P6.1 |
| Tests for adapter classes | Adapters removed | P6.2 |
| Tests for `to_legacy_dict` | Methods removed | P6.2 |
| Tests for old SDNController | Class removed | P6.3 |
| Dual-path comparison tests | Single path only | P6.3 |

### Tests to Update

| Test File | Required Update | When |
|-----------|-----------------|------|
| `run_comparison.py` | Convert from dual-path to baseline mode | P6.3 |
| Integration tests | Remove legacy path branches | P6.2 |
| RL tests | Remove old env references | P6.2 |

---

## Verification Checklist per Micro-Phase

### P6.1 Verification

```bash
# Before P6.1
pytest fusion/tests/ -v  # Establish baseline

# After each P6.1 removal
grep -r "<removed_item>" fusion/ --include="*.py"  # Verify removal
pytest fusion/tests/ -v  # All tests pass
python tests/run_comparison.py --baseline-mode  # Regression pass
```

### P6.2 Verification

```bash
# After P6.2
ls fusion/core/adapters/  # Should not exist
grep -r "use_orchestrator" fusion/  # Should return nothing
pytest fusion/tests/ -v
python tests/run_comparison.py --baseline-mode --all-configs
```

### P6.3 Verification

```bash
# After P6.3
ls fusion/core/sdn_controller.py  # Should not exist
ls fusion/core/properties.py  # Should not exist
cd docs && make html  # Docs build without warnings
pytest fusion/tests/ -v --cov=fusion --cov-report=html
python tests/run_comparison.py --baseline-mode --all-configs
```

---

## Rollback Testing

If a Phase 6 change must be rolled back:

1. Revert the PR
2. Run full test suite to verify restoration
3. Run regression tests to verify parity restored

```bash
# After revert
git checkout main
git pull
pytest fusion/tests/ -v
python tests/run_comparison.py --compare-paths --all-configs  # Dual-path mode
```

---

## Related Documents

- [Test Strategy](./test_strategy.md) - overall testing philosophy
- [Phase 6 Legacy Removal](../migration/phase_6_legacy_removal.md) - what's being removed
- [ADR-0013: Legacy Removal](../decisions/0013-legacy-removal.md) - decision rationale
