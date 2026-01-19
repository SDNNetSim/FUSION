# Phase 3 Migration Checklist

This document provides step-by-step checklists for completing Phase 3: SDNOrchestrator as Thin Coordination Layer.

## Overview

Phase 3 introduces `SDNOrchestrator` as a thin layer that coordinates pipeline execution without containing algorithm logic. This phase is divided into five micro-phases:

| Micro-Phase | Focus | Key Deliverable |
|-------------|-------|-----------------|
| P3.1 | PipelineFactory | Pipeline instantiation |
| P3.2 | SDNOrchestrator skeleton | Basic request flow |
| P3.3 | Feature integration | Grooming, slicing, SNR |
| P3.4 | Dual-path verification | Legacy vs new comparison |
| P3.5 | Legacy deprecation | Switch to orchestrator |

---

## P3.1: PipelineFactory Implementation

### Objective
Create factory that instantiates pipelines based on configuration.

### Implementation Checklist

#### Core Factory
- [ ] Create `fusion/core/pipeline_factory.py`
- [ ] Implement `PipelineFactory` class
- [ ] Create `PipelineSet` dataclass to hold pipeline references

#### Factory Methods
- [ ] `create_routing(config) -> RoutingPipeline`
- [ ] `create_spectrum(config) -> SpectrumPipeline`
- [ ] `create_grooming(config) -> GroomingPipeline | None`
- [ ] `create_snr(config) -> SNRPipeline | None`
- [ ] `create_slicing(config) -> SlicingPipeline | None`
- [ ] `create_pipeline_set(config) -> PipelineSet`
- [ ] `create_orchestrator(config) -> SDNOrchestrator`

#### Factory Logic
- [ ] Routing: Select based on `config.route_method`
- [ ] Spectrum: Select based on `config.allocation_method`
- [ ] Grooming: Return `None` if `config.grooming_enabled == False`
- [ ] SNR: Return `None` if `config.snr_enabled == False`
- [ ] Slicing: Return `None` if `config.slicing_enabled == False`

### Verification Commands

```bash
# Run factory tests
pytest fusion/tests/core/test_pipeline_factory.py -v

# Verify factory creates correct types
python -c "
from fusion.core.pipeline_factory import PipelineFactory
from fusion.domain.config import SimulationConfig

config = SimulationConfig(...)  # minimal config
pipelines = PipelineFactory.create_pipeline_set(config)
assert pipelines.routing is not None
assert pipelines.spectrum is not None
print('Factory verification passed')
"
```

### Completion Criteria
- [ ] Factory creates correct pipeline types based on config
- [ ] Optional pipelines return `None` when disabled
- [ ] All pipeline types implement their protocols
- [ ] Unit tests cover all factory methods

---

## P3.2: SDNOrchestrator Skeleton

### Objective
Create basic orchestrator with plain KSP flow (no optional features).

### Implementation Checklist

#### Core Structure
- [ ] Create `fusion/core/orchestrator.py`
- [ ] `SDNOrchestrator.__init__(config, pipelines)`
- [ ] Store pipeline references (NOT NetworkState)
- [ ] No algorithm logic in orchestrator

#### Basic Flow Methods
- [ ] `handle_arrival(request, network_state, forced_path?) -> AllocationResult`
- [ ] `handle_release(request, network_state) -> None`
- [ ] `_try_allocate_on_path(request, path, mods, weight, bw, network_state) -> AllocationResult | None`

#### Plain KSP Flow
```
1. RoutingPipeline.find_routes(src, dst, bw, network_state)
2. For each path:
   a. SpectrumPipeline.find_spectrum(path, mods, bw, network_state)
   b. If is_free: NetworkState.create_lightpath(...)
   c. Return AllocationResult(success=True)
3. If all fail: Return AllocationResult(success=False, block_reason=NO_SPECTRUM)
```

#### Size Constraints
- [ ] Total lines in orchestrator.py < 200
- [ ] No method > 50 lines
- [ ] No algorithm implementations

### Verification Commands

```bash
# Run orchestrator unit tests
pytest fusion/tests/core/test_orchestrator.py::TestOrchestratorPlainKSP -v

# Verify size constraints
wc -l fusion/core/orchestrator.py
# Should be < 200

# Count longest method
python -c "
import ast
with open('fusion/core/orchestrator.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        lines = node.end_lineno - node.lineno + 1
        print(f'{node.name}: {lines} lines')
"
```

### Completion Criteria
- [ ] Plain KSP flow works end-to-end
- [ ] Orchestrator passes size constraints
- [ ] No algorithm logic in orchestrator
- [ ] Unit tests with mocked pipelines pass

---

## P3.3: Feature Integration

### Objective
Add grooming, slicing, and SNR to orchestrator flow.

### Implementation Checklist

#### Grooming Integration
- [ ] Add grooming stage before routing
- [ ] Handle `GroomingResult.fully_groomed` -> skip routing
- [ ] Handle `GroomingResult.partially_groomed` -> continue with remaining bw
- [ ] Handle `forced_path` from grooming
- [ ] Implement rollback on subsequent failure

#### Slicing Integration
- [ ] Add slicing fallback after spectrum failure
- [ ] `SlicingPipeline.try_slice()` creates multiple lightpaths
- [ ] Combine sliced lightpaths in result
- [ ] Handle partial slice failure (rollback all)

#### SNR Integration
- [ ] Add SNR validation after lightpath creation
- [ ] On SNR failure: release lightpath, try next slot/path
- [ ] `SNRPipeline.validate()` after create
- [ ] `SNRPipeline.validate_protected()` for 1+1

#### Combined Flows
- [ ] Grooming + Routing + Spectrum
- [ ] Grooming + Routing + Spectrum + SNR
- [ ] Routing + Spectrum + Slicing
- [ ] Routing + Spectrum + Slicing + SNR
- [ ] Grooming + Routing + Spectrum + Slicing + SNR

### Flow Diagram
```
Request
    |
    v
[Grooming enabled?]--no--> RoutingPipeline.find_routes()
    |                              |
    yes                            v
    |                      For each path:
    v                              |
GroomingPipeline.try_groom()       v
    |                      SpectrumPipeline.find_spectrum()
    |                              |
    +--fully_groomed---> Return    +--is_free=True
    |                              |       |
    +--partially_groomed           |       v
            |                      | NetworkState.create_lightpath()
            v                      |       |
    Continue with                  |       v
    remaining bw,                  | [SNR enabled?]--no--> Return
    forced_path                    |       |
                                   |      yes
                                   |       |
                                   |       v
                                   | SNRPipeline.validate()
                                   |       |
                                   |       +--fail--> release, try next
                                   |       |
                                   |       +--pass--> Return
                                   |
                                   +--is_free=False
                                           |
                                           v
                                   [Slicing enabled?]--no--> try next path
                                           |
                                          yes
                                           |
                                           v
                                   SlicingPipeline.try_slice()
```

### Verification Commands

```bash
# Run feature-specific tests
pytest fusion/tests/core/test_orchestrator.py::TestOrchestratorGrooming -v
pytest fusion/tests/core/test_orchestrator.py::TestOrchestratorSlicing -v
pytest fusion/tests/core/test_orchestrator.py::TestOrchestratorSNR -v

# Run integration tests
pytest fusion/tests/integration/test_orchestrator_flows.py -v

# Verify orchestrator size still under limit
wc -l fusion/core/orchestrator.py
```

### Completion Criteria
- [ ] All feature combinations work correctly
- [ ] Rollback handled properly on failures
- [ ] Orchestrator still < 200 lines
- [ ] Each feature adds < 20 lines to orchestrator
- [ ] Integration tests pass

---

## P3.4: Dual-Path Verification

### Objective
Run simulations through both legacy SDNController and new SDNOrchestrator, comparing results.

### Implementation Checklist

#### Feature Flag
- [ ] Add `config.use_orchestrator: bool` flag
- [ ] `SimulationEngine` selects path based on flag
- [ ] Both paths use same `NetworkState` instance

#### Comparison Infrastructure
- [ ] Update `tests/run_comparison.py`
- [ ] Add `--compare-paths` mode
- [ ] Compare key metrics: blocking probability, throughput, modulations
- [ ] Tolerance: `abs_tol=0.02` for statistical variations

#### Test Configurations
- [ ] `plain_ksp` - Basic K-shortest path
- [ ] `ksp_with_grooming` - Grooming enabled
- [ ] `ksp_with_snr` - SNR validation enabled
- [ ] `ksp_with_slicing` - Slicing enabled
- [ ] `ksp_with_all_features` - All features combined

### Comparison Script

```python
# tests/run_comparison.py

def compare_paths():
    configs = [
        "plain_ksp",
        "ksp_with_grooming",
        "ksp_with_snr",
        "ksp_with_slicing",
        "ksp_with_all_features",
    ]

    for config_name in configs:
        print(f"Testing {config_name}...")

        # Run legacy path
        legacy = run_simulation(config_name, use_orchestrator=False)

        # Run orchestrator path
        new = run_simulation(config_name, use_orchestrator=True)

        # Compare
        assert_metrics_equal(legacy, new, tolerance=0.02)
        print(f"  PASS: blocking_prob legacy={legacy['bp']:.4f}, new={new['bp']:.4f}")
```

### Verification Commands

```bash
# Run dual-path comparison
python tests/run_comparison.py --compare-paths

# Run with verbose output
python tests/run_comparison.py --compare-paths --verbose

# Run specific config
python tests/run_comparison.py --compare-paths --config ksp_with_grooming

# Run with seed for reproducibility
python tests/run_comparison.py --compare-paths --seed 42
```

### Completion Criteria
- [ ] All configs produce matching results (within tolerance)
- [ ] No metric differs by more than 2%
- [ ] Both paths use identical RNG state
- [ ] Comparison runs in CI

---

## P3.5: Legacy Deprecation

### Objective
Switch default to orchestrator path, deprecate legacy SDNController.

### Implementation Checklist

#### Default Switch
- [ ] Change `config.use_orchestrator` default to `True`
- [ ] Add deprecation warning when `use_orchestrator=False`
- [ ] Update documentation to reflect new default

#### Deprecation Notices
- [ ] Add `@deprecated` decorator to legacy methods
- [ ] Log warning on legacy path usage
- [ ] Add migration timeline to deprecation message

#### Legacy Code Isolation
- [ ] Move legacy code to `fusion/legacy/` directory
- [ ] Keep minimal imports from main code
- [ ] Document how to use legacy path if needed

#### Documentation Updates
- [ ] Update README with new architecture
- [ ] Update configuration examples
- [ ] Update tutorials to use orchestrator pattern

### Verification Commands

```bash
# Verify default is orchestrator
python -c "
from fusion.domain.config import SimulationConfig
config = SimulationConfig(...)
assert config.use_orchestrator == True, 'Default should be True'
"

# Check for deprecation warnings
python -W all -c "
from fusion.core.sdn_controller import SDNController
# Should emit deprecation warning
"

# Final regression run
python tests/run_comparison.py --all-configs
```

### Completion Criteria
- [ ] Orchestrator is default path
- [ ] Legacy path emits deprecation warning
- [ ] All documentation updated
- [ ] CI runs orchestrator path by default
- [ ] No regressions in any config

---

## Rollback Procedures

### If P3.2 Orchestrator Has Issues
```bash
# Disable orchestrator, use legacy
export FUSION_USE_LEGACY_SDN=1
# Or in config:
# use_orchestrator = false
```

### If P3.3 Feature Integration Breaks
```bash
# Disable specific feature
# In config:
# grooming_enabled = false
# slicing_enabled = false
# snr_enabled = false
```

### If P3.4 Shows Discrepancies
```bash
# Run with detailed logging
python tests/run_comparison.py --compare-paths --debug

# Check specific request handling
python tests/debug_request.py --request-id 42 --both-paths
```

### If P3.5 Causes Production Issues
```bash
# Revert to legacy default
git checkout HEAD~1 -- fusion/domain/config.py
# Set use_orchestrator = false
```

---

## Progress Tracking

| Task | Status | Verified By | Date |
|------|--------|-------------|------|
| P3.1: PipelineFactory complete | | | |
| P3.1: Unit tests pass | | | |
| P3.2: Orchestrator skeleton | | | |
| P3.2: Plain KSP flow works | | | |
| P3.2: Size constraints met | | | |
| P3.3: Grooming integrated | | | |
| P3.3: Slicing integrated | | | |
| P3.3: SNR integrated | | | |
| P3.3: Combined flows work | | | |
| P3.4: Dual-path comparison | | | |
| P3.4: All configs match | | | |
| P3.5: Default switched | | | |
| P3.5: Deprecation added | | | |
| P3.5: Docs updated | | | |

---

## Performance Benchmarks

Track these metrics before/after orchestrator switch:

| Metric | Legacy | Orchestrator | Delta |
|--------|--------|--------------|-------|
| Requests/second | | | |
| Avg handle_arrival time | | | |
| Memory usage (MB) | | | |
| GC pause time (ms) | | | |

### Benchmark Script

```bash
# Run performance comparison
python tests/benchmark.py --legacy
python tests/benchmark.py --orchestrator
python tests/benchmark.py --compare
```

---

## Related Documentation

- [Architecture: Orchestration](../architecture/orchestration.md)
- [Architecture: Pipeline Interfaces](../architecture/pipeline_interfaces.md)
- [Architecture: Pipeline Walkthroughs](../architecture/pipeline_walkthroughs.md)
- [Testing: Phase 3 Testing](../testing/phase_3_testing.md)
- [ADR-0007: Orchestrator Design](../decisions/0007-orchestrator-design.md)
