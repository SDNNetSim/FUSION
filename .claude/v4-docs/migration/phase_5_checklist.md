# Phase 5 Checklist: ML Control + Protection Integration

## Prerequisites

- [ ] Phase 4 complete and verified
- [ ] UnifiedSimEnv functional
- [ ] RLSimulationAdapter tested
- [ ] `run_comparison.py` passes with RL path
- [ ] All Phase 4 tests passing

---

## P5.1: ControlPolicy Protocol

### Files to Create
- [ ] `fusion/interfaces/control_policy.py`
- [ ] `fusion/policies/__init__.py`

### Implementation Tasks
- [ ] Define `ControlPolicy` protocol with `select_action()` and `update()`
- [ ] Make protocol `@runtime_checkable`
- [ ] Add type hints for all parameters and return types
- [ ] Add docstrings with usage examples

### Tests to Create
- [ ] `fusion/tests/interfaces/test_control_policy.py`
- [ ] Test protocol can be implemented
- [ ] Test `isinstance()` checks work with `@runtime_checkable`

### Verification
```bash
pytest fusion/tests/interfaces/test_control_policy.py -v
mypy fusion/interfaces/control_policy.py --strict
ruff check fusion/interfaces/control_policy.py
```

---

## P5.2: Heuristic Policies

### Files to Create
- [ ] `fusion/policies/heuristic_policy.py`

### Policies to Implement
- [ ] `HeuristicPolicy` abstract base class
- [ ] `FirstFeasiblePolicy`
- [ ] `ShortestFeasiblePolicy`
- [ ] `LeastCongestedPolicy`
- [ ] `RandomFeasiblePolicy`
- [ ] `LoadBalancedPolicy`
- [ ] `ModulationAwarePolicy` (optional)

### Implementation Checklist
- [ ] All policies implement `ControlPolicy` protocol
- [ ] `update()` is no-op for all heuristics
- [ ] All policies handle empty options list (return -1)
- [ ] All policies handle all-infeasible options (return -1)
- [ ] `RandomFeasiblePolicy` accepts seed parameter
- [ ] `LoadBalancedPolicy` accepts alpha parameter

### Tests to Create
- [ ] `fusion/tests/policies/test_heuristic_policy.py`
- [ ] Test each policy selects correct path
- [ ] Test behavior with no feasible options
- [ ] Test determinism (same input = same output, except Random)
- [ ] Test RandomFeasiblePolicy seed reproducibility

### Verification
```bash
pytest fusion/tests/policies/test_heuristic_policy.py -v
mypy fusion/policies/heuristic_policy.py
```

---

## P5.3: ML Policy Support

### Files to Create
- [ ] `fusion/policies/ml_policy.py`

### Implementation Checklist
- [ ] `MLControlPolicy` class implementing `ControlPolicy`
- [ ] Support PyTorch model loading
- [ ] Support scikit-learn model loading (optional)
- [ ] Support ONNX model loading (optional)
- [ ] Feature extraction method `_build_features()`
- [ ] Action masking to respect feasibility
- [ ] Fallback behavior when model fails
- [ ] Device configuration (cpu/cuda)

### Tests to Create
- [ ] `fusion/tests/policies/test_ml_policy.py`
- [ ] Test with mock model
- [ ] Test action masking works correctly
- [ ] Test fallback to first feasible on error
- [ ] Test device placement

### Verification
```bash
pytest fusion/tests/policies/test_ml_policy.py -v
mypy fusion/policies/ml_policy.py
```

---

## P5.4: Protection Pipeline

### Files to Create
- [ ] `fusion/pipelines/protection_pipeline.py`

### Files to Modify
- [ ] `fusion/domain/lightpath.py` - Add protection fields
- [ ] `fusion/domain/network_state.py` - Add protected methods
- [ ] `fusion/domain/results.py` - Add backup path fields

### Lightpath Extensions
- [ ] Add `backup_path: Optional[list[str]]`
- [ ] Add `is_protected: bool`
- [ ] Add `active_path: str` ("primary" or "backup")
- [ ] Add `current_path` property

### NetworkState Extensions
- [ ] Add `create_protected_lightpath()` method
- [ ] Add `get_lightpaths_on_link()` method
- [ ] Allocate spectrum on both paths in protected creation

### RouteResult Extensions
- [ ] Add `backup_paths: Optional[list[list[str]]]`
- [ ] Add `backup_weights_km: Optional[list[float]]`
- [ ] Add `backup_modulations: Optional[list[list[str | None]]]`

### ProtectionPipeline Implementation
- [ ] `find_protected_routes()` using NetworkX disjoint paths
- [ ] Support link-disjoint and node-disjoint modes
- [ ] `handle_failure()` for switchover
- [ ] Configurable switchover time

### Tests to Create
- [ ] `fusion/tests/pipelines/test_protection_pipeline.py`
- [ ] Test disjoint path finding
- [ ] Test no protection available (< 2 disjoint paths)
- [ ] Test failure switchover
- [ ] Test spectrum allocation on both paths

### Verification
```bash
pytest fusion/tests/pipelines/test_protection_pipeline.py -v
pytest fusion/tests/domain/test_lightpath.py::test_protection -v
pytest fusion/tests/domain/test_network_state.py::test_protected -v
mypy fusion/pipelines/protection_pipeline.py
```

---

## P5.5: Policy Factory

### Files to Create
- [ ] `fusion/policies/policy_factory.py`

### Files to Modify
- [ ] `fusion/core/pipeline_factory.py`

### PolicyFactory Implementation
- [ ] Factory method `create()` for all policy types
- [ ] Registry pattern for extensibility
- [ ] `register()` method for custom policies

### PipelineFactory Updates
- [ ] Add `create_policy()` method
- [ ] Add `create_routing()` updates for protection
- [ ] Add `create_spectrum()` updates for protected spectrum

### Tests to Create
- [ ] `fusion/tests/policies/test_policy_factory.py`
- [ ] Test creation of each policy type
- [ ] Test invalid policy type raises error
- [ ] Test custom policy registration

### Verification
```bash
pytest fusion/tests/policies/test_policy_factory.py -v
pytest fusion/tests/core/test_pipeline_factory.py -v
```

---

## P5.6: Orchestrator Integration

### Files to Modify
- [ ] `fusion/core/orchestrator.py`

### Implementation Checklist
- [ ] Add `policy` parameter to `__init__`
- [ ] Default to `FirstFeasiblePolicy` if not provided
- [ ] Add `handle_arrival_with_policy()` method
- [ ] Integrate with `RLSimulationAdapter`
- [ ] Call `policy.update()` after allocation

### Tests to Create
- [ ] `fusion/tests/core/test_orchestrator.py::TestPolicyIntegration`
- [ ] Test with each heuristic policy
- [ ] Test with mock ML policy
- [ ] Test policy fallback behavior

### Verification
```bash
pytest fusion/tests/core/test_orchestrator.py::TestPolicyIntegration -v
```

---

## P5.7: Configuration

### Files to Modify
- [ ] `fusion/configs/templates/default.ini`

### Files to Create
- [ ] `fusion/cli/parameters/policy.py`

### Configuration Sections
- [ ] Add `[policy]` section
- [ ] Add `[protection]` section
- [ ] Add `[heuristic]` section (optional parameters)

### Configuration Options
- [ ] `policy.type` - Policy type selection
- [ ] `policy.model_path` - Path for ML/RL models
- [ ] `policy.device` - Device for inference
- [ ] `protection.enabled` - Enable 1+1 protection
- [ ] `protection.disjointness` - link or node disjoint
- [ ] `protection.switchover_time_ms` - Switchover latency

### Tests
- [ ] Test configuration parsing
- [ ] Test default values

### Verification
```bash
pytest fusion/tests/configs/test_policy_config.py -v
```

---

## Integration Tests

### Test Files
- [ ] `tests/integration/test_policy_simulation.py`
- [ ] `tests/integration/test_protection_simulation.py`

### Integration Test Scenarios
- [ ] Run simulation with FirstFeasiblePolicy
- [ ] Run simulation with ShortestFeasiblePolicy
- [ ] Run simulation with LeastCongestedPolicy
- [ ] Run simulation with protection enabled
- [ ] Verify protection switchover on failure

### Verification
```bash
pytest tests/integration/test_policy_simulation.py -v
pytest tests/integration/test_protection_simulation.py -v
```

---

## Documentation

### Architecture Docs
- [ ] `architecture/control_policy.md`
- [ ] `architecture/protection_pipeline.md`
- [ ] `architecture/ml_policies.md`
- [ ] `architecture/heuristic_policies.md`

### Migration Docs
- [ ] `migration/phase_5_ml_protection.md`
- [ ] `migration/phase_5_checklist.md` (this file)

### Decision Records
- [ ] `decisions/0010-control-policy-protocol.md`
- [ ] `decisions/0011-ml-control-policy.md`
- [ ] `decisions/0012-protection-pipeline.md`

### Testing Docs
- [ ] `testing/phase_5_testing.md`

### Tutorials
- [ ] `tutorials/implementing_a_new_policy.md`

---

## Final Verification

### All Tests Pass
```bash
# Unit tests
pytest fusion/tests/interfaces/test_control_policy.py -v
pytest fusion/tests/policies/ -v
pytest fusion/tests/pipelines/test_protection_pipeline.py -v
pytest fusion/tests/core/test_orchestrator.py::TestPolicyIntegration -v

# Integration tests
pytest tests/integration/test_policy_simulation.py -v
pytest tests/integration/test_protection_simulation.py -v
```

### Type Checking
```bash
mypy fusion/interfaces/control_policy.py
mypy fusion/policies/
mypy fusion/pipelines/protection_pipeline.py
```

### Lint Checking
```bash
ruff check fusion/interfaces/control_policy.py
ruff check fusion/policies/
ruff check fusion/pipelines/protection_pipeline.py
```

### Comparison Test
```bash
python tests/run_comparison.py --policy first_feasible
python tests/run_comparison.py --protection enabled
```

---

## Exit Criteria Summary

- [ ] ControlPolicy protocol defined and tested
- [ ] All 5+ heuristic policies implemented
- [ ] MLControlPolicy functional
- [ ] Protection pipeline complete
- [ ] Policy factory creates all types
- [ ] Orchestrator integrates policies
- [ ] Configuration documented
- [ ] All tests pass
- [ ] Type checking passes
- [ ] Lint checking passes
- [ ] `run_comparison.py` passes
- [ ] Documentation complete
- [ ] No Phase 4 regressions
