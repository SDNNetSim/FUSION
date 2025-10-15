# RL Policies Tests

This directory contains comprehensive tests for the RL policies module.

## Test Files

### `test_base_policies.py`
Tests for baseline heuristic policies:
- **KSPFFPolicy**: K-Shortest Path First-Fit policy
- **OnePlusOnePolicy**: 1+1 protection policy

### `test_action_masking.py`
Tests for action masking utilities:
- **compute_action_mask**: Feasibility mask computation
- **apply_fallback_policy**: Fallback policy application

### `test_rl_policies.py`
Tests for learned RL policies:
- **BCPolicy**: Behavior Cloning policy
- **IQLPolicy**: Implicit Q-Learning policy

## Running Tests

Run all RL policy tests:
```bash
pytest fusion/modules/rl/policies/tests/ -v
```

Run specific test file:
```bash
pytest fusion/modules/rl/policies/tests/test_base_policies.py -v
pytest fusion/modules/rl/policies/tests/test_action_masking.py -v
pytest fusion/modules/rl/policies/tests/test_rl_policies.py -v
```

Run with coverage:
```bash
pytest fusion/modules/rl/policies/tests/ -v --cov=fusion.modules.rl.policies
```

## Test Coverage

The test suite covers:
- Policy initialization and loading
- Path selection logic
- Action masking behavior
- Error handling (AllPathsMaskedError)
- Model loading from checkpoints
- State tensor conversion
- Edge cases and boundary conditions

## Fixtures

Tests use pytest fixtures for:
- **sample_state**: Standard state dictionary for testing
- **bc_model**: Temporary BC model file
- **iql_model**: Temporary IQL actor model file

## Dependencies

Tests require:
- pytest
- torch (PyTorch)
- fusion.modules.rl.policies

## Notes

- BC and IQL tests create temporary model files using pytest's `tmp_path` fixture
- All tests use CPU device to avoid GPU requirements
- Model architectures are simplified for testing purposes
