# Machine Learning Module TODOs

## Status: Beta

This module provides utilities for supervised/unsupervised learning. It is actively used by the legacy simulation path when `deploy_model=True`.

## High Priority

### MLControlPolicy Integration (v6.x)
- **Issue**: This module and `MLControlPolicy` were built independently and don't share code
- **Current State**:
  - `fusion/modules/ml/` has `extract_ml_features()`, `load_model()`, etc. for legacy path
  - `fusion/policies/ml_policy.py` has its own `FeatureBuilder` and `load_model()` for orchestrator
  - Both implement similar functionality but for different input types (legacy dicts vs domain objects)
- **Duplication**:
  - Feature extraction: `extract_ml_features()` vs `FeatureBuilder`
  - Model loading: `load_model()` in both places
- **Solution**: Unify so that `MLControlPolicy` uses this module's utilities
  - Adapt `extract_ml_features()` to work with domain objects (`Request`, `NetworkState`)
  - Consolidate model I/O into this module
  - Make evaluation/visualization utilities available for policy analysis
- **Target version**: v6.x

### Test Coverage
- **Issue**: Test coverage needs review
- **Files**: All files
- **Solution**: Ensure tests in `fusion/modules/tests/ml/` are up to date

## Medium Priority

### Visualization Integration
- **Issue**: ML visualization not yet integrated into central `fusion/visualization/` plugin system
- **Files**: `visualization.py`
- **Related**: `fusion/visualization/plugins/`, routing/spectrum/snr/rl plugins
- **Solution**: Create ML visualization plugin and integrate with central system

### Registry Implementation
- **Issue**: `registry.py` is empty - no models registered
- **Files**: `registry.py`
- **Solution**: Implement model registry for easier model management

## Low Priority

### Documentation Examples
- **Issue**: Could use more working examples
- **Solution**: Add Jupyter notebook examples

## Notes

- **Legacy path**: Uses `get_ml_obs()` for feature extraction, `load_model()` for loading trained models
- **Orchestrator path**: Does not currently use ML module (uses RL module), but integration is planned
- Related code: `fusion/core/ml_metrics.py` (training data collection)
- Enable with `deploy_model=True` in engine_props
