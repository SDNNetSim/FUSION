# Machine Learning Module

> **Status: Beta**
>
> This module provides ML utilities for supervised/unsupervised learning. It is actively
> used by the legacy simulation path. Orchestrator integration is planned for a future version.

## Overview

The Machine Learning module provides utilities for:
- Feature engineering from network states
- Data preprocessing and transformation
- Model persistence (save/load)
- Model evaluation and comparison
- Visualization of results

**Note**: This module provides ML *utilities*, not ML algorithms. You bring your own
models (sklearn, tensorflow, pytorch, etc.) and use these utilities around them.

## Current Integration

| Path | Status | How It's Used |
|------|--------|---------------|
| Legacy (`use_orchestrator=False`) | **Works** | `get_ml_obs()` for features, `load_model()` for models |
| Orchestrator (`use_orchestrator=True`) | Not yet | Planned for v6.x |

Enable ML model deployment with `deploy_model=True` in engine_props.

## Relationship with MLControlPolicy

There are currently **two separate** ML-related areas in FUSION:

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| **ML Utilities** (this module) | `fusion/modules/ml/` | Feature extraction, preprocessing, model I/O, evaluation | Used by legacy path |
| **ML Control Policy** | `fusion/policies/ml_policy.py` | Path selection using pre-trained models | Used by orchestrator |

**Current State (Not Yet Integrated):**

These two components were built independently and do not share code:

- `fusion/modules/ml/` has `extract_ml_features()` that works with legacy dicts (`engine_props`, `sdn_props`)
- `fusion/policies/ml_policy.py` has its own `FeatureBuilder` that works with domain objects (`Request`, `NetworkState`)
- Both have separate model loading implementations

**Future Integration (Planned for v6.x):**

The goal is to unify these so that `MLControlPolicy` uses this module's utilities:

- `MLControlPolicy` would use this module's feature extraction (adapted for domain objects)
- Model I/O would be consolidated into this module
- Evaluation and visualization utilities would be available for policy analysis

This will eliminate duplicate code and provide a single, well-tested ML utility layer for all paths.

## Module Structure

```
fusion/modules/ml/
├── __init__.py            # Public API exports
├── README.md              # This file
├── TODO.md                # Development roadmap
├── preprocessing.py       # Data preparation utilities
├── feature_engineering.py # Feature extraction and creation
├── model_io.py            # Model saving/loading
├── evaluation.py          # Model evaluation metrics
├── visualization.py       # Plotting and visualization
├── constants.py           # Shared constants
└── registry.py            # Model registry (currently empty)
```

## Quick Start

```python
from fusion.modules.ml import (
    # Feature engineering
    extract_ml_features,
    create_interaction_features,

    # Preprocessing
    process_training_data,
    balance_training_data,
    normalize_features,

    # Model I/O
    save_model,
    load_model,

    # Evaluation
    evaluate_classifier,
    evaluate_regressor,

    # Visualization
    plot_confusion_matrix,
    plot_feature_importance,
)

# Extract features from network request
features = extract_ml_features(request_dict, engine_props, sdn_props)

# Process and balance training data
processed = process_training_data(sim_dict, raw_data, erlang)
balanced = balance_training_data(processed)

# Train your model (you provide this)
model = your_model.fit(balanced)

# Evaluate
metrics = evaluate_classifier(model, test_features, test_labels)

# Save for later
save_model(sim_dict, model, "algorithm_name", "1000")
```

## Key Conventions

### Naming Conventions

Use descriptive suffixes for ML variables:

```python
data_normalized = normalize_features(data_raw)  # Transformed data
features_df = extract_features(...)              # DataFrames: _df or _dataframe
predictions_array = model.predict(...)           # Arrays: _array
trained_model = train(...)                       # Models: _model
feature_scaler = StandardScaler()                # Scalers: _scaler
```

### Model Interface Pattern

When adding models to the registry, follow this interface:

```python
def train_model(
    training_data: pd.DataFrame,
    target_column: str,
    model_parameters: dict[str, Any],
    random_seed: int = 42
) -> tuple[Any, dict[str, float]]:
    """Train and return (model, metrics)."""
    ...

def predict(model: Any, input_data: pd.DataFrame) -> np.ndarray:
    """Make predictions."""
    ...

def evaluate_model(
    model: Any,
    test_data: pd.DataFrame,
    target_column: str
) -> dict[str, float]:
    """Return evaluation metrics."""
    ...
```

### Standard Metrics

For classification, always report:
- Accuracy (overall and per-class)
- Precision, Recall, F1-score
- Confusion matrix

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

Optional (for specific models):
- tensorflow
- pytorch

## Testing

```bash
# Run all ML module tests
pytest fusion/modules/tests/ml/ -v

# Run with coverage
pytest fusion/modules/tests/ml/ -v --cov=fusion.modules.ml
```

## Backward Compatibility

Old function names are aliased for compatibility:

| Old Name | New Name |
|----------|----------|
| `get_ml_obs` | `extract_ml_features` |
| `process_data` | `process_training_data` |
| `even_process_data` | `balance_training_data` |
| `plot_data` | `plot_data_distributions` |
| `plot_confusion` | `plot_confusion_matrix` |
