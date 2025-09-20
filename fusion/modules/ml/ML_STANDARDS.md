# Machine Learning Module Standards

This document outlines the standards and guidelines specific to the FUSION ML module. These standards complement the project-wide CODING_STANDARDS.md and provide ML-specific guidance.

## Table of Contents
1. [Module Structure](#module-structure)
2. [Data Conventions](#data-conventions)
3. [Model Interface Standards](#model-interface-standards)
4. [Visualization Standards](#visualization-standards)
5. [Evaluation Metrics](#evaluation-metrics)
6. [File Organization](#file-organization)

## Module Structure

The ML module is organized by functional responsibility:

```
modules/ml/
├── __init__.py              # Public API exports
├── ML_STANDARDS.md          # This file
├── registry.py              # Model registry and factory
├── visualization.py         # All plotting and visualization
├── preprocessing.py         # Data preparation and transformation
├── feature_engineering.py   # Feature creation and extraction
├── model_io.py             # Model persistence (save/load)
└── evaluation.py           # Metrics and model evaluation
```

### Design Principles
- **Separation of Concerns**: Each file handles one specific aspect of ML workflow
- **Extensibility**: Easy to add new models, metrics, or visualizations
- **Framework Agnostic**: Support multiple ML frameworks (sklearn, tensorflow, pytorch)

## Data Conventions

### DataFrame Standards
- Always use explicit column names, never rely on positional indexing
- Preserve original data where possible, create copies for transformations
- Use descriptive suffixes for transformed data:
  ```python
  data_normalized = normalize_data(data_raw)
  data_encoded = encode_categorical(data_normalized)
  ```

### Variable Naming for ML
- Use full descriptive names with appropriate suffixes:
  - `_dataframe` or `_df` for pandas DataFrames
  - `_array` for numpy arrays
  - `_tensor` for PyTorch/TensorFlow tensors
  - `_model` for trained models
  - `_scaler` for preprocessing scalers

### Feature Naming
- Features should be self-documenting:
  ```python
  # Good
  'average_congestion_percent'
  'path_length_kilometers'
  'bandwidth_gigabits'
  
  # Avoid
  'avg_cong'
  'len'
  'bw'
  ```

## Model Interface Standards

### Model Functions
Every model integration should provide these standard functions:

```python
def train_model(
    training_data: pd.DataFrame,
    target_column: str,
    model_parameters: Dict[str, Any],
    random_seed: int = 42
) -> Tuple[Any, Dict[str, float]]:
    """Train a model and return both model and metrics."""
    pass

def predict(
    model: Any,
    input_data: pd.DataFrame
) -> np.ndarray:
    """Make predictions using trained model."""
    pass

def evaluate_model(
    model: Any,
    test_data: pd.DataFrame,
    target_column: str
) -> Dict[str, float]:
    """Evaluate model performance."""
    pass
```

### Model Registry
- All models must be registered in `registry.py`
- Use descriptive names that indicate the algorithm and purpose
- Include model metadata (requirements, limitations, use cases)

## Visualization Standards

### Plot Requirements
- All plots must include:
  - Clear titles with context (e.g., "Feature Importance - 1000 Erlang Traffic")
  - Labeled axes with units where applicable
  - Legends when multiple series are present
  - High DPI (300) for publication quality

### Color Schemes
- Use colorblind-friendly palettes
- Maintain consistency across related plots
- Default palette: `seaborn.color_palette("colorblind")`

### File Naming for Plots
```python
# Pattern: {plot_type}_{metric}_{context}.png
'confusion_matrix_1000erlang.png'
'feature_importance_randomforest_500erlang.png'
'learning_curve_epoch50.png'
```

## Evaluation Metrics

### Standard Metrics
For classification tasks, always report:
- Accuracy (overall and per-class)
- Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC when applicable

### Metric Formatting
```python
metrics_dict = {
    'accuracy': 0.9234,           # 4 decimal places
    'precision': 0.9156,
    'recall': 0.9089,
    'f1_score': 0.9122,
    'confusion_matrix': array,    # Keep as numpy array
    'per_class_accuracy': {       # Class-specific metrics
        'class_1': 0.9456,
        'class_2': 0.8934,
    }
}
```

## File Organization

### Import Organization
```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Data science libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML frameworks (as needed)
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch

# FUSION imports
from fusion.utils.logging_config import get_logger
from fusion.core.metrics import MLMetrics
```

### Function Organization Priority
Within each file, organize functions as:
1. Public API functions
2. Helper functions (private)
3. Validation/checking functions
4. Deprecated functions (clearly marked)

## Testing Standards

### Test Coverage Requirements
- All public functions must have unit tests
- Test edge cases (empty data, single sample, missing values)
- Test model save/load cycle integrity
- Validate metric calculations against known results

### Test Data
- Use small, synthetic datasets for unit tests
- Store test fixtures in `tests/fixtures/ml/`
- Document expected outputs for regression tests

## Performance Considerations

### Memory Management
- Use generators for large dataset processing
- Clean up large objects explicitly when done
- Monitor memory usage for model training

### Computational Efficiency
- Profile expensive operations
- Cache computed features when appropriate
- Use vectorized operations over loops
- Consider batch processing for large datasets

## Documentation Requirements

### Function Documentation
ML functions require additional documentation:

```python
def train_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
    algorithm: str = "random_forest"
) -> Tuple[Any, Dict[str, float]]:
    """Train a classification model.
    
    This function supports multiple algorithms and automatically
    handles class imbalance through SMOTE or class weights.
    
    :param features: Feature matrix with shape (n_samples, n_features)
    :type features: pd.DataFrame
    :param labels: Target labels with shape (n_samples,)
    :type labels: pd.Series
    :param algorithm: Algorithm choice from ['random_forest', 'svm', 'xgboost']
    :type algorithm: str
    :return: Trained model and evaluation metrics
    :rtype: Tuple[Any, Dict[str, float]]
    
    :raises ValueError: If algorithm not supported
    :raises DataError: If features/labels shape mismatch
    
    Example:
        >>> features = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
        >>> labels = pd.Series([0, 1])
        >>> model, metrics = train_classifier(features, labels)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
```

### Algorithm Documentation
Each algorithm implementation should include:
- Time complexity
- Space complexity
- Assumptions and limitations
- Recommended use cases
- Parameter tuning guidelines

## Integration Guidelines

### Adding New Models
1. Create model class/functions following interface standards
2. Register in `registry.py`
3. Add comprehensive tests
4. Document parameters and usage
5. Add example notebook if complex

### Extending Functionality
When adding new functionality:
- Check if it belongs in existing files or needs new module
- Follow the single responsibility principle
- Ensure backward compatibility
- Update relevant documentation

---

*These standards are specific to the ML module. Always defer to project-wide CODING_STANDARDS.md for general guidelines.*