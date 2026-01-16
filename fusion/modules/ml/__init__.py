"""
Machine Learning module for FUSION.

This module provides machine learning utilities for supervised/unsupervised
learning in network optimization.

.. note::
    **Status: Beta** - This module is actively used by the legacy simulation path.
    Orchestrator integration is planned for a future version.

The module provides utilities for:

- Feature engineering from network states
- Data preprocessing and transformation
- Model persistence (save/load)
- Model evaluation metrics
- Visualization

**Current Integration:**

- Legacy path (use_orchestrator=False): Uses get_ml_obs() and load_model()
- Orchestrator path: Not yet integrated (planned for v6.x)

Enable with deploy_model=True in engine_props.

**Relationship with MLControlPolicy:**

This module (``fusion/modules/ml/``) and ``fusion/policies/ml_policy.py`` are
currently **separate implementations**:

- This module: utilities for legacy path (works with ``engine_props``, ``sdn_props``)
- MLControlPolicy: path selection for orchestrator (works with ``Request``, ``NetworkState``)

Both have their own feature extraction and model loading. The plan for v6.x is to
unify these so ``MLControlPolicy`` uses this module's utilities, eliminating duplication.

Note: This module provides ML utilities, not ML algorithms. You bring your own
models (sklearn, tensorflow, pytorch, etc.) and use these utilities around them.
"""

from fusion.modules.ml.evaluation import (
    analyze_prediction_errors,
    compare_models,
    cross_validate_model,
    evaluate_classifier,
    evaluate_model_stability,
    evaluate_regressor,
)

from fusion.modules.ml.feature_engineering import (
    create_interaction_features,
    create_polynomial_features,
    engineer_network_features,
    extract_ml_features,
)

from fusion.modules.ml.model_io import (
    check_model_compatibility,
    export_model_for_deployment,
    load_model,
    load_model_with_metadata,
    save_model,
    save_model_ensemble,
)

from fusion.modules.ml.preprocessing import (
    balance_training_data,
    normalize_features,
    prepare_prediction_features,
    process_training_data,
    split_features_labels,
)

from fusion.modules.ml.visualization import (
    plot_2d_clusters,
    plot_3d_clusters,
    plot_confusion_matrix,
    plot_data_distributions,
    plot_feature_importance,
)

# Backward compatibility aliases
# These maintain compatibility with existing code using old function names
plot_data = plot_data_distributions
plot_confusion = plot_confusion_matrix
get_ml_obs = extract_ml_features
process_data = process_training_data
even_process_data = balance_training_data

# Version info
__version__ = "1.0.0"

__all__ = [
    # Visualization
    "plot_data_distributions",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_2d_clusters",
    "plot_3d_clusters",
    # Preprocessing
    "process_training_data",
    "balance_training_data",
    "prepare_prediction_features",
    "normalize_features",
    "split_features_labels",
    # Feature Engineering
    "extract_ml_features",
    "create_interaction_features",
    "create_polynomial_features",
    "engineer_network_features",
    # Model I/O
    "save_model",
    "load_model",
    "load_model_with_metadata",
    "save_model_ensemble",
    "export_model_for_deployment",
    "check_model_compatibility",
    # Evaluation
    "evaluate_classifier",
    "evaluate_regressor",
    "cross_validate_model",
    "evaluate_model_stability",
    "compare_models",
    "analyze_prediction_errors",
    # Backward compatibility
    "plot_data",
    "plot_confusion",
    "get_ml_obs",
    "process_data",
    "even_process_data",
]
