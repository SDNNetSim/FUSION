"""
Machine Learning module for FUSION.

This module provides machine learning capabilities for network optimization,
organized by functional areas according to ML_STANDARDS.md.
"""

# Evaluation exports
from fusion.modules.ml.evaluation import (
    analyze_prediction_errors,
    compare_models,
    cross_validate_model,
    evaluate_classifier,
    evaluate_model_stability,
    evaluate_regressor,
)

# Feature engineering exports
from fusion.modules.ml.feature_engineering import (  # Previously get_ml_obs
    create_interaction_features,
    create_polynomial_features,
    engineer_network_features,
    extract_ml_features,
)

# Model I/O exports
from fusion.modules.ml.model_io import (
    check_model_compatibility,
    export_model_for_deployment,
    load_model,
    load_model_with_metadata,
    save_model,
    save_model_ensemble,
)

# Preprocessing exports
from fusion.modules.ml.preprocessing import (  # Previously even_process_data; Previously process_data
    balance_training_data,
    normalize_features,
    prepare_prediction_features,
    process_training_data,
    split_features_labels,
)

# Registry (unchanged)
from fusion.modules.ml.registry import *

# Visualization exports
from fusion.modules.ml.visualization import (  # Previously plot_confusion; Previously plot_data
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
