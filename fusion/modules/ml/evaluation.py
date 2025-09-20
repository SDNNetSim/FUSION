"""
Model evaluation utilities for machine learning module.

This module provides comprehensive evaluation metrics and analysis tools
for assessing model performance.
"""

from typing import Dict, Any, List
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.base import clone

from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_classifier(
        true_labels: np.ndarray,
        predictions: np.ndarray,
        class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of classification model performance.
    
    :param true_labels: Ground truth labels
    :type true_labels: np.ndarray
    :param predictions: Model predictions
    :type predictions: np.ndarray
    :param class_names: Optional names for classes
    :type class_names: List[str]
    :return: Dictionary containing various metrics
    :rtype: Dict[str, Any]
    
    Example:
        >>> y_true = np.array([0, 1, 0, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> metrics = evaluate_classifier(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted'),
        'f1_score': f1_score(true_labels, predictions, average='weighted')
    }

    # Per-class metrics
    metrics['precision_per_class'] = precision_score(
        true_labels, predictions, average=None
    ).tolist()
    metrics['recall_per_class'] = recall_score(
        true_labels, predictions, average=None
    ).tolist()
    metrics['f1_score_per_class'] = f1_score(
        true_labels, predictions, average=None
    ).tolist()

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(true_labels, predictions).tolist()

    # Classification report
    if class_names:
        metrics['classification_report'] = classification_report(
            true_labels, predictions, target_names=class_names, output_dict=True
        )
    else:
        metrics['classification_report'] = classification_report(
            true_labels, predictions, output_dict=True
        )

    # Per-class accuracy
    metrics['accuracy_per_class'] = _calculate_per_class_accuracy(
        true_labels, predictions
    )

    # Try to calculate AUC if applicable
    try:
        unique_classes = np.unique(true_labels)
        if len(unique_classes) == 2:
            metrics['roc_auc'] = roc_auc_score(true_labels, predictions)
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # AUC not applicable for this case

    logger.info("Model evaluation - Accuracy: %.4f, F1-Score: %.4f",
                metrics['accuracy'], metrics['f1_score'])

    return metrics


def evaluate_regressor(
        true_values: np.ndarray,
        predictions: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive evaluation of regression model performance.
    
    :param true_values: Ground truth values
    :type true_values: np.ndarray
    :param predictions: Model predictions
    :type predictions: np.ndarray
    :return: Dictionary containing regression metrics
    :rtype: Dict[str, float]
    
    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 2.2, 2.9, 3.8])
        >>> metrics = evaluate_regressor(y_true, y_pred)
        >>> print(f"MSE: {metrics['mse']:.4f}")
    """
    metrics = {
        'mse': mean_squared_error(true_values, predictions),
        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
        'mae': mean_absolute_error(true_values, predictions),
        'r2': r2_score(true_values, predictions),
        'mape': _calculate_mape(true_values, predictions),
        'max_error': np.max(np.abs(true_values - predictions)),
        'explained_variance': 1 - np.var(true_values - predictions) / np.var(true_values)
    }

    logger.info("Model evaluation - RMSE: %.4f, R²: %.4f",
                metrics['rmse'], metrics['r2'])

    return metrics


def _calculate_per_class_accuracy(
        true_labels: np.ndarray,
        predictions: np.ndarray
) -> Dict[str, float]:
    """Calculate accuracy for each class separately."""
    per_class_accuracy = {}

    for class_label in np.unique(true_labels):
        mask = true_labels == class_label
        class_accuracy = accuracy_score(true_labels[mask], predictions[mask])
        per_class_accuracy[str(class_label)] = class_accuracy

    return per_class_accuracy


def _calculate_mape(true_values: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = true_values != 0
    if not any(mask):
        return float('inf')

    mape = np.mean(np.abs((true_values[mask] - predictions[mask]) / true_values[mask])) * 100
    return float(mape)


def cross_validate_model(
        model: Any,
        features: pd.DataFrame,
        labels: pd.Series,
        cv_folds: int = 5,
        scoring_metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Perform cross-validation and return detailed results.
    
    :param model: Model to evaluate
    :type model: Any
    :param features: Feature matrix
    :type features: pd.DataFrame
    :param labels: Target labels
    :type labels: pd.Series
    :param cv_folds: Number of cross-validation folds
    :type cv_folds: int
    :param scoring_metrics: List of scoring metrics to use
    :type scoring_metrics: List[str]
    :return: Cross-validation results
    :rtype: Dict[str, Any]
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> results = cross_validate_model(model, X, y, cv_folds=5)
        >>> print(f"Mean accuracy: {results['accuracy']['mean']:.4f}")
    """
    if scoring_metrics is None:
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    cv_results = cross_validate(
        model,
        features,
        labels,
        cv=cv_folds,
        scoring=scoring_metrics,
        return_train_score=True,
        n_jobs=-1
    )

    # Process results
    results = {}
    for metric in scoring_metrics:
        test_key = f'test_{metric}'
        train_key = f'train_{metric}'

        if test_key in cv_results:
            results[metric] = {
                'test_mean': np.mean(cv_results[test_key]),
                'test_std': np.std(cv_results[test_key]),
                'test_scores': cv_results[test_key].tolist(),
                'train_mean': np.mean(cv_results[train_key]),
                'train_std': np.std(cv_results[train_key]),
                'train_scores': cv_results[train_key].tolist(),
            }

    # Add timing information
    results['fit_time'] = {
        'mean': np.mean(cv_results['fit_time']),
        'std': np.std(cv_results['fit_time']),
        'total': np.sum(cv_results['fit_time'])
    }

    results['score_time'] = {
        'mean': np.mean(cv_results['score_time']),
        'std': np.std(cv_results['score_time'])
    }

    return results


def evaluate_model_stability(
        model: Any,
        features: pd.DataFrame,
        labels: pd.Series,
        n_iterations: int = 10,
        test_size: float = 0.3
) -> Dict[str, Any]:
    """
    Evaluate model stability across multiple train/test splits.
    
    :param model: Model to evaluate
    :type model: Any
    :param features: Feature matrix
    :type features: pd.DataFrame
    :param labels: Target labels
    :type labels: pd.Series
    :param n_iterations: Number of random splits to test
    :type n_iterations: int
    :param test_size: Proportion of data for testing
    :type test_size: float
    :return: Stability analysis results
    :rtype: Dict[str, Any]
    
    Example:
        >>> stability = evaluate_model_stability(model, X, y, n_iterations=20)
        >>> print(f"Accuracy variance: {stability['accuracy']['variance']:.6f}")
    """
    metrics_over_iterations = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    for i in range(n_iterations):
        # Random split
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=i
        )

        # Clone model to avoid contamination
        model_clone = clone(model)

        # Train and predict
        model_clone.fit(x_train, y_train)
        predictions = model_clone.predict(x_test)

        # Calculate metrics
        metrics_over_iterations['accuracy'].append(
            accuracy_score(y_test, predictions)
        )
        metrics_over_iterations['precision'].append(
            precision_score(y_test, predictions, average='weighted')
        )
        metrics_over_iterations['recall'].append(
            recall_score(y_test, predictions, average='weighted')
        )
        metrics_over_iterations['f1_score'].append(
            f1_score(y_test, predictions, average='weighted')
        )

    # Calculate statistics
    stability_results = {}
    for metric_name, values in metrics_over_iterations.items():
        stability_results[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'variance': np.var(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else float('inf')
        }

    return stability_results


def compare_models(
        models: Dict[str, Any],
        features: pd.DataFrame,
        labels: pd.Series,
        test_size: float = 0.3,
        random_state: int = 42
) -> pd.DataFrame:
    """
    Compare multiple models on the same dataset.
    
    :param models: Dictionary of model_name: model pairs
    :type models: Dict[str, Any]
    :param features: Feature matrix
    :type features: pd.DataFrame
    :param labels: Target labels
    :type labels: pd.Series
    :param test_size: Test set proportion
    :type test_size: float
    :param random_state: Random seed
    :type random_state: int
    :return: DataFrame with comparison results
    :rtype: pd.DataFrame
    
    Example:
        >>> models = {
        ...     'RF': RandomForestClassifier(),
        ...     'SVM': SVC(),
        ...     'LR': LogisticRegression()
        ... }
        >>> comparison = compare_models(models, X, y)
        >>> print(comparison.sort_values('f1_score', ascending=False))
    """
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    results = []

    for model_name, model in models.items():
        logger.info("Evaluating %s...", model_name)

        # Time training
        start_time = time.time()
        model.fit(x_train, y_train)
        train_time = time.time() - start_time

        # Time prediction
        start_time = time.time()
        predictions = model.predict(x_test)
        predict_time = time.time() - start_time

        # Calculate metrics
        result = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'train_time': train_time,
            'predict_time': predict_time,
            'total_time': train_time + predict_time
        }

        results.append(result)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.round(4)

    return comparison_df


def analyze_prediction_errors(
        true_labels: np.ndarray,
        predictions: np.ndarray,
        features: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Analyze prediction errors to identify patterns.
    
    :param true_labels: Ground truth labels
    :type true_labels: np.ndarray
    :param predictions: Model predictions
    :type predictions: np.ndarray
    :param features: Optional features for error analysis
    :type features: pd.DataFrame
    :return: Error analysis results
    :rtype: Dict[str, Any]
    
    Example:
        >>> analysis = analyze_prediction_errors(y_true, y_pred, X_test)
        >>> print(f"Most confused pair: {analysis['most_confused_pair']}")
    """
    error_mask = true_labels != predictions
    error_indices = np.where(error_mask)[0]

    analysis = {
        'total_errors': len(error_indices),
        'error_rate': np.mean(error_mask),
        'error_indices': error_indices.tolist()
    }

    # Analyze confusion patterns
    if len(error_indices) > 0:
        error_true = true_labels[error_mask]
        error_pred = predictions[error_mask]

        # Find most common misclassification
        confusion_pairs = list(zip(error_true, error_pred))
        pair_counts = Counter(confusion_pairs)

        if pair_counts:
            most_common = pair_counts.most_common(1)[0]
            analysis['most_confused_pair'] = {
                'true_class': most_common[0][0],
                'predicted_class': most_common[0][1],
                'count': most_common[1]
            }

        # Analyze by class
        analysis['errors_by_true_class'] = {}
        for class_label in np.unique(true_labels):
            class_mask = true_labels == class_label
            class_errors = np.sum((true_labels == class_label) & error_mask)
            class_total = np.sum(class_mask)
            analysis['errors_by_true_class'][str(class_label)] = {
                'count': int(class_errors),
                'total': int(class_total),
                'rate': float(class_errors / class_total) if class_total > 0 else 0.0
            }

    # Feature analysis if provided
    if features is not None and len(error_indices) > 0:
        error_features = features.iloc[error_indices]
        correct_features = features.iloc[~error_mask]

        analysis['feature_stats'] = {}
        for column in features.select_dtypes(include=[np.number]).columns:
            analysis['feature_stats'][column] = {
                'error_mean': float(error_features[column].mean()),
                'correct_mean': float(correct_features[column].mean()),
                'error_std': float(error_features[column].std()),
                'correct_std': float(correct_features[column].std())
            }

    return analysis
