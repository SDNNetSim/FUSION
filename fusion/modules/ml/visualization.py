"""
Visualization utilities for machine learning module.

This module handles all plotting and visualization tasks including
feature importance, confusion matrices, and data distribution plots.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Optional imports
try:
    from sklearn.inspection import permutation_importance

    HAS_PERMUTATION_IMPORTANCE = True
except ImportError:
    HAS_PERMUTATION_IMPORTANCE = False

from fusion.utils.logging_config import get_logger
from fusion.utils.os import create_directory

logger = get_logger(__name__)


def plot_data_distributions(
    simulation_dict: dict[str, Any], input_dataframe: pd.DataFrame, erlang: float
) -> None:
    """
    Plot data distributions for machine learning simulation runs.

    Creates pie charts and histograms for key features in the dataset.

    :param simulation_dict: Dictionary containing simulation parameters
    :type simulation_dict: Dict[str, Any]
    :param input_dataframe: DataFrame containing simulation data
    :type input_dataframe: pd.DataFrame
    :param erlang: Traffic volume value
    :type erlang: float
    :return: None

    Example:
        >>> sim_dict = {'train_file_path': 'experiment_001'}
        >>> data = pd.DataFrame({'bandwidth': [50, 100, 200]})
        >>> plot_data_distributions(sim_dict, data, 1000.0)
    """
    save_filepath = os.path.join(
        "data", "plots", simulation_dict["train_file_path"], "input_analysis"
    )
    create_directory(directory_path=save_filepath)

    _plot_pie_charts(
        erlang=erlang, input_dataframe=input_dataframe, save_filepath=save_filepath
    )
    _plot_histograms(
        erlang=erlang, input_dataframe=input_dataframe, save_filepath=save_filepath
    )


def _plot_pie_charts(
    input_dataframe: pd.DataFrame, erlang: float, save_filepath: str
) -> None:
    """Plot pie charts for categorical features."""
    categorical_columns = ["old_bandwidth", "num_segments", "longest_reach"]

    for column in categorical_columns:
        if column not in input_dataframe.columns:
            logger.warning(
                "Column '%s' not found in dataframe, skipping pie chart", column
            )
            continue

        plt.figure(figsize=(6, 6), dpi=300)
        counts = input_dataframe[column].value_counts()

        # Create pie chart with percentages
        input_dataframe[column].value_counts().plot(
            kind="pie",
            autopct=lambda p: f"{p:.1f}%",
            textprops={"color": "white", "weight": "bold"},
        )

        plt.title(f"Distribution of {column} - {erlang} Erlang", weight="bold")

        # Create custom labels showing counts
        labels = [f"{label}: {count:,}" for label, count in counts.items()]
        plt.legend(labels, loc="best", bbox_to_anchor=(1.1, 1))

        output_path = os.path.join(save_filepath, f"pie_chart_{column}_{erlang}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


def _plot_histograms(
    erlang: float, save_filepath: str, input_dataframe: pd.DataFrame
) -> None:
    """Plot histograms and box plots for continuous features."""
    continuous_columns = ["path_length", "ave_cong"]

    for column in continuous_columns:
        if column not in input_dataframe.columns:
            logger.warning(
                "Column '%s' not found in dataframe, skipping histogram", column
            )
            continue

        plt.figure(figsize=(12, 6), dpi=300)

        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(input_dataframe[column], kde=True, bins="auto")
        plt.title(f"Distribution of {column} - {erlang} Erlang", weight="bold")
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)

        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=input_dataframe[column])
        plt.title(f"Box Plot of {column} - {erlang} Erlang", weight="bold")
        plt.xlabel(column.replace("_", " ").title())

        output_path = os.path.join(save_filepath, f"distribution_{column}_{erlang}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


def plot_feature_importance(
    simulation_dict: dict[str, Any],
    model: Any,
    feature_names: list[str],
    erlang: float,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    """
    Plot feature importance for a trained model.

    Supports tree-based models, linear models, and uses permutation
    importance for models without built-in importance scores.

    :param simulation_dict: Dictionary containing simulation parameters
    :type simulation_dict: Dict[str, Any]
    :param model: Trained machine learning model
    :type model: Any
    :param feature_names: List of feature names
    :type feature_names: List[str]
    :param erlang: Traffic volume value
    :type erlang: float
    :param test_features: Test feature matrix
    :type test_features: np.ndarray
    :param test_labels: Test labels
    :type test_labels: np.ndarray
    :return: None

    Example:
        >>> features = ['path_length', 'bandwidth', 'congestion']
        >>> plot_feature_importance(sim_dict, model, features, 1000.0, X_test, y_test)
    """
    try:
        # Try tree-based models first
        importances = model.feature_importances_
    except AttributeError:
        try:
            # Try linear models
            importances = np.abs(model.coef_[0])
        except AttributeError:
            # Fall back to permutation importance
            if not HAS_PERMUTATION_IMPORTANCE:
                logger.warning("permutation_importance not available, using zeros")
                importances = np.zeros(len(feature_names))
            else:
                permutation_result = permutation_importance(
                    model, test_features, test_labels, n_repeats=10, random_state=42
                )
                importances = permutation_result.importances_mean

    # Sort features by importance
    indices = np.argsort(importances)[::-1]

    # Create plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.title(f"Feature Importance Rankings - {erlang} Erlang", weight="bold")

    # Create bars
    bars = plt.bar(
        range(len(importances)),
        importances[indices],
        color=sns.color_palette("colorblind", len(importances)),
    )

    # Add value labels on bars
    for bar_plot, importance in zip(bars, importances[indices], strict=False):
        height = bar_plot.get_height()
        plt.text(
            bar_plot.get_x() + bar_plot.get_width() / 2.0,
            height,
            f"{importance:.3f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_filepath = os.path.join("data", "plots", simulation_dict["train_file_path"])
    create_directory(directory_path=save_filepath)
    output_path = os.path.join(save_filepath, f"feature_importance_{erlang}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    simulation_dict: dict[str, Any],
    test_labels: np.ndarray,
    predictions: np.ndarray,
    erlang: str,
    algorithm: str,
) -> dict[str, float]:
    """
    Plot confusion matrix and calculate classification metrics.

    Creates a heatmap visualization of the confusion matrix and
    returns accuracy, precision, recall, and F1 scores.

    :param simulation_dict: Dictionary containing simulation parameters
    :type simulation_dict: Dict[str, Any]
    :param test_labels: True labels
    :type test_labels: np.ndarray
    :param predictions: Model predictions
    :type predictions: np.ndarray
    :param erlang: Traffic volume value as string
    :type erlang: str
    :param algorithm: Name of the algorithm used
    :type algorithm: str
    :return: Dictionary of classification metrics
    :rtype: Dict[str, float]

    Example:
        >>> metrics = plot_confusion_matrix(sim_dict, y_test, y_pred, "1000", "RandomForest")
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(test_labels, predictions, average="weighted"),
        "recall": recall_score(test_labels, predictions, average="weighted"),
        "f1_score": f1_score(test_labels, predictions, average="weighted"),
    }

    # Get unique labels
    labels = np.unique(np.concatenate((test_labels, predictions)))

    # Create confusion matrix
    confusion_matrix_data = confusion_matrix(test_labels, predictions, labels=labels)

    # Create figure
    plt.figure(figsize=(10, 8), dpi=300)

    # Plot heatmap
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )

    plt.title(
        f"Confusion Matrix - {algorithm} ({erlang} Erlang)", weight="bold", fontsize=14
    )
    plt.xlabel("Predicted Class", weight="bold")
    plt.ylabel("Actual Class", weight="bold")

    # Add metrics to plot
    _add_metrics_to_plot(test_labels, predictions, metrics)

    # Save plot
    save_filepath = os.path.join("data", "plots", simulation_dict["train_file_path"])
    create_directory(directory_path=save_filepath)
    output_path = os.path.join(save_filepath, f"confusion_matrix_{erlang}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return metrics


def _add_metrics_to_plot(
    test_labels: np.ndarray, predictions: np.ndarray, metrics: dict[str, float]
) -> None:
    """Add classification metrics as text to the current plot."""
    # Calculate per-class accuracy
    unique_labels = np.unique(test_labels)
    per_class_accuracy = []

    for label in unique_labels:
        mask = test_labels == label
        class_accuracy = accuracy_score(test_labels[mask], predictions[mask])
        per_class_accuracy.append(f"Class {label}: {class_accuracy:.3f}")

    # Format text
    accuracy_text = ", ".join(per_class_accuracy)

    # Add text to plot
    text_y_position = 1.02
    plt.text(
        0.02,
        text_y_position,
        f"Overall Accuracy: {metrics['accuracy']:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
    )
    plt.text(
        0.02,
        text_y_position + 0.04,
        f"Per-class: {accuracy_text}",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
    )
    plt.text(
        0.02,
        text_y_position + 0.08,
        f"Precision: {metrics['precision']:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
    )
    plt.text(
        0.02,
        text_y_position + 0.12,
        f"Recall: {metrics['recall']:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
    )
    plt.text(
        0.02,
        text_y_position + 0.16,
        f"F1 Score: {metrics['f1_score']:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
    )


def plot_2d_clusters(pca_dataframe: pd.DataFrame, output_path: str = None) -> None:
    """
    Plot 2D visualization of clusters using PCA-reduced data.

    :param pca_dataframe: DataFrame with PCA components and predicted labels
    :type pca_dataframe: pd.DataFrame
    :param output_path: Optional path to save the plot
    :type output_path: str
    :return: None

    Example:
        >>> df_pca = pd.DataFrame({'PC1': [...], 'PC2': [...], 'predicted_label': [...]})
        >>> plot_2d_clusters(df_pca, 'output/clusters_2d.png')
    """
    plt.figure(figsize=(10, 8), dpi=300)

    # Create scatter plot
    scatter = plt.scatter(
        pca_dataframe["PC1"],
        pca_dataframe["PC2"],
        c=pca_dataframe["predicted_label"],
        cmap="Set1",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    plt.title("Predicted Clusters (2D PCA Projection)", weight="bold", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.colorbar(scatter, label="Predicted Class")
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_3d_clusters(pca_dataframe: pd.DataFrame, output_path: str = None) -> None:
    """
    Plot 3D visualization of clusters using PCA-reduced data.

    :param pca_dataframe: DataFrame with PCA components and predicted labels
    :type pca_dataframe: pd.DataFrame
    :param output_path: Optional path to save the plot
    :type output_path: str
    :return: None

    Example:
        >>> df_pca = pd.DataFrame({'PC1': [...], 'PC2': [...], 'PC3': [...], 'predicted_label': [...]})
        >>> plot_3d_clusters(df_pca, 'output/clusters_3d.png')
    """
    fig = plt.figure(figsize=(12, 9), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Create 3D scatter plot
    scatter = ax.scatter(
        pca_dataframe["PC1"],
        pca_dataframe["PC2"],
        pca_dataframe["PC3"],
        c=pca_dataframe["predicted_label"],
        cmap="Set1",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_title("Predicted Clusters (3D PCA Projection)", weight="bold", fontsize=14)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_zlabel("Principal Component 3", fontsize=12)

    # Add colorbar
    fig.colorbar(scatter, label="Predicted Class", pad=0.1)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.close()
