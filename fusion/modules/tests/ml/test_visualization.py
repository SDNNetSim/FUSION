"""Unit tests for fusion.modules.ml.visualization module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from fusion.modules.ml.visualization import (
    plot_2d_clusters,
    plot_3d_clusters,
    plot_confusion_matrix,
    plot_data_distributions,
    plot_feature_importance,
)


class TestPlotDataDistributions:
    """Tests for plot_data_distributions function."""

    @patch("fusion.modules.ml.visualization.create_directory")
    @patch("fusion.modules.ml.visualization._plot_pie_charts")
    @patch("fusion.modules.ml.visualization._plot_histograms")
    def test_plot_creates_directory(
        self, mock_hist: Mock, mock_pie: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that plot_data_distributions creates output directory."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        df = pd.DataFrame({"bandwidth": [50, 100]})
        erlang = 1000.0

        # Act
        plot_data_distributions(sim_dict, df, erlang)

        # Assert
        mock_create_dir.assert_called_once()

    @patch("fusion.modules.ml.visualization.create_directory")
    @patch("fusion.modules.ml.visualization._plot_pie_charts")
    @patch("fusion.modules.ml.visualization._plot_histograms")
    def test_plot_calls_pie_and_histogram_functions(
        self, mock_hist: Mock, mock_pie: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that both pie charts and histograms are generated."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        df = pd.DataFrame({"bandwidth": [50, 100]})
        erlang = 1000.0

        # Act
        plot_data_distributions(sim_dict, df, erlang)

        # Assert
        mock_pie.assert_called_once()
        mock_hist.assert_called_once()


class TestPlotFeatureImportance:
    """Tests for plot_feature_importance function."""

    @patch("fusion.modules.ml.visualization.create_directory")
    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_with_tree_model_uses_feature_importances(
        self, mock_close: Mock, mock_savefig: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that tree-based model importances are used."""
        # Arrange
        model = Mock()
        model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        sim_dict = {"train_file_path": "experiment_001"}
        features = ["f1", "f2", "f3"]
        erlang = 1000.0
        X_test = np.array([[1, 2, 3]])
        y_test = np.array([0])

        # Act
        plot_feature_importance(sim_dict, model, features, erlang, X_test, y_test)

        # Assert
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("fusion.modules.ml.visualization.create_directory")
    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_with_linear_model_uses_coefficients(
        self, mock_close: Mock, mock_savefig: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that linear model coefficients are used."""
        # Arrange
        model = Mock()
        del model.feature_importances_  # No feature_importances_
        model.coef_ = np.array([[0.5, -0.3, 0.2]])
        sim_dict = {"train_file_path": "experiment_001"}
        features = ["f1", "f2", "f3"]
        erlang = 1000.0
        X_test = np.array([[1, 2, 3]])
        y_test = np.array([0])

        # Act
        plot_feature_importance(sim_dict, model, features, erlang, X_test, y_test)

        # Assert
        mock_savefig.assert_called_once()

    @patch("fusion.modules.ml.visualization.create_directory")
    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    @patch("fusion.modules.ml.visualization.HAS_PERMUTATION_IMPORTANCE", False)
    def test_plot_without_importances_uses_zeros(
        self, mock_close: Mock, mock_savefig: Mock, mock_create_dir: Mock
    ) -> None:
        """Test fallback to zeros when no importance available."""
        # Arrange
        model = Mock()
        del model.feature_importances_
        del model.coef_
        sim_dict = {"train_file_path": "experiment_001"}
        features = ["f1", "f2", "f3"]
        erlang = 1000.0
        X_test = np.array([[1, 2, 3]])
        y_test = np.array([0])

        # Act
        plot_feature_importance(sim_dict, model, features, erlang, X_test, y_test)

        # Assert
        mock_savefig.assert_called_once()


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    @patch("fusion.modules.ml.visualization.create_directory")
    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_returns_metrics_dict(
        self, mock_close: Mock, mock_savefig: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that confusion matrix plot returns metrics."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        y_test = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        erlang = "1000"
        algorithm = "RandomForest"

        # Act
        metrics = plot_confusion_matrix(sim_dict, y_test, y_pred, erlang, algorithm)

        # Assert
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    @patch("fusion.modules.ml.visualization.create_directory")
    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_creates_plot_file(
        self, mock_close: Mock, mock_savefig: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that plot file is saved."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        y_test = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        erlang = "1000"
        algorithm = "RandomForest"

        # Act
        plot_confusion_matrix(sim_dict, y_test, y_pred, erlang, algorithm)

        # Assert
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("fusion.modules.ml.visualization.create_directory")
    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_with_perfect_predictions_returns_one(
        self, mock_close: Mock, mock_savefig: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that perfect predictions yield 1.0 metrics."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        y_test = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        erlang = "1000"
        algorithm = "RandomForest"

        # Act
        metrics = plot_confusion_matrix(sim_dict, y_test, y_pred, erlang, algorithm)

        # Assert
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0


class TestPlot2DClusters:
    """Tests for plot_2d_clusters function."""

    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_2d_with_valid_data_creates_plot(
        self, mock_close: Mock, mock_savefig: Mock
    ) -> None:
        """Test that 2D cluster plot is created with valid data."""
        # Arrange
        df = pd.DataFrame(
            {
                "PC1": [1.0, 2.0, 3.0],
                "PC2": [4.0, 5.0, 6.0],
                "predicted_label": [0, 1, 0],
            }
        )

        # Act
        plot_2d_clusters(df, None)

        # Assert
        mock_close.assert_called_once()

    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_2d_with_output_path_saves_file(
        self, mock_close: Mock, mock_savefig: Mock
    ) -> None:
        """Test that plot is saved when output path provided."""
        # Arrange
        df = pd.DataFrame(
            {"PC1": [1.0, 2.0], "PC2": [3.0, 4.0], "predicted_label": [0, 1]}
        )
        output_path = "output/clusters_2d.png"

        # Act
        plot_2d_clusters(df, output_path)

        # Assert
        mock_savefig.assert_called_once_with(output_path, bbox_inches="tight")


class TestPlot3DClusters:
    """Tests for plot_3d_clusters function."""

    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_3d_with_valid_data_creates_plot(
        self, mock_close: Mock, mock_savefig: Mock
    ) -> None:
        """Test that 3D cluster plot is created with valid data."""
        # Arrange
        df = pd.DataFrame(
            {
                "PC1": [1.0, 2.0, 3.0],
                "PC2": [4.0, 5.0, 6.0],
                "PC3": [7.0, 8.0, 9.0],
                "predicted_label": [0, 1, 0],
            }
        )

        # Act
        plot_3d_clusters(df, None)

        # Assert
        mock_close.assert_called_once()

    @patch("fusion.modules.ml.visualization.plt.savefig")
    @patch("fusion.modules.ml.visualization.plt.close")
    def test_plot_3d_with_output_path_saves_file(
        self, mock_close: Mock, mock_savefig: Mock
    ) -> None:
        """Test that 3D plot is saved when output path provided."""
        # Arrange
        df = pd.DataFrame(
            {
                "PC1": [1.0, 2.0],
                "PC2": [3.0, 4.0],
                "PC3": [5.0, 6.0],
                "predicted_label": [0, 1],
            }
        )
        output_path = "output/clusters_3d.png"

        # Act
        plot_3d_clusters(df, output_path)

        # Assert
        mock_savefig.assert_called_once_with(output_path, bbox_inches="tight")
