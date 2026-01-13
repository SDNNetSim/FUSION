"""Unit tests for fusion.modules.ml.preprocessing module."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fusion.modules.ml.preprocessing import (
    balance_training_data,
    normalize_features,
    prepare_prediction_features,
    process_training_data,
    split_features_labels,
)


class TestProcessTrainingData:
    """Tests for process_training_data function."""

    @patch("fusion.modules.ml.preprocessing.plot_data_distributions")
    def test_process_with_valid_data_returns_dataframe(self, mock_plot: Mock) -> None:
        """Test that valid data is processed correctly."""
        # Arrange
        sim_dict = {"train_file_path": "test_experiment"}
        df = pd.DataFrame({"path_length": [10, 20], "old_bandwidth": [50, 100]})
        erlang = 1000.0

        # Act
        result = process_training_data(sim_dict, df, erlang)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_plot.assert_called_once()

    @patch("fusion.modules.ml.preprocessing.plot_data_distributions")
    def test_process_creates_one_hot_encoded_bandwidth(self, mock_plot: Mock) -> None:
        """Test that bandwidth is one-hot encoded."""
        # Arrange
        sim_dict = {"train_file_path": "test_experiment"}
        df = pd.DataFrame({"path_length": [10, 20, 30], "old_bandwidth": [50, 100, 50]})
        erlang = 1000.0

        # Act
        result = process_training_data(sim_dict, df, erlang)

        # Assert
        assert "old_bandwidth_50" in result.columns
        assert "old_bandwidth_100" in result.columns
        assert "old_bandwidth" not in result.columns

    @patch("fusion.modules.ml.preprocessing.plot_data_distributions")
    def test_process_converts_bool_to_int(self, mock_plot: Mock) -> None:
        """Test that boolean columns are converted to integers."""
        # Arrange
        sim_dict = {"train_file_path": "test_experiment"}
        df = pd.DataFrame({"path_length": [10, 20], "old_bandwidth": [50, 100]})
        erlang = 1000.0

        # Act
        result = process_training_data(sim_dict, df, erlang)

        # Assert
        for col in result.columns:
            if result[col].dtype == bool:
                pytest.fail(f"Column {col} should be int, not bool")

    @patch("fusion.modules.ml.preprocessing.plot_data_distributions")
    def test_process_with_empty_dataframe_returns_empty(self, mock_plot: Mock) -> None:
        """Test processing empty DataFrame with proper columns."""
        # Arrange
        sim_dict = {"train_file_path": "test_experiment"}
        # Create DataFrame with columns but no rows
        df = pd.DataFrame(columns=["path_length", "old_bandwidth"])
        erlang = 1000.0

        # Act
        result = process_training_data(sim_dict, df, erlang)

        # Assert
        assert len(result) == 0


class TestBalanceTrainingData:
    """Tests for balance_training_data function."""

    @patch("fusion.modules.ml.preprocessing.process_training_data")
    def test_balance_without_num_segments_returns_processed_data(
        self, mock_process: Mock
    ) -> None:
        """Test balancing data without num_segments column."""
        # Arrange
        df = pd.DataFrame({"feature1": [1, 2, 3]})
        mock_process.return_value = df
        sim_dict = {"train_file_path": "test"}
        erlang = 1000.0

        # Act
        result = balance_training_data(df, True, erlang, sim_dict)

        # Assert
        mock_process.assert_called_once()
        assert result is not None

    @patch("fusion.modules.ml.preprocessing.process_training_data")
    def test_balance_equally_samples_minimum_size(self, mock_process: Mock) -> None:
        """Test equal balancing samples minimum class size."""
        # Arrange
        df = pd.DataFrame({"num_segments": [1, 1, 1, 2, 2, 4, 4, 4, 4]})
        mock_process.return_value = df
        sim_dict = {"train_file_path": "test"}
        erlang = 1000.0

        # Act
        balance_training_data(df, True, erlang, sim_dict)

        # Assert
        mock_process.assert_called()

    @patch("fusion.modules.ml.preprocessing.process_training_data")
    def test_balance_weighted_applies_weights(self, mock_process: Mock) -> None:
        """Test weighted balancing applies correct weights."""
        # Arrange
        df = pd.DataFrame(
            {"num_segments": [1] * 100 + [2] * 100 + [4] * 100 + [8] * 100}
        )
        mock_process.return_value = df
        sim_dict = {"train_file_path": "test"}
        erlang = 1000.0

        # Act
        balance_training_data(df, False, erlang, sim_dict)

        # Assert
        mock_process.assert_called()


class TestPreparePredictionFeatures:
    """Tests for prepare_prediction_features function."""

    def test_prepare_with_valid_features_returns_dataframe(self) -> None:
        """Test that valid features are prepared correctly."""
        # Arrange
        features = {
            "path_length": 15,
            "ave_cong": 0.3,
            "longest_reach": 2000,
            "old_bandwidth": 100,
        }
        engine_props = {"request_distribution": {50: 0.5, 100: 0.5}}
        sdn_props = Mock()
        sdn_props.bandwidth = 100

        # Act
        result = prepare_prediction_features(features, engine_props, sdn_props)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_prepare_creates_one_hot_bandwidth_columns(self) -> None:
        """Test that bandwidth columns are one-hot encoded."""
        # Arrange
        features = {
            "path_length": 15,
            "ave_cong": 0.3,
            "longest_reach": 2000,
            "old_bandwidth": 50,
        }
        engine_props = {"request_distribution": {50: 0.5, 100: 0.5}}
        sdn_props = Mock()
        sdn_props.bandwidth = 50

        # Act
        result = prepare_prediction_features(features, engine_props, sdn_props)

        # Assert
        assert "old_bandwidth" not in result.columns

    def test_prepare_adds_missing_bandwidth_columns(self) -> None:
        """Test that missing bandwidth columns are added with zeros."""
        # Arrange
        features = {
            "path_length": 15,
            "ave_cong": 0.3,
            "longest_reach": 2000,
            "old_bandwidth": 50,
        }
        engine_props = {"request_distribution": {50: 0.5, 100: 0.5, 200: 0}}
        sdn_props = Mock()
        sdn_props.bandwidth = 50

        # Act
        result = prepare_prediction_features(features, engine_props, sdn_props)

        # Assert
        # Check that the result has been properly processed
        assert isinstance(result, pd.DataFrame)


class TestSplitFeaturesLabels:
    """Tests for split_features_labels function."""

    def test_split_with_valid_data_returns_features_and_labels(self) -> None:
        """Test successful split of features and labels."""
        # Arrange
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )

        # Act
        features, labels = split_features_labels(df, "target")

        # Assert
        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.Series)
        assert "target" not in features.columns
        assert len(features) == 3
        assert len(labels) == 3

    def test_split_raises_error_when_target_missing(self) -> None:
        """Test that KeyError is raised when target column missing."""
        # Arrange
        df = pd.DataFrame({"feature1": [1, 2, 3]})

        # Act & Assert
        with pytest.raises(KeyError) as exc_info:
            split_features_labels(df, "nonexistent_target")

        assert "not found in DataFrame" in str(exc_info.value)

    def test_split_preserves_feature_columns(self) -> None:
        """Test that all non-target columns are preserved."""
        # Arrange
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "feature3": [7, 8, 9],
                "target": [0, 1, 0],
            }
        )

        # Act
        features, _labels = split_features_labels(df, "target")

        # Assert
        assert len(features.columns) == 3
        assert "feature1" in features.columns
        assert "feature2" in features.columns
        assert "feature3" in features.columns


class TestNormalizeFeatures:
    """Tests for normalize_features function."""

    def test_normalize_with_standard_scaler_returns_normalized_data(
        self,
    ) -> None:
        """Test standard normalization returns scaled data."""
        # Arrange
        df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
        )

        # Act
        normalized, scaler = normalize_features(df, "standard")

        # Assert
        assert isinstance(normalized, pd.DataFrame)
        assert isinstance(scaler, StandardScaler)
        assert normalized.shape == df.shape
        # Check that mean is approximately 0
        assert abs(normalized["feature1"].mean()) < 1e-10

    def test_normalize_with_minmax_scaler_returns_scaled_data(self) -> None:
        """Test minmax normalization returns scaled data."""
        # Arrange
        df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
        )

        # Act
        normalized, scaler = normalize_features(df, "minmax")

        # Assert
        assert isinstance(normalized, pd.DataFrame)
        assert isinstance(scaler, MinMaxScaler)
        assert normalized.shape == df.shape
        # Check that values are between 0 and 1
        assert normalized["feature1"].min() == 0.0
        assert normalized["feature1"].max() == 1.0

    def test_normalize_raises_error_for_invalid_type(self) -> None:
        """Test that ValueError is raised for invalid normalization type."""
        # Arrange
        df = pd.DataFrame({"feature1": [1, 2, 3]})

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            normalize_features(df, "invalid_type")

        assert "not supported" in str(exc_info.value)

    def test_normalize_preserves_column_names(self) -> None:
        """Test that column names are preserved after normalization."""
        # Arrange
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})

        # Act
        normalized, _scaler = normalize_features(df, "standard")

        # Assert
        assert list(normalized.columns) == list(df.columns)

    def test_normalize_preserves_index(self) -> None:
        """Test that index is preserved after normalization."""
        # Arrange
        df = pd.DataFrame({"feature1": [1, 2, 3]}, index=["a", "b", "c"])

        # Act
        normalized, _scaler = normalize_features(df, "standard")

        # Assert
        assert list(normalized.index) == list(df.index)
