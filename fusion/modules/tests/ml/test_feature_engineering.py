"""Unit tests for fusion.modules.ml.feature_engineering module."""

from typing import Any
from unittest.mock import Mock, patch

import pandas as pd

from fusion.modules.ml.feature_engineering import (
    create_interaction_features,
    create_polynomial_features,
    engineer_network_features,
    extract_ml_features,
)


class TestExtractMLFeatures:
    """Tests for extract_ml_features function."""

    @patch("fusion.modules.ml.feature_engineering.find_path_len")
    @patch("fusion.modules.ml.feature_engineering._calculate_congestion_metrics")
    @patch("fusion.modules.ml.feature_engineering._format_features_for_prediction")
    def test_extract_calls_path_length_calculation(self, mock_format: Mock, mock_cong: Mock, mock_path_len: Mock) -> None:
        """Test that path length is calculated."""
        # Arrange
        mock_path_len.return_value = 150.0
        mock_cong.return_value = {"average": 0.5, "max": 0.8, "min": 0.2, "std": 0.1}
        mock_format.return_value = pd.DataFrame()

        request = {"bandwidth": 100, "mod_formats": {"QPSK": {"max_length": 2000}}}
        engine_props = {"topology": Mock(), "request_distribution": {}}
        sdn_props = Mock()
        sdn_props.path_list = []
        sdn_props.network_spectrum = {}

        # Act
        extract_ml_features(request, engine_props, sdn_props)

        # Assert
        mock_path_len.assert_called_once()

    @patch("fusion.modules.ml.feature_engineering.find_path_len")
    @patch("fusion.modules.ml.feature_engineering._calculate_congestion_metrics")
    @patch("fusion.modules.ml.feature_engineering._format_features_for_prediction")
    def test_extract_calls_congestion_calculation(self, mock_format: Mock, mock_cong: Mock, mock_path_len: Mock) -> None:
        """Test that congestion metrics are calculated."""
        # Arrange
        mock_path_len.return_value = 150.0
        mock_cong.return_value = {"average": 0.5, "max": 0.8, "min": 0.2, "std": 0.1}
        mock_format.return_value = pd.DataFrame()

        request = {"bandwidth": 100, "mod_formats": {"QPSK": {"max_length": 2000}}}
        engine_props = {"topology": Mock(), "request_distribution": {}}
        sdn_props = Mock()

        # Act
        extract_ml_features(request, engine_props, sdn_props)

        # Assert
        mock_cong.assert_called_once()

    @patch("fusion.modules.ml.feature_engineering.find_path_len")
    @patch("fusion.modules.ml.feature_engineering._calculate_congestion_metrics")
    @patch("fusion.modules.ml.feature_engineering._format_features_for_prediction")
    def test_extract_returns_dataframe(self, mock_format: Mock, mock_cong: Mock, mock_path_len: Mock) -> None:
        """Test that extract_ml_features returns a DataFrame."""
        # Arrange
        mock_path_len.return_value = 150.0
        mock_cong.return_value = {"average": 0.5, "max": 0.8, "min": 0.2, "std": 0.1}
        mock_df = pd.DataFrame({"feature": [1]})
        mock_format.return_value = mock_df

        request = {"bandwidth": 100, "mod_formats": {"QPSK": {"max_length": 2000}}}
        engine_props = {"topology": Mock(), "request_distribution": {}}
        sdn_props = Mock()

        # Act
        result = extract_ml_features(request, engine_props, sdn_props)

        # Assert
        assert isinstance(result, pd.DataFrame)


class TestCreateInteractionFeatures:
    """Tests for create_interaction_features function."""

    def test_create_with_specified_interactions_adds_columns(self) -> None:
        """Test that interaction features are created."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        interactions = [("A", "B")]

        # Act
        result = create_interaction_features(df, interactions)

        # Assert
        assert "A_x_B" in result.columns
        assert result.shape[1] == 3  # Original 2 + 1 interaction

    def test_create_calculates_correct_interaction_values(self) -> None:
        """Test that interaction values are calculated correctly."""
        # Arrange
        df = pd.DataFrame({"A": [2, 3], "B": [4, 5]})
        interactions = [("A", "B")]

        # Act
        result = create_interaction_features(df, interactions)

        # Assert
        assert result["A_x_B"].iloc[0] == 8  # 2 * 4
        assert result["A_x_B"].iloc[1] == 15  # 3 * 5

    def test_create_with_default_interactions_uses_defaults(self) -> None:
        """Test that default interactions are used when none specified."""
        # Arrange
        df = pd.DataFrame(
            {
                "path_length": [10, 20],
                "ave_cong": [0.5, 0.7],
                "longest_reach": [1000, 2000],
            }
        )

        # Act
        result = create_interaction_features(df, None)

        # Assert
        assert "path_length_x_ave_cong" in result.columns
        assert "path_length_x_longest_reach" in result.columns

    def test_create_with_missing_column_skips_interaction(self) -> None:
        """Test that interaction is skipped if column missing."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2]})
        interactions = [("A", "B")]  # B doesn't exist

        # Act
        result = create_interaction_features(df, interactions)

        # Assert
        assert "A_x_B" not in result.columns

    def test_create_preserves_original_columns(self) -> None:
        """Test that original columns are preserved."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        interactions = [("A", "B")]

        # Act
        result = create_interaction_features(df, interactions)

        # Assert
        assert "A" in result.columns
        assert "B" in result.columns


class TestCreatePolynomialFeatures:
    """Tests for create_polynomial_features function."""

    def test_create_polynomial_with_degree_2_creates_squared_terms(
        self,
    ) -> None:
        """Test that polynomial features include squared terms."""
        # Arrange
        df = pd.DataFrame({"x": [1, 2, 3]})

        # Act
        result = create_polynomial_features(df, degree=2, include_bias=False)

        # Assert
        # Should have x and x^2
        assert result.shape[1] >= 2

    def test_create_polynomial_without_bias_excludes_bias(self) -> None:
        """Test that bias term is excluded when specified."""
        # Arrange
        df = pd.DataFrame({"x": [1, 2, 3]})

        # Act
        result = create_polynomial_features(df, degree=2, include_bias=False)

        # Assert
        # First column should not be all 1s (bias)
        assert not (result.iloc[:, 0] == 1).all()

    def test_create_polynomial_with_bias_includes_bias(self) -> None:
        """Test that bias term is included when specified."""
        # Arrange
        df = pd.DataFrame({"x": [1, 2, 3]})

        # Act
        result = create_polynomial_features(df, degree=2, include_bias=True)

        # Assert
        # First column should be all 1s (bias)
        assert (result.iloc[:, 0] == 1).all()

    def test_create_polynomial_preserves_row_count(self) -> None:
        """Test that row count is preserved."""
        # Arrange
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

        # Act
        result = create_polynomial_features(df, degree=2)

        # Assert
        assert len(result) == len(df)

    def test_create_polynomial_with_multiple_features(self) -> None:
        """Test polynomial features with multiple input features."""
        # Arrange
        df = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})

        # Act
        result = create_polynomial_features(df, degree=2, include_bias=False)

        # Assert
        # Should have x1, x2, x1^2, x1*x2, x2^2
        assert result.shape[1] >= 4


class TestEngineerNetworkFeatures:
    """Tests for engineer_network_features function."""

    def test_engineer_with_alternative_paths_creates_diversity_features(
        self,
    ) -> None:
        """Test that path diversity features are created."""
        # Arrange
        request = {"bandwidth": 100}
        network_state = {"alternative_paths": [[1, 2, 3], [1, 4, 3], [1, 5, 6, 3]]}

        # Act
        features = engineer_network_features(request, network_state)

        # Assert
        assert "path_diversity" in features
        assert features["path_diversity"] == 3.0
        assert "avg_alternative_length" in features

    def test_engineer_with_link_utilization_creates_bottleneck_features(
        self,
    ) -> None:
        """Test that link utilization features are created."""
        # Arrange
        request = {"bandwidth": 100}
        network_state = {"link_utilization": {"link1": 0.5, "link2": 0.8, "link3": 0.3}}

        # Act
        features = engineer_network_features(request, network_state)

        # Assert
        assert "max_link_utilization" in features
        assert features["max_link_utilization"] == 0.8
        assert "utilization_variance" in features

    def test_engineer_with_time_of_day_creates_temporal_features(self) -> None:
        """Test that temporal features are created."""
        # Arrange
        request = {"bandwidth": 100}
        network_state = {"time_of_day": 10}

        # Act
        features = engineer_network_features(request, network_state)

        # Assert
        assert "is_peak_hour" in features
        assert features["is_peak_hour"] == 1.0  # 10 is during peak hours
        assert "hour_sin" in features
        assert "hour_cos" in features

    def test_engineer_with_bandwidth_creates_category_feature(self) -> None:
        """Test that bandwidth category is created."""
        # Arrange
        request = {"bandwidth": 150}
        network_state: dict[str, Any] = {}

        # Act
        features = engineer_network_features(request, network_state)

        # Assert
        assert "bandwidth_category" in features

    def test_engineer_categorizes_bandwidth_correctly(self) -> None:
        """Test that bandwidth is categorized into correct levels."""
        # Arrange & Act
        features_low = engineer_network_features({"bandwidth": 50}, {})
        features_medium = engineer_network_features({"bandwidth": 100}, {})
        features_high = engineer_network_features({"bandwidth": 200}, {})
        features_very_high = engineer_network_features({"bandwidth": 500}, {})

        # Assert
        assert features_low["bandwidth_category"] == 0.0  # Low
        assert features_medium["bandwidth_category"] == 1.0  # Medium
        assert features_high["bandwidth_category"] == 2.0  # High
        assert features_very_high["bandwidth_category"] == 3.0  # Very High

    def test_engineer_handles_empty_network_state(self) -> None:
        """Test that empty network state is handled gracefully."""
        # Arrange
        request = {"source": "A", "destination": "B"}
        network_state: dict[str, Any] = {}

        # Act
        features = engineer_network_features(request, network_state)

        # Assert
        assert isinstance(features, dict)

    def test_engineer_handles_empty_alternative_paths(self) -> None:
        """Test handling of empty alternative paths list."""
        # Arrange
        request = {"bandwidth": 100}
        network_state: dict[str, Any] = {"alternative_paths": []}

        # Act
        features = engineer_network_features(request, network_state)

        # Assert
        assert features["path_diversity"] == 0.0
        assert features["avg_alternative_length"] == 0.0
