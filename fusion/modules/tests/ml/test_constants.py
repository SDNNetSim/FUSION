"""Unit tests for fusion.modules.ml.constants module."""


from fusion.modules.ml.constants import BANDWIDTH_CATEGORIES, EXPECTED_ML_COLUMNS


class TestConstants:
    """Tests for ML module constants."""

    def test_expected_ml_columns_has_correct_length(self) -> None:
        """Test that EXPECTED_ML_COLUMNS contains expected number of columns."""
        # Arrange & Act
        column_count = len(EXPECTED_ML_COLUMNS)

        # Assert
        assert column_count == 7

    def test_expected_ml_columns_contains_path_length(self) -> None:
        """Test that path_length is in expected columns."""
        # Assert
        assert 'path_length' in EXPECTED_ML_COLUMNS

    def test_expected_ml_columns_contains_ave_cong(self) -> None:
        """Test that ave_cong is in expected columns."""
        # Assert
        assert 'ave_cong' in EXPECTED_ML_COLUMNS

    def test_expected_ml_columns_contains_bandwidth_features(self) -> None:
        """Test that all bandwidth one-hot encoded features are present."""
        # Arrange
        expected_bandwidth_cols = [
            'old_bandwidth_50',
            'old_bandwidth_100',
            'old_bandwidth_200',
            'old_bandwidth_400'
        ]

        # Act & Assert
        for col in expected_bandwidth_cols:
            assert col in EXPECTED_ML_COLUMNS

    def test_bandwidth_categories_has_correct_values(self) -> None:
        """Test that BANDWIDTH_CATEGORIES contains expected values."""
        # Assert
        assert BANDWIDTH_CATEGORIES == [50, 100, 200, 400]

    def test_bandwidth_categories_is_sorted(self) -> None:
        """Test that bandwidth categories are in ascending order."""
        # Assert
        assert BANDWIDTH_CATEGORIES == sorted(BANDWIDTH_CATEGORIES)

    def test_expected_ml_columns_is_list(self) -> None:
        """Test that EXPECTED_ML_COLUMNS is a list."""
        # Assert
        assert isinstance(EXPECTED_ML_COLUMNS, list)

    def test_bandwidth_categories_is_list(self) -> None:
        """Test that BANDWIDTH_CATEGORIES is a list."""
        # Assert
        assert isinstance(BANDWIDTH_CATEGORIES, list)

    def test_expected_ml_columns_has_no_duplicates(self) -> None:
        """Test that EXPECTED_ML_COLUMNS has no duplicate entries."""
        # Assert
        assert len(EXPECTED_ML_COLUMNS) == len(set(EXPECTED_ML_COLUMNS))

    def test_bandwidth_categories_has_no_duplicates(self) -> None:
        """Test that BANDWIDTH_CATEGORIES has no duplicate entries."""
        # Assert
        assert len(BANDWIDTH_CATEGORIES) == len(set(BANDWIDTH_CATEGORIES))
