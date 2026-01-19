"""Unit tests for fusion.sim.utils.data module."""

from ..data import (
    calculate_matrix_statistics,
    dict_to_list,
    min_max_scale,
    sort_dict_keys,
    sort_nested_dict_values,
    update_dict_from_list,
    update_matrices,
)


class TestDictionarySorting:
    """Tests for dictionary sorting functions."""

    def test_sort_nested_dict_values_with_numeric_key_sorts_ascending(self) -> None:
        """Test sorting dictionary by nested numeric values."""
        # Arrange
        original_dict = {
            "item1": {"nested_key": 10},
            "item2": {"nested_key": 5},
            "item3": {"nested_key": 15},
        }
        expected_sorted_dict = {
            "item2": {"nested_key": 5},
            "item1": {"nested_key": 10},
            "item3": {"nested_key": 15},
        }

        # Act
        result = sort_nested_dict_values(original_dict, "nested_key")

        # Assert
        assert result == expected_sorted_dict

    def test_sort_nested_dict_values_with_single_item_returns_same(self) -> None:
        """Test sorting single-item dictionary."""
        # Arrange
        original_dict = {"item1": {"value": 10}}
        expected = {"item1": {"value": 10}}

        # Act
        result = sort_nested_dict_values(original_dict, "value")

        # Assert
        assert result == expected

    def test_sort_nested_dict_values_with_string_values_sorts_alphabetically(
        self,
    ) -> None:
        """Test sorting by nested string values."""
        # Arrange
        original_dict = {
            "item1": {"name": "zebra"},
            "item2": {"name": "apple"},
            "item3": {"name": "mango"},
        }
        expected = {
            "item2": {"name": "apple"},
            "item3": {"name": "mango"},
            "item1": {"name": "zebra"},
        }

        # Act
        result = sort_nested_dict_values(original_dict, "name")

        # Assert
        assert result == expected

    def test_sort_dict_keys_with_numeric_string_keys_sorts_descending(self) -> None:
        """Test sorting dictionary by keys in descending order."""
        # Arrange
        dictionary = {"3": "c", "1": "a", "2": "b"}
        expected = {"3": "c", "2": "b", "1": "a"}

        # Act
        result = sort_dict_keys(dictionary)

        # Assert
        assert result == expected

    def test_sort_dict_keys_with_single_key_returns_same(self) -> None:
        """Test sorting single-key dictionary."""
        # Arrange
        dictionary = {"1": "value"}
        expected = {"1": "value"}

        # Act
        result = sort_dict_keys(dictionary)

        # Assert
        assert result == expected

    def test_sort_dict_keys_preserves_values(self) -> None:
        """Test that sorting preserves value associations."""
        # Arrange
        dictionary = {"10": "ten", "5": "five", "20": "twenty"}
        expected = {"20": "twenty", "10": "ten", "5": "five"}

        # Act
        result = sort_dict_keys(dictionary)

        # Assert
        assert result == expected


class TestDictionaryToList:
    """Tests for dictionary to list conversion."""

    def test_dict_to_list_with_simple_nested_key_returns_list(self) -> None:
        """Test extracting values from nested key into list."""
        # Arrange
        data_dict = {
            "item1": {"value": 10},
            "item2": {"value": 20},
            "item3": {"value": 30},
        }
        nested_key = "value"

        # Act
        result = dict_to_list(data_dict, nested_key)

        # Assert
        assert result == [10, 20, 30]

    def test_dict_to_list_with_find_mean_returns_average(self) -> None:
        """Test finding mean of extracted values."""
        # Arrange
        data_dict = {
            "item1": {"value": 10},
            "item2": {"value": 20},
            "item3": {"value": 30},
        }
        nested_key = "value"

        # Act
        result = dict_to_list(data_dict, nested_key, find_mean=True)

        # Assert
        assert result == 20.0

    def test_dict_to_list_with_path_list_navigates_nested_structure(self) -> None:
        """Test extracting values from deeply nested dictionary."""
        # Arrange
        data_dict = {
            "item1": {"level1": {"level2": {"value": 100}}},
            "item2": {"level1": {"level2": {"value": 200}}},
        }
        path_list = ["level1", "level2"]
        nested_key = "value"

        # Act
        result = dict_to_list(data_dict, nested_key, path_list=path_list)

        # Assert
        assert result == [100, 200]

    def test_dict_to_list_with_missing_keys_skips_entries(self) -> None:
        """Test that missing nested keys are skipped."""
        # Arrange
        data_dict = {
            "item1": {"value": 10},
            "item2": {"other": 20},
            "item3": {"value": 30},
        }
        nested_key = "value"

        # Act
        result = dict_to_list(data_dict, nested_key)

        # Assert
        assert result == [10, 30]

    def test_dict_to_list_with_empty_dict_returns_empty_list(self) -> None:
        """Test extracting from empty dictionary."""
        # Arrange
        data_dict: dict = {}
        nested_key = "value"

        # Act
        result = dict_to_list(data_dict, nested_key)

        # Assert
        assert result == []


class TestMatrixStatistics:
    """Tests for matrix statistics calculation."""

    def test_calculate_matrix_statistics_with_numeric_data_returns_stats(self) -> None:
        """Test calculating min, max, and average of matrix columns."""
        # Arrange
        input_dict = {"0": [1.0, 5.0, 3.0], "1": [2.0, 4.0, 8.0], "2": [0.0, 3.0, 5.0]}
        expected_output = {
            "min": [0, 3, 3],
            "max": [2, 5, 8],
            "average": [1.0, 4.0, 5.333333333333333],
        }

        # Act
        result = calculate_matrix_statistics(input_dict)

        # Assert
        assert result == expected_output

    def test_calculate_matrix_statistics_with_single_row_returns_same_values(
        self,
    ) -> None:
        """Test statistics calculation with single row."""
        # Arrange
        input_dict = {"0": [5.0, 10.0, 15.0]}
        expected_min = [5, 10, 15]
        expected_max = [5, 10, 15]
        expected_avg = [5.0, 10.0, 15.0]

        # Act
        result = calculate_matrix_statistics(input_dict)

        # Assert
        assert result["min"] == expected_min
        assert result["max"] == expected_max
        assert result["average"] == expected_avg

    def test_calculate_matrix_statistics_with_negative_values_handles_correctly(
        self,
    ) -> None:
        """Test statistics with negative values."""
        # Arrange
        input_dict = {"0": [-5.0, 0.0, 5.0], "1": [-10.0, -5.0, 0.0]}
        expected_min = [-10, -5, 0]
        expected_max = [-5, 0, 5]

        # Act
        result = calculate_matrix_statistics(input_dict)

        # Assert
        assert result["min"] == expected_min
        assert result["max"] == expected_max

    def test_calculate_matrix_statistics_with_identical_values_returns_same(
        self,
    ) -> None:
        """Test statistics when all values are identical."""
        # Arrange
        input_dict = {"0": [5.0, 5.0, 5.0], "1": [5.0, 5.0, 5.0]}
        expected_value = [5, 5, 5]

        # Act
        result = calculate_matrix_statistics(input_dict)

        # Assert
        assert result["min"] == expected_value
        assert result["max"] == expected_value
        assert result["average"] == [5.0, 5.0, 5.0]


class TestMinMaxScaling:
    """Tests for min-max scaling function."""

    def test_min_max_scale_with_mid_value_returns_half(self) -> None:
        """Test scaling value at midpoint."""
        # Arrange
        value = 5
        min_value = 0
        max_value = 10
        expected = 0.5

        # Act
        result = min_max_scale(value, min_value, max_value)

        # Assert
        assert result == expected

    def test_min_max_scale_with_min_value_returns_zero(self) -> None:
        """Test scaling minimum value."""
        # Arrange
        value = 0
        min_value = 0
        max_value = 10
        expected = 0.0

        # Act
        result = min_max_scale(value, min_value, max_value)

        # Assert
        assert result == expected

    def test_min_max_scale_with_max_value_returns_one(self) -> None:
        """Test scaling maximum value."""
        # Arrange
        value = 10
        min_value = 0
        max_value = 10
        expected = 1.0

        # Act
        result = min_max_scale(value, min_value, max_value)

        # Assert
        assert result == expected

    def test_min_max_scale_with_negative_range_scales_correctly(self) -> None:
        """Test scaling with negative value range."""
        # Arrange
        value = 0
        min_value = -10
        max_value = 10
        expected = 0.5

        # Act
        result = min_max_scale(value, min_value, max_value)

        # Assert
        assert result == expected

    def test_min_max_scale_with_value_outside_range_allows_extrapolation(self) -> None:
        """Test that values outside range are scaled linearly."""
        # Arrange
        value = 15
        min_value = 0
        max_value = 10
        expected = 1.5

        # Act
        result = min_max_scale(value, min_value, max_value)

        # Assert
        assert result == expected


class TestMatrixUpdates:
    """Tests for matrix update functions."""

    def test_update_matrices_with_valid_info_dict_returns_matrices(self) -> None:
        """Test updating matrices from info dictionary."""
        # Arrange
        info_dict = {
            "time1": {
                "sim_list": ["sim1"],
                "network_list": ["net1"],
                "dates_list": ["2025-01-01"],
                "algorithm_list": ["algo1"],
            },
            "time2": {
                "sim_list": ["sim2"],
                "network_list": ["net2"],
                "dates_list": ["2025-01-02"],
                "algorithm_list": ["algo2"],
            },
        }

        # Act
        result = update_matrices(info_dict)

        # Assert
        assert "times_matrix" in result
        assert "sims_matrix" in result
        assert "networks_matrix" in result
        assert "dates_matrix" in result
        assert "algorithms_matrix" in result
        assert len(result["times_matrix"]) == 2

    def test_update_matrices_with_missing_algorithm_handles_gracefully(self) -> None:
        """Test handling missing algorithm_list key."""
        # Arrange
        info_dict = {
            "time1": {
                "sim_list": ["sim1"],
                "network_list": ["net1"],
                "dates_list": ["2025-01-01"],
            },
        }

        # Act
        result = update_matrices(info_dict)

        # Assert
        assert "dates_matrix" in result
        assert len(result["dates_matrix"]) == 2  # Duplicated due to KeyError handling

    def test_update_matrices_with_empty_dict_returns_empty_matrices(self) -> None:
        """Test with empty input dictionary."""
        # Arrange
        info_dict: dict = {}

        # Act
        result = update_matrices(info_dict)

        # Assert
        assert len(result["times_matrix"]) == 0
        assert len(result["sims_matrix"]) == 0

    def test_update_dict_from_list_with_valid_updates_modifies_dict(self) -> None:
        """Test updating dictionary from list of tuples."""
        # Arrange
        input_dict = {"key1": "old_value1", "key2": "old_value2"}
        updates_list = [("key1", "new_value1"), ("key3", "new_value3")]
        expected = {
            "key1": "new_value1",
            "key2": "old_value2",
            "key3": "new_value3",
        }

        # Act
        result = update_dict_from_list(input_dict, updates_list)

        # Assert
        assert result == expected

    def test_update_dict_from_list_with_empty_list_returns_unchanged(self) -> None:
        """Test that empty update list returns original dictionary."""
        # Arrange
        input_dict = {"key1": "value1", "key2": "value2"}
        updates_list: list = []

        # Act
        result = update_dict_from_list(input_dict, updates_list)

        # Assert
        assert result == input_dict

    def test_update_dict_from_list_with_duplicate_keys_uses_last_value(self) -> None:
        """Test that duplicate keys in list use the last value."""
        # Arrange
        input_dict = {"key1": "original"}
        updates_list = [("key1", "first"), ("key1", "second"), ("key1", "third")]
        expected = {"key1": "third"}

        # Act
        result = update_dict_from_list(input_dict, updates_list)

        # Assert
        assert result == expected
