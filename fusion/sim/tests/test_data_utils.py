"""Unit tests for fusion/sim/utils/data.py module."""

from typing import Any

from fusion.sim.utils.data import (
    calculate_matrix_statistics,
    dict_to_list,
    min_max_scale,
    sort_dict_keys,
    sort_nested_dict_values,
    update_dict_from_list,
    update_matrices,
)


class TestSortNestedDictValues:
    """Tests for sort_nested_dict_values function."""

    def test_sort_nested_dict_with_numeric_values_sorts_ascending(self) -> None:
        """Test that dict is sorted by nested numeric values in ascending order."""
        # Arrange
        original_dict = {
            "item1": {"score": 10},
            "item2": {"score": 5},
            "item3": {"score": 15},
        }
        nested_key = "score"

        # Act
        result = sort_nested_dict_values(original_dict, nested_key)

        # Assert
        expected = {
            "item2": {"score": 5},
            "item1": {"score": 10},
            "item3": {"score": 15},
        }
        assert result == expected

    def test_sort_nested_dict_with_string_values_sorts_alphabetically(self) -> None:
        """Test that dict is sorted by nested string values alphabetically."""
        # Arrange
        original_dict = {
            "item1": {"name": "Charlie"},
            "item2": {"name": "Alice"},
            "item3": {"name": "Bob"},
        }
        nested_key = "name"

        # Act
        result = sort_nested_dict_values(original_dict, nested_key)

        # Assert
        expected = {
            "item2": {"name": "Alice"},
            "item3": {"name": "Bob"},
            "item1": {"name": "Charlie"},
        }
        assert result == expected


class TestSortDictKeys:
    """Tests for sort_dict_keys function."""

    def test_sort_dict_keys_with_numeric_strings_sorts_descending(self) -> None:
        """Test that dict keys are sorted in descending numeric order."""
        # Arrange
        dictionary = {"3": "c", "1": "a", "2": "b"}

        # Act
        result = sort_dict_keys(dictionary)

        # Assert
        expected = {"3": "c", "2": "b", "1": "a"}
        assert result == expected

    def test_sort_dict_keys_with_single_key_returns_same_dict(self) -> None:
        """Test that single key dict returns unchanged."""
        # Arrange
        dictionary = {"1": "value"}

        # Act
        result = sort_dict_keys(dictionary)

        # Assert
        assert result == {"1": "value"}


class TestDictToList:
    """Tests for dict_to_list function."""

    def test_dict_to_list_with_simple_nested_key_returns_list(self) -> None:
        """Test that values are extracted into list for simple nested key."""
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
        """Test that mean is calculated when find_mean is True."""
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

    def test_dict_to_list_with_path_list_navigates_nested_dict(self) -> None:
        """Test that path_list is used to navigate nested structure."""
        # Arrange
        data_dict = {
            "item1": {"nested": {"value": 10}},
            "item2": {"nested": {"value": 20}},
        }
        nested_key = "value"
        path_list = ["nested"]

        # Act
        result = dict_to_list(data_dict, nested_key, path_list=path_list)

        # Assert
        assert result == [10, 20]

    def test_dict_to_list_with_missing_key_skips_item(self) -> None:
        """Test that items without the nested key are skipped."""
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


class TestCalculateMatrixStatistics:
    """Tests for calculate_matrix_statistics function."""

    def test_calculate_stats_with_multiple_episodes_returns_min_max_avg(self) -> None:
        """Test that min, max, and average are calculated correctly."""
        # Arrange
        input_dict = {"0": [1.0, 5.0, 3.0], "1": [2.0, 4.0, 8.0], "2": [0.0, 3.0, 5.0]}

        # Act
        result = calculate_matrix_statistics(input_dict)

        # Assert
        expected = {
            "min": [0, 3, 3],
            "max": [2, 5, 8],
            "average": [1.0, 4.0, 5.333333333333333],
        }
        assert result == expected

    def test_calculate_stats_with_single_episode_returns_same_values(self) -> None:
        """Test that single episode returns same values for min/max/avg."""
        # Arrange
        input_dict = {"0": [1.0, 2.0, 3.0]}

        # Act
        result = calculate_matrix_statistics(input_dict)

        # Assert
        assert result["min"] == [1.0, 2.0, 3.0]
        assert result["max"] == [1.0, 2.0, 3.0]
        assert result["average"] == [1.0, 2.0, 3.0]


class TestMinMaxScale:
    """Tests for min_max_scale function."""

    def test_min_max_scale_with_midpoint_value_returns_half(self) -> None:
        """Test that midpoint value between min and max returns 0.5."""
        # Arrange
        value = 5.0
        min_value = 0.0
        max_value = 10.0

        # Act
        result = min_max_scale(value, min_value, max_value)

        # Assert
        assert result == 0.5

    def test_min_max_scale_with_min_value_returns_zero(self) -> None:
        """Test that minimum value returns 0."""
        # Arrange
        value = 0.0
        min_value = 0.0
        max_value = 10.0

        # Act
        result = min_max_scale(value, min_value, max_value)

        # Assert
        assert result == 0.0

    def test_min_max_scale_with_max_value_returns_one(self) -> None:
        """Test that maximum value returns 1."""
        # Arrange
        value = 10.0
        min_value = 0.0
        max_value = 10.0

        # Act
        result = min_max_scale(value, min_value, max_value)

        # Assert
        assert result == 1.0


class TestUpdateDictFromList:
    """Tests for update_dict_from_list function."""

    def test_update_dict_with_new_keys_adds_entries(self) -> None:
        """Test that new keys are added to dictionary."""
        # Arrange
        input_dict: dict[str, Any] = {}
        updates_list = [("key1", "value1"), ("key2", "value2")]

        # Act
        result = update_dict_from_list(input_dict, updates_list)

        # Assert
        assert result == {"key1": "value1", "key2": "value2"}

    def test_update_dict_with_existing_keys_overwrites_values(self) -> None:
        """Test that existing keys are overwritten with new values."""
        # Arrange
        input_dict = {"key1": "old_value"}
        updates_list = [("key1", "new_value")]

        # Act
        result = update_dict_from_list(input_dict, updates_list)

        # Assert
        assert result == {"key1": "new_value"}


class TestUpdateMatrices:
    """Tests for update_matrices function."""

    def test_update_matrices_with_valid_info_dict_returns_matrices(self) -> None:
        """Test that info dict is converted to matrices correctly."""
        # Arrange
        info_dict = {
            "time1": {
                "sim_list": ["s1", "s2"],
                "network_list": ["net1"],
                "dates_list": ["2024-01-01"],
                "algorithm_list": ["algo1"],
            }
        }

        # Act
        result = update_matrices(info_dict)

        # Assert
        assert result["times_matrix"] == [["time1"]]
        assert result["sims_matrix"] == [["s1", "s2"]]
        assert result["networks_matrix"] == [["net1"]]
        assert result["dates_matrix"] == [["2024-01-01"]]
        assert result["algorithms_matrix"] == [["algo1"]]

    def test_update_matrices_without_algorithm_list_uses_dates_fallback(self) -> None:
        """Test that missing algorithm_list uses dates_list as fallback."""
        # Arrange
        info_dict = {
            "time1": {
                "sim_list": ["s1"],
                "network_list": ["net1"],
                "dates_list": ["2024-01-01"],
            }
        }

        # Act
        result = update_matrices(info_dict)

        # Assert
        # dates_matrix gets dates_list once normally, then again as fallback
        assert result["dates_matrix"] == [["2024-01-01"], ["2024-01-01"]]
