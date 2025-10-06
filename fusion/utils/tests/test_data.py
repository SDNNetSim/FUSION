"""Unit tests for fusion.utils.data module."""

from typing import Any

from fusion.utils.data import sort_dict_keys, sort_nested_dict_values


class TestSortDictKeys:
    """Tests for sort_dict_keys function."""

    def test_sort_dict_keys_with_unordered_keys_sorts_descending(self) -> None:
        """Test sorting dictionary keys in descending order."""
        # Arrange
        input_dict = {"1": "a", "3": "c", "2": "b"}

        # Act
        result = sort_dict_keys(input_dict)

        # Assert
        assert list(result.keys()) == ["3", "2", "1"]
        assert result["3"] == "c"
        assert result["2"] == "b"
        assert result["1"] == "a"

    def test_sort_dict_keys_with_already_sorted_returns_same_order(self) -> None:
        """Test sorting already sorted dictionary."""
        # Arrange
        input_dict = {"3": "c", "2": "b", "1": "a"}

        # Act
        result = sort_dict_keys(input_dict)

        # Assert
        assert list(result.keys()) == ["3", "2", "1"]

    def test_sort_dict_keys_with_single_key_returns_same_dict(self) -> None:
        """Test sorting dictionary with single key."""
        # Arrange
        input_dict = {"1": "value"}

        # Act
        result = sort_dict_keys(input_dict)

        # Assert
        assert result == {"1": "value"}

    def test_sort_dict_keys_with_empty_dict_returns_empty(self) -> None:
        """Test sorting empty dictionary."""
        # Arrange
        input_dict: dict[str, Any] = {}

        # Act
        result = sort_dict_keys(input_dict)

        # Assert
        assert result == {}

    def test_sort_dict_keys_with_numeric_string_keys_sorts_numerically(self) -> None:
        """Test that string keys are sorted by numeric value."""
        # Arrange
        input_dict = {"10": "ten", "2": "two", "100": "hundred"}

        # Act
        result = sort_dict_keys(input_dict)

        # Assert
        # Should sort as numbers: 100, 10, 2
        assert list(result.keys()) == ["100", "10", "2"]

    def test_sort_dict_keys_preserves_values(self) -> None:
        """Test that values are preserved during sorting."""
        # Arrange
        input_dict = {
            "1": {"nested": "value1"},
            "5": [1, 2, 3],
            "3": "simple",
        }

        # Act
        result = sort_dict_keys(input_dict)

        # Assert
        assert result["1"] == {"nested": "value1"}
        assert result["5"] == [1, 2, 3]
        assert result["3"] == "simple"

    def test_sort_dict_keys_with_large_numbers(self) -> None:
        """Test sorting with large numeric string keys."""
        # Arrange
        input_dict = {"1000": "a", "1": "b", "500": "c"}

        # Act
        result = sort_dict_keys(input_dict)

        # Assert
        assert list(result.keys()) == ["1000", "500", "1"]

    def test_sort_dict_keys_returns_new_dict(self) -> None:
        """Test that function returns new dictionary instance."""
        # Arrange
        input_dict = {"2": "a", "1": "b"}

        # Act
        result = sort_dict_keys(input_dict)

        # Assert
        assert result is not input_dict
        # Original dict should be unchanged
        assert list(input_dict.keys()) == ["2", "1"]


class TestSortNestedDictValues:
    """Tests for sort_nested_dict_values function."""

    def test_sort_nested_dict_values_with_numeric_nested_key(self) -> None:
        """Test sorting by numeric nested key value."""
        # Arrange
        input_dict = {
            "item1": {"priority": 3, "name": "third"},
            "item2": {"priority": 1, "name": "first"},
            "item3": {"priority": 2, "name": "second"},
        }

        # Act
        result = sort_nested_dict_values(input_dict, "priority")

        # Assert
        keys = list(result.keys())
        assert keys == ["item2", "item3", "item1"]
        assert result["item2"]["priority"] == 1
        assert result["item3"]["priority"] == 2
        assert result["item1"]["priority"] == 3

    def test_sort_nested_dict_values_with_string_nested_key(self) -> None:
        """Test sorting by string nested key value."""
        # Arrange
        input_dict = {
            "item1": {"name": "charlie"},
            "item2": {"name": "alice"},
            "item3": {"name": "bob"},
        }

        # Act
        result = sort_nested_dict_values(input_dict, "name")

        # Assert
        keys = list(result.keys())
        assert keys == ["item2", "item3", "item1"]

    def test_sort_nested_dict_values_ascending_order(self) -> None:
        """Test that sorting is in ascending order."""
        # Arrange
        input_dict = {
            "a": {"value": 100},
            "b": {"value": 10},
            "c": {"value": 50},
        }

        # Act
        result = sort_nested_dict_values(input_dict, "value")

        # Assert
        values = [v["value"] for v in result.values()]
        assert values == [10, 50, 100]

    def test_sort_nested_dict_values_with_single_item_returns_same(self) -> None:
        """Test sorting single-item dictionary."""
        # Arrange
        input_dict = {"only": {"score": 42}}

        # Act
        result = sort_nested_dict_values(input_dict, "score")

        # Assert
        assert result == input_dict

    def test_sort_nested_dict_values_with_empty_dict_returns_empty(self) -> None:
        """Test sorting empty dictionary."""
        # Arrange
        input_dict: dict[str, Any] = {}

        # Act
        result = sort_nested_dict_values(input_dict, "any_key")

        # Assert
        assert result == {}

    def test_sort_nested_dict_values_with_equal_values_preserves_order(self) -> None:
        """Test sorting with equal nested values preserves relative order."""
        # Arrange
        input_dict = {
            "first": {"rank": 5},
            "second": {"rank": 5},
            "third": {"rank": 5},
        }

        # Act
        result = sort_nested_dict_values(input_dict, "rank")

        # Assert
        # All have same rank, order should be stable
        keys = list(result.keys())
        assert len(keys) == 3
        assert all(result[k]["rank"] == 5 for k in keys)

    def test_sort_nested_dict_values_with_float_values(self) -> None:
        """Test sorting with float nested values."""
        # Arrange
        input_dict = {
            "a": {"score": 3.14},
            "b": {"score": 2.71},
            "c": {"score": 1.41},
        }

        # Act
        result = sort_nested_dict_values(input_dict, "score")

        # Assert
        keys = list(result.keys())
        assert keys == ["c", "b", "a"]

    def test_sort_nested_dict_values_preserves_all_nested_data(self) -> None:
        """Test that all nested data is preserved during sorting."""
        # Arrange
        input_dict = {
            "item1": {"priority": 2, "data": "value1", "extra": [1, 2, 3]},
            "item2": {"priority": 1, "data": "value2", "extra": [4, 5, 6]},
        }

        # Act
        result = sort_nested_dict_values(input_dict, "priority")

        # Assert
        first_key = list(result.keys())[0]
        assert first_key == "item2"
        assert result["item2"]["data"] == "value2"
        assert result["item2"]["extra"] == [4, 5, 6]

    def test_sort_nested_dict_values_with_complex_nested_values(self) -> None:
        """Test sorting with complex nested structures."""
        # Arrange
        input_dict = {
            "x": {"metric": 30, "details": {"a": 1, "b": 2}},
            "y": {"metric": 10, "details": {"a": 3, "b": 4}},
            "z": {"metric": 20, "details": {"a": 5, "b": 6}},
        }

        # Act
        result = sort_nested_dict_values(input_dict, "metric")

        # Assert
        keys = list(result.keys())
        assert keys == ["y", "z", "x"]
        assert result["y"]["details"] == {"a": 3, "b": 4}

    def test_sort_nested_dict_values_returns_new_dict(self) -> None:
        """Test that function returns new dictionary instance."""
        # Arrange
        input_dict = {
            "a": {"val": 2},
            "b": {"val": 1},
        }

        # Act
        result = sort_nested_dict_values(input_dict, "val")

        # Assert
        assert result is not input_dict
