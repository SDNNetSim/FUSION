"""Unit tests for fusion.modules.rl.utils.sim_filters module."""

import json
from typing import Any
from unittest.mock import mock_open, patch

from fusion.modules.rl.utils.sim_filters import (
    _and_filters,
    _check_filters,
    _not_filters,
    _or_filters,
    find_times,
)


class TestNotFilters:
    """Tests for _not_filters function."""

    def test_not_filters_passes_when_value_not_matched(self) -> None:
        """Test that _not_filters returns True when value doesn't match."""
        # Arrange
        filter_dict = {"not_filter_list": [["key1", "reject_value"]]}
        file_dict = {"key1": "different_value"}

        # Act
        result = _not_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_not_filters_fails_when_value_matched(self) -> None:
        """Test that _not_filters returns False when value matches."""
        # Arrange
        filter_dict = {"not_filter_list": [["key1", "reject_value"]]}
        file_dict = {"key1": "reject_value"}

        # Act
        result = _not_filters(filter_dict, file_dict)

        # Assert
        assert result is False

    def test_not_filters_with_nested_keys(self) -> None:
        """Test _not_filters with nested dictionary traversal."""
        # Arrange
        filter_dict = {"not_filter_list": [["level1", "level2", "reject"]]}
        file_dict = {"level1": {"level2": "reject"}}

        # Act
        result = _not_filters(filter_dict, file_dict)

        # Assert
        assert result is False

    def test_not_filters_with_missing_key(self) -> None:
        """Test _not_filters when key path doesn't exist."""
        # Arrange
        filter_dict = {"not_filter_list": [["missing_key", "value"]]}
        file_dict = {"other_key": "value"}

        # Act
        result = _not_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_not_filters_with_empty_filter_list(self) -> None:
        """Test _not_filters with empty filter list."""
        # Arrange
        filter_dict: dict[str, list[list[Any]]] = {"not_filter_list": []}
        file_dict = {"key": "value"}

        # Act
        result = _not_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_not_filters_with_multiple_filters(self) -> None:
        """Test _not_filters with multiple filter conditions."""
        # Arrange
        filter_dict = {"not_filter_list": [["key1", "reject1"], ["key2", "reject2"]]}
        file_dict = {"key1": "accept", "key2": "reject2"}

        # Act
        result = _not_filters(filter_dict, file_dict)

        # Assert
        assert result is False


class TestOrFilters:
    """Tests for _or_filters function."""

    def test_or_filters_passes_when_one_matches(self) -> None:
        """Test that _or_filters returns True when one filter matches."""
        # Arrange
        filter_dict = {"or_filter_list": [["key1", "value1"], ["key2", "value2"]]}
        file_dict = {"key1": "value1", "key2": "different"}

        # Act
        result = _or_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_or_filters_fails_when_none_match(self) -> None:
        """Test that _or_filters returns False when no filters match."""
        # Arrange
        filter_dict = {"or_filter_list": [["key1", "value1"], ["key2", "value2"]]}
        file_dict = {"key1": "different1", "key2": "different2"}

        # Act
        result = _or_filters(filter_dict, file_dict)

        # Assert
        assert result is False

    def test_or_filters_with_empty_list_returns_true(self) -> None:
        """Test that _or_filters returns True with empty or_filter_list."""
        # Arrange
        filter_dict: dict[str, list[list[Any]]] = {}
        file_dict = {"key": "value"}

        # Act
        result = _or_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_or_filters_with_nested_keys(self) -> None:
        """Test _or_filters with nested dictionary traversal."""
        # Arrange
        filter_dict = {"or_filter_list": [["level1", "level2", "target"]]}
        file_dict = {"level1": {"level2": "target"}}

        # Act
        result = _or_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_or_filters_with_missing_key_path(self) -> None:
        """Test _or_filters when key path doesn't exist."""
        # Arrange
        filter_dict = {"or_filter_list": [["missing", "key", "value"]]}
        file_dict = {"other": "data"}

        # Act
        result = _or_filters(filter_dict, file_dict)

        # Assert
        assert result is False


class TestAndFilters:
    """Tests for _and_filters function."""

    def test_and_filters_passes_when_all_match(self) -> None:
        """Test that _and_filters returns True when all filters match."""
        # Arrange
        filter_dict = {"and_filter_list": [["key1", "value1"], ["key2", "value2"]]}
        file_dict = {"key1": "value1", "key2": "value2"}

        # Act
        result = _and_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_and_filters_fails_when_one_doesnt_match(self) -> None:
        """Test that _and_filters returns False when one filter doesn't match."""
        # Arrange
        filter_dict = {"and_filter_list": [["key1", "value1"], ["key2", "value2"]]}
        file_dict = {"key1": "value1", "key2": "different"}

        # Act
        result = _and_filters(filter_dict, file_dict)

        # Assert
        assert result is False

    def test_and_filters_with_empty_list_returns_true(self) -> None:
        """Test that _and_filters returns True with empty and_filter_list."""
        # Arrange
        filter_dict: dict[str, list[list[Any]]] = {}
        file_dict = {"key": "value"}

        # Act
        result = _and_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_and_filters_with_nested_keys(self) -> None:
        """Test _and_filters with nested dictionary traversal."""
        # Arrange
        filter_dict = {"and_filter_list": [["level1", "level2", "value"]]}
        file_dict = {"level1": {"level2": "value"}}

        # Act
        result = _and_filters(filter_dict, file_dict)

        # Assert
        assert result is True

    def test_and_filters_with_missing_key_returns_false(self) -> None:
        """Test _and_filters when key path doesn't exist."""
        # Arrange
        filter_dict = {"and_filter_list": [["missing_key", "value"]]}
        file_dict = {"other_key": "data"}

        # Act
        result = _and_filters(filter_dict, file_dict)

        # Assert
        assert result is False


class TestCheckFilters:
    """Tests for _check_filters function."""

    def test_check_filters_passes_all_conditions(self) -> None:
        """Test _check_filters when all filter conditions pass."""
        # Arrange
        filter_dict = {
            "and_filter_list": [["network", "nsfnet"]],
            "or_filter_list": [["algo", "ppo"], ["algo", "dqn"]],
            "not_filter_list": [["status", "failed"]],
        }
        file_dict = {"network": "nsfnet", "algo": "ppo", "status": "success"}

        # Act
        result = _check_filters(file_dict, filter_dict)

        # Assert
        assert result is True

    def test_check_filters_fails_and_condition(self) -> None:
        """Test _check_filters when and condition fails."""
        # Arrange
        filter_dict = {
            "and_filter_list": [["network", "nsfnet"]],
            "or_filter_list": [],
            "not_filter_list": [],
        }
        file_dict = {"network": "different"}

        # Act
        result = _check_filters(file_dict, filter_dict)

        # Assert
        assert result is False

    def test_check_filters_fails_or_condition(self) -> None:
        """Test _check_filters when or condition fails."""
        # Arrange
        filter_dict = {
            "and_filter_list": [],
            "or_filter_list": [["algo", "ppo"]],
            "not_filter_list": [],
        }
        file_dict = {"algo": "different"}

        # Act
        result = _check_filters(file_dict, filter_dict)

        # Assert
        assert result is False

    def test_check_filters_fails_not_condition(self) -> None:
        """Test _check_filters when not condition triggers."""
        # Arrange
        filter_dict = {
            "and_filter_list": [],
            "or_filter_list": [],
            "not_filter_list": [["status", "failed"]],
        }
        file_dict = {"status": "failed"}

        # Act
        result = _check_filters(file_dict, filter_dict)

        # Assert
        assert result is False


class TestFindTimes:
    """Tests for find_times function."""

    @patch("fusion.modules.rl.utils.sim_filters.update_matrices")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_find_times_with_valid_directories(self, mock_isdir: Any, mock_listdir: Any, mock_file: Any, mock_update: Any) -> None:
        """Test find_times with valid directory structure."""
        # Arrange
        dates_dict = {"0101": "nsfnet"}
        filter_dict: dict[str, list[list[Any]]] = {
            "and_filter_list": [],
            "or_filter_list": [],
            "not_filter_list": [],
        }

        mock_isdir.return_value = True
        mock_listdir.side_effect = [
            ["20240101_120000"],  # times_list
            ["sim_input_0.json"],  # input_file_list
        ]

        file_content = json.dumps({"path_algorithm": "ppo", "network": "nsfnet"})
        mock_file.return_value.read.return_value = file_content
        mock_update.return_value = {
            "times": ["20240101_120000"],
            "algorithms_matrix": [],
        }

        # Act
        result = find_times(dates_dict, filter_dict)

        # Assert
        assert "algorithms_matrix" in result
        mock_update.assert_called_once()

    @patch("fusion.modules.rl.utils.sim_filters.update_matrices")
    @patch("os.path.isdir")
    def test_find_times_skips_nonexistent_directories(self, mock_isdir: Any, mock_update: Any) -> None:
        """Test that find_times skips non-existent directories."""
        # Arrange
        dates_dict = {"0101": "nonexistent_network"}
        filter_dict: dict[str, list[list[Any]]] = {}

        mock_isdir.return_value = False
        mock_update.return_value = {"algorithms_matrix": []}

        # Act
        result = find_times(dates_dict, filter_dict)

        # Assert
        assert result["algorithms_matrix"] == []
        mock_update.assert_called_once_with(info_dict={})

    @patch("fusion.modules.rl.utils.sim_filters.update_matrices")
    @patch("builtins.print")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_find_times_skips_invalid_json(
        self,
        mock_isdir: Any,
        mock_listdir: Any,
        mock_file: Any,
        mock_print: Any,
        mock_update: Any,
    ) -> None:
        """Test that find_times skips files with invalid JSON."""
        # Arrange
        dates_dict = {"0101": "nsfnet"}
        filter_dict: dict[str, list[list[Any]]] = {}

        mock_isdir.return_value = True
        mock_listdir.side_effect = [["20240101_120000"], ["sim_input_0.json"]]
        side_effect = json.JSONDecodeError("Invalid", "", 0)
        mock_file.return_value.__enter__.return_value.read.side_effect = side_effect
        mock_update.return_value = {}

        # Act
        find_times(dates_dict, filter_dict)

        # Assert
        assert any("Skipping file" in str(call) for call in mock_print.call_args_list)

    @patch("fusion.modules.rl.utils.sim_filters.update_matrices")
    @patch("builtins.open")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_find_times_applies_filters(self, mock_isdir: Any, mock_listdir: Any, mock_file: Any, mock_update: Any) -> None:
        """Test that find_times correctly applies filter conditions."""
        # Arrange
        dates_dict = {"0101": "nsfnet"}
        filter_dict = {"and_filter_list": [["algo", "ppo"]]}

        mock_isdir.return_value = True
        mock_listdir.side_effect = [["20240101_120000"], ["sim_input_0.json"]]

        file_content = json.dumps({"algo": "dqn"})  # Won't match filter
        mock_file.return_value.__enter__.return_value = mock_open(read_data=file_content)()
        mock_update.return_value = {}

        # Act
        find_times(dates_dict, filter_dict)

        # Assert
        # Should have empty info_dict because filter rejected the file
        mock_update.assert_called_once_with(info_dict={})
