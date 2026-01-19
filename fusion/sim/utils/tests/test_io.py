"""Unit tests for fusion.sim.utils.io module."""

import os
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from ..io import modify_multiple_json_values, parse_yaml_file


class TestYAMLParsing:
    """Tests for YAML file parsing functionality."""

    @patch("builtins.open", new_callable=mock_open, read_data="key: value")
    @patch("yaml.safe_load")
    def test_parse_yaml_file_with_valid_file_returns_dict(self, mock_yaml_load: MagicMock, mock_file: Any) -> None:
        """Test successful parsing of valid YAML file.

        :param mock_yaml_load: Mock for yaml.safe_load
        :type mock_yaml_load: MagicMock
        :param mock_file: Mock for file operations
        :type mock_file: Any
        """
        # Arrange
        mock_yaml_load.return_value = {"key": "value"}
        yaml_file = "test_file.yaml"

        # Act
        result = parse_yaml_file(yaml_file)

        # Assert
        assert result == {"key": "value"}
        mock_file.assert_called_once_with(yaml_file, encoding="utf-8")
        mock_yaml_load.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: content")
    @patch("yaml.safe_load")
    def test_parse_yaml_file_with_invalid_yaml_returns_exception(self, mock_yaml_load: MagicMock, mock_file: Any) -> None:
        """Test parsing of invalid YAML returns exception.

        :param mock_yaml_load: Mock for yaml.safe_load
        :type mock_yaml_load: MagicMock
        :param mock_file: Mock for file operations
        :type mock_file: Any
        """
        # Arrange
        yaml_error = yaml.YAMLError("Invalid YAML")
        mock_yaml_load.side_effect = yaml_error
        yaml_file = "invalid.yaml"

        # Act
        result = parse_yaml_file(yaml_file)

        # Assert
        assert isinstance(result, yaml.YAMLError)
        mock_file.assert_called_once_with(yaml_file, encoding="utf-8")

    @patch("builtins.open", new_callable=mock_open, read_data="")
    @patch("yaml.safe_load")
    def test_parse_yaml_file_with_empty_file_returns_none(self, mock_yaml_load: MagicMock, mock_file: Any) -> None:
        """Test parsing of empty YAML file.

        :param mock_yaml_load: Mock for yaml.safe_load
        :type mock_yaml_load: MagicMock
        :param mock_file: Mock for file operations
        :type mock_file: Any
        """
        # Arrange
        mock_yaml_load.return_value = None
        yaml_file = "empty.yaml"

        # Act
        result = parse_yaml_file(yaml_file)

        # Assert
        assert result is None
        mock_file.assert_called_once_with(yaml_file, encoding="utf-8")

    @patch("builtins.open", new_callable=mock_open, read_data="items:\n  - one\n  - two")
    @patch("yaml.safe_load")
    def test_parse_yaml_file_with_list_structure_returns_list(self, mock_yaml_load: MagicMock, mock_file: Any) -> None:
        """Test parsing YAML file with list structure.

        :param mock_yaml_load: Mock for yaml.safe_load
        :type mock_yaml_load: MagicMock
        :param mock_file: Mock for file operations
        :type mock_file: Any
        """
        # Arrange
        mock_yaml_load.return_value = {"items": ["one", "two"]}
        yaml_file = "list.yaml"

        # Act
        result = parse_yaml_file(yaml_file)

        # Assert
        assert result == {"items": ["one", "two"]}
        assert isinstance(result["items"], list)


class TestJSONModification:
    """Tests for JSON file modification functionality."""

    @patch("json.load")
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_multiple_json_values_with_valid_keys_updates_correctly(
        self, mock_file: Any, mock_json_dump: MagicMock, mock_json_load: MagicMock
    ) -> None:
        """Test successful modification of multiple JSON values.

        :param mock_file: Mock for file operations
        :type mock_file: Any
        :param mock_json_dump: Mock for json.dump
        :type mock_json_dump: MagicMock
        :param mock_json_load: Mock for json.load
        :type mock_json_load: MagicMock
        """
        # Arrange
        mock_json_load.return_value = {"key1": "old_value1", "key2": "old_value2"}
        trial_num = 1
        file_path = "test_dir"
        update_list = [("key1", "new_value1"), ("key2", "new_value2")]
        expected_data = {"key1": "new_value1", "key2": "new_value2"}

        # Act
        modify_multiple_json_values(trial_num, file_path, update_list)

        # Assert
        mock_file.assert_any_call(os.path.join(file_path, "sim_input_s1.json"), encoding="utf-8")
        mock_file.assert_any_call(os.path.join(file_path, "sim_input_s2.json"), "w", encoding="utf-8")
        mock_json_dump.assert_called_once()
        args = mock_json_dump.call_args[0]
        assert args[0] == expected_data

    @patch("json.load")
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_multiple_json_values_with_invalid_key_raises_error(self, mock_file: Any, mock_json_load: MagicMock) -> None:
        """Test that invalid key raises KeyError.

        :param mock_file: Mock for file operations
        :type mock_file: Any
        :param mock_json_load: Mock for json.load
        :type mock_json_load: MagicMock
        """
        # Arrange
        mock_json_load.return_value = {"key1": "value1"}
        trial_num = 1
        file_path = "test_dir"
        update_list = [("invalid_key", "new_value")]

        # Act & Assert
        with pytest.raises(KeyError, match="Key 'invalid_key' not found"):
            modify_multiple_json_values(trial_num, file_path, update_list)

    @patch("json.load")
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_multiple_json_values_with_empty_list_keeps_original(
        self, mock_file: Any, mock_json_dump: MagicMock, mock_json_load: MagicMock
    ) -> None:
        """Test that empty update list keeps original data.

        :param mock_file: Mock for file operations
        :type mock_file: Any
        :param mock_json_dump: Mock for json.dump
        :type mock_json_dump: MagicMock
        :param mock_json_load: Mock for json.load
        :type mock_json_load: MagicMock
        """
        # Arrange
        original_data = {"key1": "value1", "key2": "value2"}
        mock_json_load.return_value = original_data.copy()
        trial_num = 1
        file_path = "test_dir"
        update_list: list = []

        # Act
        modify_multiple_json_values(trial_num, file_path, update_list)

        # Assert
        args = mock_json_dump.call_args[0]
        assert args[0] == original_data

    @patch("json.load")
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_multiple_json_values_with_different_types_updates_correctly(
        self, mock_file: Any, mock_json_dump: MagicMock, mock_json_load: MagicMock
    ) -> None:
        """Test updating values with different data types.

        :param mock_file: Mock for file operations
        :type mock_file: Any
        :param mock_json_dump: Mock for json.dump
        :type mock_json_dump: MagicMock
        :param mock_json_load: Mock for json.load
        :type mock_json_load: MagicMock
        """
        # Arrange
        mock_json_load.return_value = {"key1": "string", "key2": 123, "key3": True}
        trial_num = 2
        file_path = "test_dir"
        update_list = [("key1", 456), ("key2", False), ("key3", "new_string")]
        expected_data = {"key1": 456, "key2": False, "key3": "new_string"}

        # Act
        modify_multiple_json_values(trial_num, file_path, update_list)

        # Assert
        args = mock_json_dump.call_args[0]
        assert args[0] == expected_data

    @patch("json.load")
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_multiple_json_values_creates_correct_output_filename(
        self, mock_file: Any, mock_json_dump: MagicMock, mock_json_load: MagicMock
    ) -> None:
        """Test that output filename is correctly generated.

        :param mock_file: Mock for file operations
        :type mock_file: Any
        :param mock_json_dump: Mock for json.dump
        :type mock_json_dump: MagicMock
        :param mock_json_load: Mock for json.load
        :type mock_json_load: MagicMock
        """
        # Arrange
        mock_json_load.return_value = {"key": "value"}
        trial_num = 5
        file_path = "test_dir"
        update_list = [("key", "new_value")]
        expected_output = os.path.join(file_path, "sim_input_s6.json")

        # Act
        modify_multiple_json_values(trial_num, file_path, update_list)

        # Assert
        mock_file.assert_any_call(expected_output, "w", encoding="utf-8")
