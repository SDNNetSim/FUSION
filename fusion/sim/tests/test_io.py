"""Unit tests for fusion/sim/utils/io.py module."""

from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from fusion.sim.utils.io import modify_multiple_json_values, parse_yaml_file


class TestParseYamlFile:
    """Tests for parse_yaml_file function."""

    @patch("builtins.open", new_callable=mock_open, read_data="key: value")
    @patch("fusion.sim.utils.io.yaml.safe_load")
    def test_parse_yaml_file_with_valid_file_returns_data(
        self, mock_yaml_load: MagicMock, mock_file: MagicMock
    ) -> None:
        """Test that valid YAML file is parsed successfully."""
        # Arrange
        mock_yaml_load.return_value = {"key": "value"}
        yaml_file = "test_config.yaml"

        # Act
        result = parse_yaml_file(yaml_file)

        # Assert
        assert result == {"key": "value"}
        mock_file.assert_called_once_with(yaml_file, encoding="utf-8")
        mock_yaml_load.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: data:")
    @patch("fusion.sim.utils.io.yaml.safe_load")
    def test_parse_yaml_file_with_invalid_yaml_returns_exception(
        self, mock_yaml_load: MagicMock, mock_file: MagicMock
    ) -> None:
        """Test that invalid YAML file returns YAMLError."""
        # Arrange
        yaml_error = yaml.YAMLError("Invalid YAML")
        mock_yaml_load.side_effect = yaml_error
        yaml_file = "invalid.yaml"

        # Act
        result = parse_yaml_file(yaml_file)

        # Assert
        assert isinstance(result, yaml.YAMLError)
        mock_file.assert_called_once_with(yaml_file, encoding="utf-8")


class TestModifyMultipleJsonValues:
    """Tests for modify_multiple_json_values function."""

    @patch("json.load", return_value={"key1": "value1", "key2": "value2"})
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_json_with_valid_keys_updates_values(
        self, mock_file: MagicMock, mock_json_dump: MagicMock, mock_json_load: MagicMock
    ) -> None:
        """Test that JSON values are modified correctly for valid keys."""
        # Arrange
        trial_num = 1
        file_path = "test_dir"
        update_list = [("key1", "new_value1"), ("key2", "new_value2")]
        mock_file_handle = mock_file.return_value.__enter__.return_value

        # Act
        modify_multiple_json_values(trial_num, file_path, update_list)

        # Assert
        # Verify read operation
        mock_file.assert_any_call("test_dir/sim_input_s1.json", encoding="utf-8")
        # Verify write operation
        mock_file.assert_any_call(
            "test_dir/sim_input_s2.json", "w", encoding="utf-8"
        )
        # Verify data was dumped correctly
        mock_json_dump.assert_called_once_with(
            {"key1": "new_value1", "key2": "new_value2"}, mock_file_handle, indent=4
        )

    @patch("json.load", return_value={"key1": "value1"})
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_json_with_invalid_key_raises_key_error(
        self, mock_file: MagicMock, mock_json_load: MagicMock
    ) -> None:
        """Test that KeyError is raised when key not found in JSON."""
        # Arrange
        trial_num = 1
        file_path = "test_dir"
        update_list = [("nonexistent_key", "value")]

        # Act & Assert
        with pytest.raises(KeyError, match="Key 'nonexistent_key' not found"):
            modify_multiple_json_values(trial_num, file_path, update_list)

    @patch("json.load", return_value={"key1": "value1", "key2": "value2"})
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_json_with_empty_update_list_preserves_data(
        self, mock_file: MagicMock, mock_json_dump: MagicMock, mock_json_load: MagicMock
    ) -> None:
        """Test that empty update list preserves original data."""
        # Arrange
        trial_num = 1
        file_path = "test_dir"
        update_list: list[tuple[str, str]] = []
        mock_file_handle = mock_file.return_value.__enter__.return_value

        # Act
        modify_multiple_json_values(trial_num, file_path, update_list)

        # Assert
        mock_json_dump.assert_called_once_with(
            {"key1": "value1", "key2": "value2"}, mock_file_handle, indent=4
        )

    @patch("json.load", return_value={"key1": "value1"})
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_modify_json_increments_trial_number_in_filename(
        self, mock_file: MagicMock, mock_json_dump: MagicMock, mock_json_load: MagicMock
    ) -> None:
        """Test that output filename uses incremented trial number."""
        # Arrange
        trial_num = 5
        file_path = "test_dir"
        update_list = [("key1", "new_value")]

        # Act
        modify_multiple_json_values(trial_num, file_path, update_list)

        # Assert
        mock_file.assert_any_call("test_dir/sim_input_s1.json", encoding="utf-8")
        mock_file.assert_any_call(
            "test_dir/sim_input_s6.json", "w", encoding="utf-8"
        )
