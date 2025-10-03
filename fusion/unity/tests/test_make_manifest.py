"""Unit tests for fusion.unity.make_manifest module."""

import csv
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fusion.unity.make_manifest import (
    _cast,
    _encode,
    _expand_grid,
    _explicit,
    _fetch,
    _is_rl,
    _parse_literal,
    _read_spec,
    _resolve_spec_path,
    _str_to_bool,
    _to_list,
    _validate_keys,
    _validate_resource_keys,
    _write_csv,
)


class TestStrToBool:
    """Tests for _str_to_bool function."""

    def test_str_to_bool_with_true_values_returns_true(self) -> None:
        """Test _str_to_bool returns True for recognized true values."""
        # Arrange
        true_values = ["true", "yes", "1", "TRUE", "YES", "True", "Yes"]

        # Act & Assert
        for value in true_values:
            assert _str_to_bool(value) is True

    def test_str_to_bool_with_false_values_returns_false(self) -> None:
        """Test _str_to_bool returns False for unrecognized values."""
        # Arrange
        false_values = ["false", "no", "0", "FALSE", "NO", "anything_else", ""]

        # Act & Assert
        for value in false_values:
            assert _str_to_bool(value) is False


class TestParseLiteral:
    """Tests for _parse_literal function."""

    def test_parse_literal_with_list_returns_list(self) -> None:
        """Test _parse_literal correctly parses list literals."""
        # Arrange
        input_val = "[1, 2, 3]"
        expected = [1, 2, 3]

        # Act
        result = _parse_literal(input_val)

        # Assert
        assert result == expected

    def test_parse_literal_with_dict_returns_dict(self) -> None:
        """Test _parse_literal correctly parses dict literals."""
        # Arrange
        input_val = '{"key": "value"}'
        expected = {"key": "value"}

        # Act
        result = _parse_literal(input_val)

        # Assert
        assert result == expected

    def test_parse_literal_with_number_returns_number(self) -> None:
        """Test _parse_literal correctly parses numeric literals."""
        # Arrange
        assert _parse_literal("42") == 42
        assert _parse_literal("3.14") == 3.14

    def test_parse_literal_with_string_literal_returns_string(self) -> None:
        """Test _parse_literal correctly parses string literals."""
        # Arrange
        input_val = '"string"'
        expected = "string"

        # Act
        result = _parse_literal(input_val)

        # Assert
        assert result == expected

    def test_parse_literal_with_invalid_syntax_returns_original_string(self) -> None:
        """Test _parse_literal returns original string for invalid literals."""
        # Arrange
        invalid_inputs = ["invalid_literal", "func()", "1 + 2"]

        # Act & Assert
        for input_val in invalid_inputs:
            result = _parse_literal(input_val)
            assert result == input_val


class TestCast:
    """Tests for _cast function."""

    def test_cast_with_bool_type_and_string_returns_bool(self) -> None:
        """Test _cast converts string to bool when type is bool."""
        # Arrange
        param_types = {"enable_feature": bool}

        # Act & Assert
        with patch("fusion.unity.make_manifest._PARAM_TYPES", param_types):
            assert _cast("enable_feature", "true") is True
            assert _cast("enable_feature", "false") is False
            assert _cast("enable_feature", True) is True

    def test_cast_with_list_type_and_string_returns_list(self) -> None:
        """Test _cast parses string to list when type is list."""
        # Arrange
        param_types = {"algorithms": list}

        # Act
        with patch("fusion.unity.make_manifest._PARAM_TYPES", param_types):
            result = _cast("algorithms", '["ppo", "dqn"]')

        # Assert
        assert result == ["ppo", "dqn"]

    def test_cast_with_unknown_param_returns_original_value(self) -> None:
        """Test _cast returns original value for unknown parameters."""
        # Arrange
        value = "some_value"

        # Act
        result = _cast("unknown_param", value)

        # Assert
        assert result == value

    def test_cast_with_int_type_converts_string_to_int(self) -> None:
        """Test _cast converts string to int when type is int."""
        # Arrange
        param_types = {"num_episodes": int}

        # Act
        with patch("fusion.unity.make_manifest._PARAM_TYPES", param_types):
            result = _cast("num_episodes", "100")

        # Assert
        assert result == 100
        assert isinstance(result, int)


class TestEncode:
    """Tests for _encode function."""

    def test_encode_with_true_bool_returns_true_string(self) -> None:
        """Test _encode converts True to 'true' string."""
        # Arrange & Act
        result = _encode(True)

        # Assert
        assert result == "true"

    def test_encode_with_false_bool_returns_false_string(self) -> None:
        """Test _encode converts False to 'false' string."""
        # Arrange & Act
        result = _encode(False)

        # Assert
        assert result == "false"

    def test_encode_with_list_returns_json_string(self) -> None:
        """Test _encode converts list to compact JSON string."""
        # Arrange
        input_list = [1, 2, 3]

        # Act
        result = _encode(input_list)

        # Assert
        assert result == "[1,2,3]"

    def test_encode_with_dict_returns_json_string(self) -> None:
        """Test _encode converts dict to compact JSON string."""
        # Arrange
        input_dict = {"key": "value"}

        # Act
        result = _encode(input_dict)

        # Assert
        assert result == '{"key":"value"}'

    def test_encode_with_float_removes_trailing_zeros(self) -> None:
        """Test _encode removes trailing zeros from float values."""
        # Arrange & Act & Assert
        assert _encode(3.14159) == "3.14159"
        assert _encode(1.0) == "1"
        assert _encode(0.000057) == "0.000057"
        assert _encode(1000.0) == "1000"

    def test_encode_with_string_returns_string(self) -> None:
        """Test _encode returns string as-is."""
        # Arrange
        input_str = "test_string"

        # Act
        result = _encode(input_str)

        # Assert
        assert result == "test_string"


class TestIsRl:
    """Tests for _is_rl function."""

    def test_is_rl_with_rl_algorithm_returns_yes(self) -> None:
        """Test _is_rl returns 'yes' for RL algorithms."""
        # Arrange
        rl_algorithms = [
            "ppo",
            "qr_dqn",
            "a2c",
            "dqn",
            "epsilon_greedy_bandit",
            "ucb_bandit",
            "q_learning",
        ]

        # Act & Assert
        for alg in rl_algorithms:
            assert _is_rl(alg) == "yes"

    def test_is_rl_with_non_rl_algorithm_returns_no(self) -> None:
        """Test _is_rl returns 'no' for non-RL algorithms."""
        # Arrange
        non_rl_algorithms = ["shortest_path", "k_shortest_path", "dijkstra"]

        # Act & Assert
        for alg in non_rl_algorithms:
            assert _is_rl(alg) == "no"


class TestValidateResourceKeys:
    """Tests for _validate_resource_keys function."""

    def test_validate_resource_keys_with_valid_keys_does_not_raise(self) -> None:
        """Test _validate_resource_keys accepts valid resource keys."""
        # Arrange
        valid_resources = {
            "partition": "gpu",
            "time": "24:00:00",
            "mem": "32G",
            "cpus": "8",
            "gpus": "1",
            "nodes": "1",
        }

        # Act & Assert - should not raise
        _validate_resource_keys(valid_resources)

    def test_validate_resource_keys_with_invalid_key_raises_system_exit(self) -> None:
        """Test _validate_resource_keys raises SystemExit for invalid keys."""
        # Arrange
        invalid_resources = {"invalid_key": "value", "partition": "gpu"}

        # Act & Assert
        with pytest.raises(SystemExit):
            _validate_resource_keys(invalid_resources)

    def test_validate_resource_keys_with_empty_dict_does_not_raise(self) -> None:
        """Test _validate_resource_keys accepts empty resource dict."""
        # Arrange
        empty_resources: dict[str, Any] = {}

        # Act & Assert - should not raise
        _validate_resource_keys(empty_resources)


class TestValidateKeys:
    """Tests for _validate_keys function."""

    @patch("fusion.unity.make_manifest._PARAM_TYPES", {"network": str})
    def test_validate_keys_with_valid_params_does_not_raise(self) -> None:
        """Test _validate_keys accepts recognized parameter keys."""
        # Arrange
        mapping = {"network": "test_network"}

        # Act & Assert - should not raise
        _validate_keys(mapping, "test_context")

    @patch("fusion.unity.make_manifest._PARAM_TYPES", {"network": str})
    def test_validate_keys_with_resource_keys_does_not_raise(self) -> None:
        """Test _validate_keys accepts resource keys."""
        # Arrange
        mapping = {"partition": "gpu", "mem": "32G"}

        # Act & Assert - should not raise
        _validate_keys(mapping, "test_context")

    @patch("fusion.unity.make_manifest._PARAM_TYPES", {"network": str})
    def test_validate_keys_with_invalid_key_raises_system_exit(self) -> None:
        """Test _validate_keys raises SystemExit for unrecognized keys."""
        # Arrange
        mapping = {"invalid_param": "value"}

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            _validate_keys(mapping, "test_context")

        assert "Unknown parameter" in str(exc_info.value)


class TestToList:
    """Tests for _to_list function."""

    def test_to_list_with_single_value_returns_list(self) -> None:
        """Test _to_list wraps single value in list."""
        # Arrange
        value = "single_value"

        # Act
        result = _to_list(value, ctx="test")

        # Assert
        assert result == ["single_value"]

    def test_to_list_with_list_in_grid_context_returns_list(self) -> None:
        """Test _to_list returns list as-is in grid context."""
        # Arrange
        input_list = ["item1", "item2", "item3"]

        # Act
        result = _to_list(input_list, ctx="grid")

        # Assert
        assert result == input_list

    def test_to_list_with_single_item_list_in_common_context_returns_list(
        self,
    ) -> None:
        """Test _to_list accepts single-item list in common context."""
        # Arrange
        input_list = ["single_value"]

        # Act
        result = _to_list(input_list, ctx="common")

        # Assert
        assert result == input_list

    def test_to_list_with_multiple_values_in_common_context_raises_system_exit(
        self,
    ) -> None:
        """Test _to_list raises SystemExit for multiple values in common."""
        # Arrange
        input_list = ["value1", "value2"]

        # Act & Assert
        with pytest.raises(SystemExit):
            _to_list(input_list, ctx="common")


class TestFetch:
    """Tests for _fetch function."""

    def test_fetch_with_key_in_grid_returns_grid_value(self) -> None:
        """Test _fetch retrieves value from grid section."""
        # Arrange
        grid = {"path_algorithm": ["ppo", "dqn"]}
        common = {"erlang_start": 100}

        # Act
        result = _fetch(grid, common, "path_algorithm")

        # Assert
        assert result == ["ppo", "dqn"]

    def test_fetch_with_key_in_common_returns_common_value(self) -> None:
        """Test _fetch retrieves value from common section."""
        # Arrange
        grid = {"path_algorithm": ["ppo", "dqn"]}
        common = {"erlang_start": 100}

        # Act
        result = _fetch(grid, common, "erlang_start")

        # Assert
        assert result == [100]

    def test_fetch_with_missing_key_raises_system_exit(self) -> None:
        """Test _fetch raises SystemExit when key not found."""
        # Arrange
        grid: dict[str, Any] = {}
        common: dict[str, Any] = {}

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            _fetch(grid, common, "missing_key")

        assert "missing required key" in str(exc_info.value)

    def test_fetch_prefers_grid_over_common_when_key_in_both(self) -> None:
        """Test _fetch prioritizes grid value over common."""
        # Arrange
        grid = {"k_paths": [10, 20]}
        common = {"k_paths": 5}

        # Act
        result = _fetch(grid, common, "k_paths")

        # Assert
        assert result == [10, 20]


class TestExpandGrid:
    """Tests for _expand_grid function."""

    @patch("fusion.unity.make_manifest._validate_keys")
    def test_expand_grid_with_basic_config_generates_correct_rows(
        self, _mock_validate: Any
    ) -> None:
        """Test _expand_grid generates correct number of job rows."""
        # Arrange
        grid = {
            "common": {"network": "test_network", "num_requests": 1000},
            "path_algorithm": ["ppo", "dqn"],
            "erlang_start": [100, 200],
            "k_paths": [5],
            "obs_space": ["basic"],
        }

        # Act
        rows, final_rid = _expand_grid(grid, 0)

        # Assert
        assert len(rows) == 4  # 2 alg × 2 traffic × 1 k_paths × 1 obs_space
        assert final_rid == 4

    @patch("fusion.unity.make_manifest._validate_keys")
    def test_expand_grid_generates_rows_with_required_fields(
        self, _mock_validate: Any
    ) -> None:
        """Test _expand_grid includes all required fields in rows."""
        # Arrange
        grid = {
            "common": {"network": "test_network"},
            "path_algorithm": ["ppo"],
            "erlang_start": [100],
            "k_paths": [5],
            "obs_space": ["basic"],
        }

        # Act
        rows, _ = _expand_grid(grid, 0)

        # Assert
        first_row = rows[0]
        assert "run_id" in first_row
        assert "path_algorithm" in first_row
        assert "erlang_start" in first_row
        assert "erlang_stop" in first_row
        assert "k_paths" in first_row
        assert "obs_space" in first_row
        assert "is_rl" in first_row
        assert "network" in first_row

    @patch("fusion.unity.make_manifest._validate_keys")
    def test_expand_grid_calculates_erlang_stop_correctly(
        self, _mock_validate: Any
    ) -> None:
        """Test _expand_grid sets erlang_stop to erlang_start + 50."""
        # Arrange
        grid = {
            "common": {"network": "test_network"},
            "path_algorithm": ["ppo"],
            "erlang_start": [100],
            "k_paths": [5],
            "obs_space": ["basic"],
        }

        # Act
        rows, _ = _expand_grid(grid, 0)

        # Assert
        assert rows[0]["erlang_stop"] == 150

    @patch("fusion.unity.make_manifest._validate_keys")
    def test_expand_grid_with_starting_rid_continues_numbering(
        self, _mock_validate: Any
    ) -> None:
        """Test _expand_grid continues run_id numbering from starting_rid."""
        # Arrange
        grid = {
            "common": {"network": "test_network"},
            "path_algorithm": ["ppo"],
            "erlang_start": [100],
            "k_paths": [5],
            "obs_space": ["basic"],
        }

        # Act
        rows, final_rid = _expand_grid(grid, 10)

        # Assert
        assert rows[0]["run_id"] == "00010"
        assert final_rid == 11

    def test_expand_grid_with_deprecated_repeat_key_raises_system_exit(self) -> None:
        """Test _expand_grid raises SystemExit for deprecated 'repeat' key."""
        # Arrange
        grid = {
            "repeat": 5,
            "path_algorithm": ["ppo"],
            "erlang_start": [100],
            "k_paths": [5],
            "obs_space": ["basic"],
        }

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            _expand_grid(grid, 0)

        assert "deprecated" in str(exc_info.value)


class TestExplicit:
    """Tests for _explicit function."""

    @patch("fusion.unity.make_manifest._validate_keys")
    def test_explicit_with_job_list_generates_rows(self, _mock_validate: Any) -> None:
        """Test _explicit generates rows from explicit job definitions."""
        # Arrange
        jobs = [
            {
                "algorithm": "ppo",
                "traffic": 100,
                "erlang_start": 100,
                "k_paths": 5,
                "network": "test_net",
            },
            {
                "algorithm": "dqn",
                "traffic": 200,
                "erlang_start": 200,
                "k_paths": 10,
                "network": "test_net",
            },
        ]

        # Act
        rows = _explicit(jobs)

        # Assert
        assert len(rows) == 2

    @patch("fusion.unity.make_manifest._validate_keys")
    def test_explicit_sets_correct_algorithm_field(self, _mock_validate: Any) -> None:
        """Test _explicit maps 'algorithm' to 'path_algorithm'."""
        # Arrange
        jobs = [
            {
                "algorithm": "ppo",
                "traffic": 100,
                "erlang_start": 100,
                "k_paths": 5,
            }
        ]

        # Act
        rows = _explicit(jobs)

        # Assert
        assert rows[0]["path_algorithm"] == "ppo"

    @patch("fusion.unity.make_manifest._validate_keys")
    def test_explicit_uses_erlang_start_for_traffic(self, _mock_validate: Any) -> None:
        """Test _explicit uses traffic value for erlang_start."""
        # Arrange
        jobs = [
            {
                "algorithm": "ppo",
                "traffic": 100,
                "erlang_start": 100,
                "k_paths": 5,
            }
        ]

        # Act
        rows = _explicit(jobs)

        # Assert
        assert rows[0]["erlang_start"] == 100

    @patch("fusion.unity.make_manifest._validate_keys")
    def test_explicit_calculates_default_erlang_stop(self, _mock_validate: Any) -> None:
        """Test _explicit calculates erlang_stop when not provided."""
        # Arrange
        jobs = [
            {
                "algorithm": "ppo",
                "traffic": 100,
                "erlang_start": 100,
                "k_paths": 5,
            }
        ]

        # Act
        rows = _explicit(jobs)

        # Assert
        assert rows[0]["erlang_stop"] == 150


class TestWriteCsv:
    """Tests for _write_csv function."""

    def test_write_csv_creates_file_with_correct_content(self, tmp_path: Path) -> None:
        """Test _write_csv creates CSV file with correct data."""
        # Arrange
        rows = [
            {"run_id": "00001", "algorithm": "ppo", "traffic": 100},
            {"run_id": "00002", "algorithm": "dqn", "traffic": 200},
        ]
        csv_path = tmp_path / "test_manifest.csv"

        # Act
        _write_csv(csv_path, rows)

        # Assert
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 2
        assert read_rows[0]["run_id"] == "00001"
        assert read_rows[0]["algorithm"] == "ppo"

    def test_write_csv_handles_missing_fields_with_empty_string(
        self, tmp_path: Path
    ) -> None:
        """Test _write_csv fills missing fields with empty strings."""
        # Arrange
        rows = [
            {"run_id": "00001", "algorithm": "ppo", "traffic": 100},
            {
                "run_id": "00002",
                "algorithm": "dqn",
                "traffic": 200,
                "extra_param": "value",
            },
        ]
        csv_path = tmp_path / "test_manifest.csv"

        # Act
        _write_csv(csv_path, rows)

        # Assert
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert read_rows[0]["extra_param"] == ""
        assert read_rows[1]["extra_param"] == "value"

    def test_write_csv_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test _write_csv creates parent directories if they don't exist."""
        # Arrange
        rows = [{"run_id": "00001", "algorithm": "ppo"}]
        csv_path = tmp_path / "subdir" / "nested" / "test_manifest.csv"

        # Act
        _write_csv(csv_path, rows)

        # Assert
        assert csv_path.exists()
        assert csv_path.parent.exists()


class TestReadSpec:
    """Tests for _read_spec function."""

    def test_read_spec_with_json_file_returns_dict(self, tmp_path: Path) -> None:
        """Test _read_spec correctly parses JSON specification files."""
        # Arrange
        spec_data = {"grid": {"path_algorithm": ["ppo"]}}
        spec_path = tmp_path / "test_spec.json"
        spec_path.write_text('{"grid": {"path_algorithm": ["ppo"]}}', encoding="utf-8")

        # Act
        result = _read_spec(spec_path)

        # Assert
        assert result == spec_data

    @patch("fusion.unity.make_manifest.yaml")
    def test_read_spec_with_yaml_file_returns_dict(
        self, mock_yaml: Any, tmp_path: Path
    ) -> None:
        """Test _read_spec correctly parses YAML specification files."""
        # Arrange
        spec_data = {"grid": {"path_algorithm": ["ppo"]}}
        spec_path = tmp_path / "test_spec.yml"
        spec_path.write_text("grid:\n  path_algorithm:\n    - ppo", encoding="utf-8")
        mock_yaml.safe_load.return_value = spec_data

        # Act
        result = _read_spec(spec_path)

        # Assert
        assert result == spec_data


class TestResolveSpecPath:
    """Tests for _resolve_spec_path function."""

    def test_resolve_spec_path_with_existing_absolute_path_returns_path(
        self, tmp_path: Path
    ) -> None:
        """Test _resolve_spec_path returns existing absolute path."""
        # Arrange
        spec_file = tmp_path / "test_spec.json"
        spec_file.write_text("{}", encoding="utf-8")

        # Act
        result = _resolve_spec_path(str(spec_file))

        # Assert
        assert result == spec_file

    def test_resolve_spec_path_with_nonexistent_file_raises_system_exit(
        self,
    ) -> None:
        """Test _resolve_spec_path raises SystemExit for missing file."""
        # Arrange
        nonexistent_file = "nonexistent_file_12345.json"

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            _resolve_spec_path(nonexistent_file)

        assert "not found" in str(exc_info.value)
