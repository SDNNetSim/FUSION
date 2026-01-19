"""Unit tests for fusion.utils.config module."""

from typing import Any
from unittest.mock import patch

import pytest

from fusion.configs.errors import ConfigTypeConversionError
from fusion.utils.config import (
    apply_cli_override,
    convert_dict_params_if_needed,
    convert_string_to_dict,
    safe_type_convert,
    str_to_bool,
)


class TestStrToBool:
    """Tests for str_to_bool function."""

    @pytest.mark.parametrize(
        "input_string,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("Yes", True),
            ("YES", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("no", False),
            ("No", False),
            ("NO", False),
            ("0", False),
            ("", False),
            ("random", False),
        ],
    )
    def test_str_to_bool_with_various_inputs_returns_expected(self, input_string: str, expected: bool) -> None:
        """Test str_to_bool with various string inputs."""
        # Act
        result = str_to_bool(input_string)

        # Assert
        assert result == expected


class TestConvertStringToDict:
    """Tests for convert_string_to_dict function."""

    def test_convert_string_to_dict_with_valid_dict_string_returns_dict(self) -> None:
        """Test conversion of valid dictionary string."""
        # Arrange
        dict_string = "{'key1': 'value1', 'key2': 2}"

        # Act
        result = convert_string_to_dict(dict_string)

        # Assert
        assert isinstance(result, dict)
        assert result == {"key1": "value1", "key2": 2}

    def test_convert_string_to_dict_with_empty_dict_returns_dict(self) -> None:
        """Test conversion of empty dictionary string."""
        # Arrange
        dict_string = "{}"

        # Act
        result = convert_string_to_dict(dict_string)

        # Assert
        assert isinstance(result, dict)
        assert result == {}

    def test_convert_string_to_dict_with_invalid_string_returns_original(self) -> None:
        """Test that invalid string returns original value."""
        # Arrange
        invalid_string = "not a dict"

        # Act
        result = convert_string_to_dict(invalid_string)

        # Assert
        assert result == invalid_string

    def test_convert_string_to_dict_with_list_string_returns_original(self) -> None:
        """Test that list string returns original value."""
        # Arrange
        list_string = "[1, 2, 3]"

        # Act
        result = convert_string_to_dict(list_string)

        # Assert
        assert result == list_string

    def test_convert_string_to_dict_with_syntax_error_returns_original(self) -> None:
        """Test that syntax error returns original string."""
        # Arrange
        malformed_string = "{'key': 'value'"

        # Act
        result = convert_string_to_dict(malformed_string)

        # Assert
        assert result == malformed_string

    def test_convert_string_to_dict_with_nested_dict_returns_dict(self) -> None:
        """Test conversion of nested dictionary string."""
        # Arrange
        nested_dict_string = "{'outer': {'inner': 'value'}}"

        # Act
        result = convert_string_to_dict(nested_dict_string)

        # Assert
        assert isinstance(result, dict)
        assert result == {"outer": {"inner": "value"}}


class TestApplyCliOverride:
    """Tests for apply_cli_override function."""

    def test_apply_cli_override_with_none_cli_value_returns_config_value(self) -> None:
        """Test that None CLI value returns configuration value."""
        # Arrange
        config_value = "original"
        cli_value = None

        # Act
        result = apply_cli_override(config_value, cli_value, str)

        # Assert
        assert result == "original"

    def test_apply_cli_override_with_none_cli_and_string_config_converts(self) -> None:
        """Test type conversion when CLI value is None."""
        # Arrange
        config_value = "42"
        cli_value = None

        # Act
        result = apply_cli_override(config_value, cli_value, int)

        # Assert
        assert result == 42

    def test_apply_cli_override_with_cli_value_returns_cli_value(self) -> None:
        """Test that CLI value overrides configuration value."""
        # Arrange
        config_value = "original"
        cli_value = "override"

        # Act
        result = apply_cli_override(config_value, cli_value, str)

        # Assert
        assert result == "override"

    def test_apply_cli_override_with_bool_and_false_cli_returns_config(self) -> None:
        """Test boolean store_true behavior with False CLI value."""
        # Arrange
        config_value = "true"
        cli_value = False

        # Act
        result = apply_cli_override(config_value, cli_value, str_to_bool)

        # Assert
        assert result is True

    def test_apply_cli_override_with_bool_and_true_cli_returns_cli(self) -> None:
        """Test boolean store_true behavior with True CLI value."""
        # Arrange
        config_value = "false"
        cli_value = True

        # Act
        result = apply_cli_override(config_value, cli_value, str_to_bool)

        # Assert
        assert result is True

    def test_apply_cli_override_with_non_string_config_returns_as_is(self) -> None:
        """Test that non-string config value is returned without conversion."""
        # Arrange
        config_value = 42
        cli_value = None

        # Act
        result = apply_cli_override(config_value, cli_value, str)

        # Assert
        assert result == 42


class TestSafeTypeConvert:
    """Tests for safe_type_convert function."""

    def test_safe_type_convert_with_valid_int_conversion_succeeds(self) -> None:
        """Test successful integer conversion."""
        # Arrange
        value = "42"
        option_name = "test_option"

        # Act
        result = safe_type_convert(value, int, option_name)

        # Assert
        assert result == 42

    def test_safe_type_convert_with_valid_float_conversion_succeeds(self) -> None:
        """Test successful float conversion."""
        # Arrange
        value = "3.14"
        option_name = "test_option"

        # Act
        result = safe_type_convert(value, float, option_name)

        # Assert
        assert result == 3.14

    def test_safe_type_convert_with_invalid_conversion_raises_error(self) -> None:
        """Test that invalid conversion raises ConfigTypeConversionError."""
        # Arrange
        value = "not_a_number"
        option_name = "test_option"

        # Act & Assert
        with pytest.raises(ConfigTypeConversionError) as exc_info:
            safe_type_convert(value, int, option_name)

        assert "test_option" in str(exc_info.value)
        assert "not_a_number" in str(exc_info.value)

    def test_safe_type_convert_with_str_to_bool_conversion_succeeds(self) -> None:
        """Test successful boolean conversion."""
        # Arrange
        value = "true"
        option_name = "bool_option"

        # Act
        result = safe_type_convert(value, str_to_bool, option_name)

        # Assert
        assert result is True

    def test_safe_type_convert_error_includes_converter_name(self) -> None:
        """Test that error message includes converter function name."""
        # Arrange
        value = "invalid"
        option_name = "test_option"

        # Act & Assert
        with pytest.raises(ConfigTypeConversionError) as exc_info:
            safe_type_convert(value, int, option_name)

        assert "int" in str(exc_info.value)


class TestConvertDictParamsIfNeeded:
    """Tests for convert_dict_params_if_needed function."""

    @patch("fusion.utils.config.DICT_PARAM_OPTIONS", ["dict_option"])
    def test_convert_dict_params_with_dict_option_converts_string(self) -> None:
        """Test conversion of dict parameter from string."""
        # Arrange
        value = "{'key': 'value'}"
        option = "dict_option"

        # Act
        result = convert_dict_params_if_needed(value, option)

        # Assert
        assert isinstance(result, dict)
        assert result == {"key": "value"}

    @patch("fusion.utils.config.DICT_PARAM_OPTIONS", ["dict_option"])
    def test_convert_dict_params_with_non_dict_option_returns_original(self) -> None:
        """Test that non-dict option returns original value."""
        # Arrange
        value = "some_string"
        option = "regular_option"

        # Act
        result = convert_dict_params_if_needed(value, option)

        # Assert
        assert result == "some_string"

    @patch("fusion.utils.config.DICT_PARAM_OPTIONS", ["dict_option"])
    def test_convert_dict_params_with_dict_value_returns_original(self) -> None:
        """Test that already-dict value is returned as-is."""
        # Arrange
        value: dict[str, Any] = {"key": "value"}
        option = "dict_option"

        # Act
        result = convert_dict_params_if_needed(value, option)

        # Assert
        assert result == {"key": "value"}

    @patch("fusion.utils.config.DICT_PARAM_OPTIONS", ["dict_option"])
    def test_convert_dict_params_with_invalid_dict_string_returns_original(
        self,
    ) -> None:
        """Test that invalid dict string is returned as-is."""
        # Arrange
        value = "not a dict"
        option = "dict_option"

        # Act
        result = convert_dict_params_if_needed(value, option)

        # Assert
        assert result == "not a dict"
