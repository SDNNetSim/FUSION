"""Configuration utility functions for the FUSION simulator."""

import ast
from collections.abc import Callable
from typing import Any

from fusion.configs.constants import DICT_PARAM_OPTIONS_LIST
from fusion.configs.errors import ConfigTypeConversionError


def str_to_bool(string: str) -> bool:
    """Convert string to boolean value.

    Args:
        string: Input string to convert

    Returns:
        Boolean value
    """
    return string.lower() in ["true", "yes", "1"]


def convert_string_to_dict(value: str) -> str | dict[str, Any]:
    """Convert string representation of dictionary to actual dictionary.

    Args:
        value: String that may contain a dictionary literal

    Returns:
        Dictionary if conversion successful, original string otherwise
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value  # Keep as string if parsing fails


def apply_cli_override(
    config_value: Any, cli_value: Any | None, type_converter: Callable
) -> Any:
    """Apply CLI argument override with proper handling of boolean store_true arguments.

    Args:
        config_value: Value from config file
        cli_value: Value from CLI arguments (may be None)
        type_converter: Function to convert string values to appropriate type

    Returns:
        Final value after considering CLI override
    """
    if cli_value is None:
        return (
            type_converter(config_value)
            if isinstance(config_value, str)
            else config_value
        )

    # For boolean store_true arguments, only override if explicitly set to True
    if type_converter is str_to_bool and cli_value is False:
        return (
            type_converter(config_value)
            if isinstance(config_value, str)
            else config_value
        )

    return cli_value


def safe_type_convert(value: str, type_converter: Callable, option_name: str) -> Any:
    """Safely convert value with proper error context.

    Args:
        value: String value to convert
        type_converter: Function to perform conversion
        option_name: Name of option for error reporting

    Returns:
        Converted value

    Raises:
        ConfigTypeConversionError: If conversion fails
    """
    try:
        return type_converter(value)
    except (ValueError, TypeError) as e:
        raise ConfigTypeConversionError(
            f"Failed to convert {option_name}='{value}' using {type_converter.__name__}: {e}"
        ) from e


def convert_dict_params_if_needed(value: Any, option: str) -> Any:
    """Convert dictionary parameters from string to dict if needed.

    Args:
        value: Configuration value
        option: Option name

    Returns:
        Converted value if it was a dict param, otherwise original value
    """
    if option in DICT_PARAM_OPTIONS_LIST and isinstance(value, str):
        return convert_string_to_dict(value)
    return value
