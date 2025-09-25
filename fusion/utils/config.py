"""
Configuration utility functions for the FUSION simulator.

This module provides utilities for configuration management including:
- String to boolean conversion
- Dictionary string parsing
- CLI argument override handling
- Type conversion with error handling
- Dictionary parameter conversion
"""

import ast
from collections.abc import Callable
from typing import Any

from fusion.configs.constants import DICT_PARAM_OPTIONS
from fusion.configs.errors import ConfigTypeConversionError


def str_to_bool(string: str) -> bool:
    """
    Convert string to boolean value.
    
    :param string: Input string to convert
    :type string: str
    :return: Boolean value
    :rtype: bool
    """
    return string.lower() in ["true", "yes", "1"]


def convert_string_to_dict(value: str) -> str | dict[str, Any]:
    """
    Convert string representation of dictionary to actual dictionary.
    
    :param value: String that may contain a dictionary literal
    :type value: str
    :return: Dictionary if conversion successful, original string otherwise
    :rtype: str | dict[str, Any]
    """
    try:
        result = ast.literal_eval(value)
        if isinstance(result, dict):
            return result
        else:
            return value
    except (ValueError, SyntaxError):
        return value  # Keep as string if parsing fails


def apply_cli_override(
    configuration_value: Any, cli_argument_value: Any | None, type_converter: Callable
) -> Any:
    """
    Apply CLI argument override with proper handling of boolean store_true arguments.
    
    :param configuration_value: Value from config file
    :type configuration_value: Any
    :param cli_argument_value: Value from CLI arguments (may be None)
    :type cli_argument_value: Any | None
    :param type_converter: Function to convert string values to appropriate type
    :type type_converter: Callable
    :return: Final value after considering CLI override
    :rtype: Any
    """
    if cli_argument_value is None:
        return (
            type_converter(configuration_value)
            if isinstance(configuration_value, str)
            else configuration_value
        )

    # For boolean store_true arguments, only override if explicitly set to True
    if type_converter is str_to_bool and cli_argument_value is False:
        return (
            type_converter(configuration_value)
            if isinstance(configuration_value, str)
            else configuration_value
        )

    return cli_argument_value


def safe_type_convert(value: str, type_converter: Callable, option_name: str) -> Any:
    """
    Safely convert value with proper error context.
    
    :param value: String value to convert
    :type value: str
    :param type_converter: Function to perform conversion
    :type type_converter: Callable
    :param option_name: Name of option for error reporting
    :type option_name: str
    :return: Converted value
    :rtype: Any
    :raises ConfigTypeConversionError: If conversion fails
    """
    try:
        return type_converter(value)
    except (ValueError, TypeError) as e:
        raise ConfigTypeConversionError(
            f"Failed to convert {option_name}='{value}' "
            f"using {type_converter.__name__}: {e}"
        ) from e


def convert_dict_params_if_needed(value: Any, option: str) -> Any:
    """
    Convert dictionary parameters from string to dict if needed.
    
    :param value: Configuration value
    :type value: Any
    :param option: Option name
    :type option: str
    :return: Converted value if it was a dict param, otherwise original value
    :rtype: Any
    """
    if option in DICT_PARAM_OPTIONS and isinstance(value, str):
        return convert_string_to_dict(value)
    return value
