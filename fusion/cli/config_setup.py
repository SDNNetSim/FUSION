"""
Configuration setup module for the FUSION simulator CLI.
"""

import os
import re
from configparser import ConfigParser
from pathlib import Path
from typing import Any

from fusion.configs.constants import (CONFIG_DIR_PATH, DEFAULT_CONFIG_PATH,
                                      DEFAULT_THREAD_NAME, REQUIRED_SECTION,
                                      THREAD_SECTION_PATTERN)
from fusion.configs.errors import (ConfigError, ConfigFileNotFoundError,
                                   ConfigParseError, ConfigTypeConversionError,
                                   MissingRequiredOptionError)
from fusion.configs.schema import (OPTIONAL_OPTIONS_DICT,
                                   SIM_REQUIRED_OPTIONS_DICT)
from fusion.utils.config import (apply_cli_override,
                                 convert_dict_params_if_needed,
                                 safe_type_convert)
from fusion.utils.os import create_directory


def normalize_config_path(config_path: str) -> str:
    """
    Normalize the config file path.

    :param config_path: Path to config file (relative or absolute)
    :type config_path: str
    :return: Absolute path to config file
    :rtype: str
    :raises ConfigFileNotFoundError: If config file does not exist
    """
    config_path = os.path.expanduser(config_path)
    if not os.path.isabs(config_path):
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / config_path

    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise ConfigFileNotFoundError(f"Could not find config file at: {config_path}")

    return config_path


def setup_config_from_cli(args: Any) -> dict[str, Any]:
    """
    Setup the config from command line input.

    :param args: Parsed command line arguments
    :type args: Any
    :return: Configuration dictionary
    :rtype: Dict[str, Any]
    """
    args_dict = vars(args)
    config_path = args_dict.get("config_path")

    try:
        config_data = load_config(config_path, args_dict)
        return config_data
    except (
        ConfigFileNotFoundError,
        ConfigParseError,
        MissingRequiredOptionError,
        ConfigTypeConversionError,
    ) as e:
        print(f"[ERROR] Configuration error: {e}")
        return {}
    except (OSError, ValueError, TypeError) as e:
        print(f"[ERROR] Unexpected error loading config: {e}")
        return {}


def _process_required_options(
    config: ConfigParser,
    config_dict: dict[str, Any],
    required_dict: dict[str, dict[str, Any]],
    optional_dict: dict[str, dict[str, Any]],
    args_dict: dict[str, Any],
) -> None:
    for category, options_dict in required_dict.items():
        for option, type_obj in options_dict.items():
            if not config.has_option(category, option):
                raise MissingRequiredOptionError(
                    f"Missing required option '{option}' in section [{category}]"
                )

            config_value = config[category][option]
            # Convert the config value to the appropriate type
            try:
                config_dict[DEFAULT_THREAD_NAME][option] = safe_type_convert(
                    config_value, type_obj, option
                )
                type_converter = type_obj
            except ConfigTypeConversionError:
                # Try fallback to optional_dict if available
                if category in optional_dict and option in optional_dict[category]:
                    type_converter = optional_dict[category][option]
                    config_dict[DEFAULT_THREAD_NAME][option] = safe_type_convert(
                        config_value, type_converter, option
                    )
                else:
                    raise

            # Handle dictionary parameters
            config_dict[DEFAULT_THREAD_NAME][option] = convert_dict_params_if_needed(
                config_dict[DEFAULT_THREAD_NAME][option], option
            )

            # Apply CLI override if provided
            cli_value = args_dict.get(option)
            final_value = apply_cli_override(
                config_dict[DEFAULT_THREAD_NAME][option], cli_value, type_converter
            )
            config_dict[DEFAULT_THREAD_NAME][option] = final_value


def _process_optional_options(
    config: ConfigParser,
    config_dict: dict[str, Any],
    optional_dict: dict[str, dict[str, Any]],
    args_dict: dict[str, Any],
) -> None:
    for category, options_dict in optional_dict.items():
        for option, type_obj in options_dict.items():
            if option not in config[category]:
                config_dict[DEFAULT_THREAD_NAME][option] = None
            else:
                try:
                    config_value = config[category][option]
                    converted_value = safe_type_convert(config_value, type_obj, option)

                    converted_value = convert_dict_params_if_needed(
                        converted_value, option
                    )

                    # Apply CLI override
                    cli_value = args_dict.get(option)
                    config_dict[DEFAULT_THREAD_NAME][option] = apply_cli_override(
                        converted_value, cli_value, type_obj
                    )
                except ConfigTypeConversionError:
                    # Skip options that can't be converted - they're optional
                    continue


def _validate_config_structure(config: ConfigParser) -> None:
    if not config.has_section(REQUIRED_SECTION):
        create_directory(CONFIG_DIR_PATH)
        raise ConfigParseError(
            f"Missing '{REQUIRED_SECTION}' section in config file. "
            "Ensure config.ini exists in ini/run_ini/."
        )


def _read_config_file(config_path: str) -> ConfigParser:
    config = ConfigParser()
    try:
        config.read(config_path)
    except Exception as e:
        raise ConfigParseError(f"Failed to parse config file {config_path}: {e}") from e
    return config


def _resolve_config_path(config_path: str | None) -> str:
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = normalize_config_path(config_path)

    if not os.path.exists(config_path):
        raise ConfigFileNotFoundError(f"Could not find config file at: {config_path}")

    return config_path


def load_config(
    config_path: str | None, args_dict: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Load an existing config from a config file.

    This function handles the complete configuration loading process:
    1. Resolves and validates the config file path
    2. Reads and parses the INI configuration file
    3. Validates the configuration structure and required sections
    4. Processes both required and optional configuration options
    5. Applies type conversions and CLI argument overrides
    6. Handles multi-threaded configuration sections

    Returns empty dict on error for backward compatibility.
    Use setup_config_from_cli for better error handling.

    :param config_path: Path to configuration file
    :type config_path: Optional[str]
    :param args_dict: Optional CLI arguments dictionary for overrides
    :type args_dict: Optional[Dict[str, Any]]
    :return: Configuration dictionary with thread-based structure
    :rtype: Dict[str, Any]
    """
    if args_dict is None:
        args_dict = {}

    config_dict = {DEFAULT_THREAD_NAME: {}}

    try:
        resolved_path = _resolve_config_path(config_path)
        config = _read_config_file(resolved_path)
        _validate_config_structure(config)
        _process_required_options(
            config,
            config_dict,
            SIM_REQUIRED_OPTIONS_DICT,
            OPTIONAL_OPTIONS_DICT,
            args_dict,
        )
        _process_optional_options(config, config_dict, OPTIONAL_OPTIONS_DICT, args_dict)

        thread_sections = [s for s in config.sections() if s != REQUIRED_SECTION]
        if thread_sections:
            config_dict = _setup_threads(
                config=config,
                config_dict=config_dict,
                section_list=thread_sections,
                types_dict=SIM_REQUIRED_OPTIONS_DICT,
                optional_dict=OPTIONAL_OPTIONS_DICT,
                args_dict=args_dict,
            )

        return config_dict or {}

    except (
        ConfigFileNotFoundError,
        ConfigParseError,
        MissingRequiredOptionError,
        ConfigTypeConversionError,
    ) as error:
        print(f"[ERROR] Configuration error: {error}")
        return {}
    except (OSError, ValueError, TypeError) as error:
        print(f"[ERROR] Unexpected error loading config: {error}")
        return {}


def _setup_threads(
    config: ConfigParser,
    config_dict: dict[str, Any],
    section_list: list[str],
    types_dict: dict[str, dict[str, Any]],
    optional_dict: dict[str, dict[str, Any]],
    args_dict: dict[str, Any],
) -> dict[str, Any]:
    for new_thread in section_list:
        if not re.match(THREAD_SECTION_PATTERN, new_thread):
            continue

        config_dict = _copy_dict_vals(dest_key=new_thread, dictionary=config_dict)

        for key, value in config.items(new_thread):
            category = _find_category(types_dict, key) or _find_category(
                optional_dict, key
            )
            if category is None:
                continue

            type_obj = types_dict.get(category, {}).get(key) or optional_dict.get(
                category, {}
            ).get(key)
            if type_obj is None:
                continue

            try:
                converted_value = safe_type_convert(value, type_obj, key)

                # Apply CLI override
                cli_value = args_dict.get(key)
                config_dict[new_thread][key] = apply_cli_override(
                    converted_value, cli_value, type_obj
                )
            except ConfigTypeConversionError:
                # Skip options that can't be converted in threads
                continue

    return config_dict


def _copy_dict_vals(dest_key: str, dictionary: dict[str, Any]) -> dict[str, Any]:
    dictionary[dest_key] = dict(dictionary[DEFAULT_THREAD_NAME].items())
    return dictionary


def _find_category(
    category_dict: dict[str, dict[str, Any]], target_key: str
) -> str | None:
    for category, subdict in category_dict.items():
        if target_key in subdict:
            return category
    return None


def load_and_validate_config(args: Any) -> dict[str, Any]:
    """
    Load and validate configuration from CLI arguments.

    :param args: Parsed command line arguments
    :type args: Any
    :return: Validated configuration dictionary
    :rtype: Dict[str, Any]
    """
    config_dict = load_config(args.config_path, vars(args))
    return config_dict


class ConfigManager:
    """
    Centralized configuration management for FUSION simulator.

    Provides a unified interface for accessing configuration from both
    INI files and command-line arguments, with proper validation and
    error handling. Supports multi-threaded configuration sections.
    """

    def __init__(self, config_dict: dict[str, Any], args: Any) -> None:
        """
        Initialize ConfigManager with configuration dictionary and arguments.

        :param config_dict: Parsed configuration dictionary
        :type config_dict: Dict[str, Any]
        :param args: Command line arguments
        :type args: Any
        """
        self._config = config_dict
        self._args = args
        self._validate_config()

    def _validate_config(self) -> None:
        # Allow empty config for backward compatibility
        if self._config and DEFAULT_THREAD_NAME not in self._config:
            # Only validate if config is non-empty
            pass  # Config might have other threads, which is valid

    @classmethod
    def from_args(cls, args: Any) -> "ConfigManager":
        """
        Load arguments from command line input.

        :param args: Parsed command line arguments
        :type args: Any
        :return: ConfigManager instance
        :rtype: ConfigManager
        :raises ConfigError: If configuration loading fails
        """
        try:
            config_dict = load_config(args.config_path, vars(args))
            return cls(config_dict, args)
        except ConfigError:
            # Re-raise config errors
            raise
        except Exception as e:
            raise ConfigError(f"Failed to create ConfigManager: {e}") from e

    @classmethod
    def from_file(
        cls, config_path: str, args_dict: dict[str, Any] | None = None
    ) -> "ConfigManager":
        """
        Create ConfigManager from config file path.

        :param config_path: Path to configuration file
        :type config_path: str
        :param args_dict: Optional dictionary of arguments to override config
        :type args_dict: Optional[Dict[str, Any]]
        :return: ConfigManager instance
        :rtype: ConfigManager
        """
        config_dict = load_config(config_path, args_dict)
        # Create a simple namespace object for args if none provided
        args = type("Args", (), args_dict or {})()
        return cls(config_dict, args)

    def as_dict(self) -> dict[str, Any]:
        """
        Get config as dict.

        :return: Configuration dictionary
        :rtype: Dict[str, Any]
        """
        return self._config

    def get(self, thread: str = DEFAULT_THREAD_NAME) -> dict[str, Any]:
        """
        Return a single config thread.

        :param thread: Thread identifier, defaults to 's1'
        :type thread: str
        :return: Configuration for specified thread
        :rtype: Dict[str, Any]
        """
        return self._config.get(thread, {})

    def get_value(
        self, key: str, thread: str = DEFAULT_THREAD_NAME, default: Any = None
    ) -> Any:
        """
        Get a specific configuration value.

        :param key: Configuration key
        :type key: str
        :param thread: Thread identifier, defaults to 's1'
        :type thread: str
        :param default: Default value if key not found, defaults to None
        :type default: Any
        :return: Configuration value or default
        :rtype: Any
        """
        thread_config = self._config.get(thread, {})
        return thread_config.get(key, default)

    def has_thread(self, thread: str) -> bool:
        """
        Check if a thread exists in configuration.

        :param thread: Thread identifier
        :type thread: str
        :return: True if thread exists, False otherwise
        :rtype: bool
        """
        return thread in self._config

    def get_threads(self) -> list[str]:
        """
        Get list of all configured threads.

        :return: List of thread identifiers
        :rtype: List[str]
        """
        return list(self._config.keys())

    def get_args(self) -> Any:
        """
        Get args.

        :return: Command line arguments
        :rtype: Any
        """
        return self._args


if __name__ == "__main__":
    dummy_args = {"run_id": "debug_test"}
    print(load_config("ini/run_ini/config.ini", dummy_args))
