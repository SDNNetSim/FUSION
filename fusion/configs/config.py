"""Centralized configuration management for FUSION simulator."""

import configparser
import json
import os
from dataclasses import dataclass
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from .cli_to_config import CLIToConfigMapper
from .errors import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigTypeConversionError,
)
from .validate import SchemaValidator


@dataclass
class SimulationConfig:
    """
    Data class for simulation configuration.

    Contains all configuration sections needed for FUSION simulation.
    Each attribute represents a configuration section with its parameters.
    """

    general: dict[str, Any]
    topology: dict[str, Any]
    spectrum: dict[str, Any]
    snr: dict[str, Any]
    rl: dict[str, Any]
    ml: dict[str, Any]
    file: dict[str, Any]


class ConfigManager:
    """
    Centralized configuration manager with schema validation.

    Manages loading, validation, and access to FUSION configuration from
    various file formats (INI, JSON, YAML). Supports schema validation
    and CLI argument merging.
    """

    def __init__(self, config_path: str | None = None, schema_dir: str | None = None):
        """
        Initialize configuration manager.

        :param config_path: Path to configuration file to load on initialization
        :type config_path: Optional[str]
        :param schema_dir: Directory containing schema files for validation
        :type schema_dir: Optional[str]
        """
        self.config_path = config_path
        self.schema_validator = SchemaValidator(schema_dir) if schema_dir else None
        self._config: SimulationConfig | None = None
        self._raw_config: dict[str, Any] = {}

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, path: str) -> SimulationConfig:
        """
        Load and validate configuration from file.

        Automatically detects file format based on extension and loads
        the configuration. Validates against schema if available.

        :param path: Path to configuration file (INI, JSON, or YAML)
        :type path: str
        :return: Validated configuration object
        :rtype: SimulationConfig
        :raises ConfigFileNotFoundError: If configuration file doesn't exist
        :raises ConfigParseError: If file format is unsupported or parsing fails
        :raises ConfigError: If configuration validation fails

        Example:
            >>> manager = ConfigManager()
            >>> config = manager.load_config('config.ini')
            >>> print(config.general['holding_time'])
            10
        """
        if not os.path.exists(path):
            raise ConfigFileNotFoundError(f"Configuration file not found: '{path}'. Please ensure the file exists or provide a valid path.")

        # Load configuration based on file extension
        try:
            if path.endswith(".ini"):
                self._raw_config = self._load_ini(path)
            elif path.endswith(".json"):
                self._raw_config = self._load_json(path)
            elif path.endswith((".yaml", ".yml")):
                self._raw_config = self._load_yaml(path)
            else:
                raise ConfigParseError(f"Unsupported configuration file format: '{path}'. Supported formats: .ini, .json, .yaml, .yml")
        except Exception as e:
            if isinstance(e, (ConfigError, ConfigFileNotFoundError, ConfigParseError)):
                raise
            raise ConfigParseError(f"Failed to parse configuration file: {str(e)}") from e

        # Validate against schema if validator is available
        if self.schema_validator:
            try:
                self.schema_validator.validate(self._raw_config)
            except Exception as e:
                raise ConfigError(f"Configuration validation failed: {str(e)}") from e

        # Convert to structured configuration object
        self._config = self._create_config_object(self._raw_config)
        return self._config

    def _load_ini(self, path: str) -> dict[str, Any]:
        config = configparser.ConfigParser()
        config.read(path)

        result_dict: dict[str, Any] = {}
        for section_name in config.sections():
            section_dict: dict[str, Any] = {}
            for key, value in config[section_name].items():
                # Parse JSON values first (handles lists/dicts/null)
                try:
                    section_dict[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Parse numeric values
                    try:
                        if "." in value:
                            section_dict[key] = float(value)
                        else:
                            section_dict[key] = int(value)
                    except ValueError:
                        # Parse boolean values
                        if value.lower() in ("true", "false"):
                            section_dict[key] = value.lower() == "true"
                        elif value.lower() == "none":
                            section_dict[key] = None
                        else:
                            section_dict[key] = value
            result_dict[section_name] = section_dict

        return result_dict

    def _load_json(self, path: str) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            raise ConfigParseError(f"Invalid JSON format in '{path}': {str(e)}") from e

    def _load_yaml(self, path: str) -> dict[str, Any]:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML configuration files. Install it with: pip install pyyaml")
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except yaml.YAMLError as e:
            raise ConfigParseError(f"Invalid YAML format in '{path}': {str(e)}") from e

    def _create_config_object(self, raw_config: dict[str, Any]) -> SimulationConfig:
        return SimulationConfig(
            general=raw_config.get("general_settings", {}),
            topology=raw_config.get("topology_settings", {}),
            spectrum=raw_config.get("spectrum_settings", {}),
            snr=raw_config.get("snr_settings", {}),
            rl=raw_config.get("rl_settings", {}),
            ml=raw_config.get("ml_settings", {}),
            file=raw_config.get("file_settings", {}),
        )

    def get_config(self) -> SimulationConfig | None:
        """
        Get the loaded configuration object.

        :return: Loaded configuration or None if not loaded
        :rtype: Optional[SimulationConfig]
        """
        return self._config

    def get_module_config(self, module_name: str) -> dict[str, Any]:
        """
        Get configuration for a specific module.

        Returns the configuration section(s) relevant to a specific module.
        Some modules like 'routing' may combine multiple sections.

        :param module_name: Name of the module (e.g., 'routing', 'spectrum', 'snr')
        :type module_name: str
        :return: Module-specific configuration dictionary
        :rtype: Dict[str, Any]

        Example:
            >>> config = manager.get_module_config('routing')
            >>> print(config['k_paths'])  # From general settings
            3
        """
        if not self._config:
            return {}

        module_map_dict: dict[str, dict[str, Any]] = {
            "general": self._config.general,
            "topology": self._config.topology,
            "spectrum": self._config.spectrum,
            "snr": self._config.snr,
            "rl": self._config.rl,
            "ml": self._config.ml,
            "file": self._config.file,
            "routing": {**self._config.general, **self._config.topology},
        }

        return module_map_dict.get(module_name, {})

    def save_config(self, path: str, format_type: str = "ini") -> None:
        """
        Save current configuration to file.

        Saves the current configuration in the specified format.

        :param path: Output file path
        :type path: str
        :param format_type: Output format ('ini', 'json', 'yaml'), defaults to 'ini'
        :type format_type: str
        :raises ValueError: If no configuration is loaded
        :raises ConfigError: If format is unsupported or save fails
        """
        if not self._raw_config:
            raise ValueError("No configuration loaded to save. Please load a configuration first using load_config().")

        try:
            if format_type == "ini":
                self._save_ini(path)
            elif format_type == "json":
                self._save_json(path)
            elif format_type == "yaml":
                self._save_yaml(path)
            else:
                raise ConfigError(f"Unsupported format: '{format_type}'. Supported formats: 'ini', 'json', 'yaml'")
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Failed to save configuration: {str(e)}") from e

    def _save_ini(self, path: str) -> None:
        config = configparser.ConfigParser()

        for section_name, section_data in self._raw_config.items():
            config[section_name] = {}
            for key, value in section_data.items():
                if isinstance(value, (dict, list)):
                    # Complex types saved as JSON strings
                    config[section_name][key] = json.dumps(value)
                else:
                    config[section_name][key] = str(value)

        with open(path, "w", encoding="utf-8") as f:
            config.write(f)

    def _save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._raw_config, f, indent=2)

    def _save_yaml(self, path: str) -> None:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML configuration files. Install it with: pip install pyyaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._raw_config, f, default_flow_style=False)

    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value.

        Updates a single configuration value and recreates the
        configuration object to reflect the change.

        :param section: Configuration section name
        :type section: str
        :param key: Configuration key within the section
        :type key: str
        :param value: New value to set
        :type value: Any

        Example:
            >>> manager.update_config('general_settings', 'holding_time', 20)
        """
        if section not in self._raw_config:
            self._raw_config[section] = {}

        self._raw_config[section][key] = value

        # Recreate config object to reflect changes
        if self._config:
            self._config = self._create_config_object(self._raw_config)

    def merge_cli_args(self, args: dict[str, Any]) -> None:
        """
        Merge CLI arguments into configuration.

        CLI arguments take precedence over existing configuration values.

        :param args: Dictionary of CLI arguments to merge
        :type args: Dict[str, Any]
        :raises ConfigTypeConversionError: If CLI argument mapping fails

        Example:
            >>> cli_args = {'holding_time': 15, 'network': 'nsfnet'}
            >>> manager.merge_cli_args(cli_args)
        """
        try:
            mapper = CLIToConfigMapper()
            cli_config_dict = mapper.map_args_to_config(args)

            # Merge CLI config into existing config
            for section, values in cli_config_dict.items():
                if section not in self._raw_config:
                    self._raw_config[section] = {}
                self._raw_config[section].update(values)

            # Recreate config object with merged values
            self._config = self._create_config_object(self._raw_config)
        except Exception as e:
            raise ConfigTypeConversionError(f"Failed to merge CLI arguments: {str(e)}") from e
