"""Centralized configuration management for FUSION simulator."""

import os
import json
import configparser
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import yaml
except ImportError:
    yaml = None

from .validate import SchemaValidator
from .cli_to_config import CLIToConfigMapper


@dataclass
class SimulationConfig:
    """Data class for simulation configuration."""
    general: Dict[str, Any]
    topology: Dict[str, Any]
    spectrum: Dict[str, Any]
    snr: Dict[str, Any]
    rl: Dict[str, Any]
    ml: Dict[str, Any]
    file: Dict[str, Any]


class ConfigManager:
    """Centralized configuration manager with schema validation."""

    def __init__(self, config_path: Optional[str] = None, schema_dir: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            schema_dir: Directory containing schema files
        """
        self.config_path = config_path
        self.schema_validator = SchemaValidator(schema_dir) if schema_dir else None
        self._config: Optional[SimulationConfig] = None
        self._raw_config: Dict[str, Any] = {}

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, path: str) -> SimulationConfig:
        """Load and validate configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Validated configuration object
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If configuration file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Determine file type and load accordingly
        if path.endswith('.ini'):
            self._raw_config = self._load_ini(path)
        elif path.endswith('.json'):
            self._raw_config = self._load_json(path)
        elif path.endswith(('.yaml', '.yml')):
            self._raw_config = self._load_yaml(path)
        else:
            raise ValueError(f"Unsupported configuration file format: {path}")

        # Validate against schema if validator is available
        if self.schema_validator:
            self.schema_validator.validate(self._raw_config)

        # Convert to structured configuration
        self._config = self._create_config_object(self._raw_config)
        return self._config

    def _load_ini(self, path: str) -> Dict[str, Any]:
        """Load INI configuration file."""
        config = configparser.ConfigParser()
        config.read(path)

        result = {}
        for section_name in config.sections():
            section = {}
            for key, value in config[section_name].items():
                # Try to parse JSON values (for lists/dicts)
                try:
                    section[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Try to parse as number
                    try:
                        if '.' in value:
                            section[key] = float(value)
                        else:
                            section[key] = int(value)
                    except ValueError:
                        # Try to parse as boolean
                        if value.lower() in ('true', 'false'):
                            section[key] = value.lower() == 'true'
                        elif value.lower() == 'none':
                            section[key] = None
                        else:
                            section[key] = value
            result[section_name] = section

        return result

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if yaml is None:
            raise ImportError("PyYAML is required for YAML configuration files")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_config_object(self, raw_config: Dict[str, Any]) -> SimulationConfig:
        """Create structured configuration object from raw config."""
        return SimulationConfig(
            general=raw_config.get('general_settings', {}),
            topology=raw_config.get('topology_settings', {}),
            spectrum=raw_config.get('spectrum_settings', {}),
            snr=raw_config.get('snr_settings', {}),
            rl=raw_config.get('rl_settings', {}),
            ml=raw_config.get('ml_settings', {}),
            file=raw_config.get('file_settings', {})
        )

    def get_config(self) -> Optional[SimulationConfig]:
        """Get the loaded configuration object."""
        return self._config

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for a specific module.
        
        Args:
            module_name: Name of the module (e.g., 'routing', 'spectrum', 'snr')
            
        Returns:
            Module-specific configuration dictionary
        """
        if not self._config:
            return {}

        module_map = {
            'general': self._config.general,
            'topology': self._config.topology,
            'spectrum': self._config.spectrum,
            'snr': self._config.snr,
            'rl': self._config.rl,
            'ml': self._config.ml,
            'file': self._config.file,
            'routing': {
                **self._config.general,
                **self._config.topology
            },
        }

        return module_map.get(module_name, {})

    def save_config(self, path: str, format_type: str = 'ini') -> None:
        """Save current configuration to file.
        
        Args:
            path: Output file path
            format_type: Output format ('ini', 'json', 'yaml')
        """
        if not self._raw_config:
            raise ValueError("No configuration loaded to save")

        if format_type == 'ini':
            self._save_ini(path)
        elif format_type == 'json':
            self._save_json(path)
        elif format_type == 'yaml':
            self._save_yaml(path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _save_ini(self, path: str) -> None:
        """Save configuration as INI file."""
        config = configparser.ConfigParser()

        for section_name, section_data in self._raw_config.items():
            config[section_name] = {}
            for key, value in section_data.items():
                if isinstance(value, (dict, list)):
                    config[section_name][key] = json.dumps(value)
                else:
                    config[section_name][key] = str(value)

        with open(path, 'w', encoding='utf-8') as f:
            config.write(f)

    def _save_json(self, path: str) -> None:
        """Save configuration as JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._raw_config, f, indent=2)

    def _save_yaml(self, path: str) -> None:
        """Save configuration as YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required for YAML configuration files")
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False)

    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a specific configuration value.
        
        Args:
            section: Configuration section name
            key: Configuration key
            value: New value
        """
        if section not in self._raw_config:
            self._raw_config[section] = {}

        self._raw_config[section][key] = value

        # Recreate config object
        if self._config:
            self._config = self._create_config_object(self._raw_config)

    def merge_cli_args(self, args: Dict[str, Any]) -> None:
        """Merge CLI arguments into configuration.
        
        Args:
            args: Dictionary of CLI arguments
        """
        mapper = CLIToConfigMapper()
        cli_config = mapper.map_args_to_config(args)

        # Merge CLI config into existing config
        for section, values in cli_config.items():
            if section not in self._raw_config:
                self._raw_config[section] = {}
            self._raw_config[section].update(values)

        # Recreate config object
        self._config = self._create_config_object(self._raw_config)
