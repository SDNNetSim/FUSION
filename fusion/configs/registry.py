"""Configuration registry and factory for FUSION simulator."""

import glob
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import ConfigManager
from .errors import ConfigError, ConfigFileNotFoundError
from .validate import SchemaValidator


class ConfigRegistry:
    """Registry for managing configuration templates and presets.
    
    This class provides a centralized registry for configuration templates,
    predefined profiles, and utilities for creating custom configurations.
    It supports template loading, validation, and profile-based configuration.
    """

    def __init__(self, templates_dir: Optional[str] = None, schemas_dir: Optional[str] = None):
        """Initialize configuration registry.
        
        :param templates_dir: Directory containing configuration templates,
                             defaults to 'templates' subdirectory
        :type templates_dir: Optional[str]
        :param schemas_dir: Directory containing schema files,
                           defaults to 'schemas' subdirectory
        :type schemas_dir: Optional[str]
        """
        self.templates_dir = templates_dir or os.path.join(os.path.dirname(__file__), 'templates')
        self.schemas_dir = schemas_dir or os.path.join(os.path.dirname(__file__), 'schemas')
        self.validator = SchemaValidator(self.schemas_dir)
        self._templates_dict: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        if not os.path.exists(self.templates_dir):
            return

        # Load all .ini template files from templates directory
        template_pattern = os.path.join(self.templates_dir, '*.ini')
        for template_path in glob.glob(template_pattern):
            template_name = Path(template_path).stem
            self._templates_dict[template_name] = template_path

    def list_templates(self) -> List[str]:
        """Get list of available configuration templates.
        
        :return: List of available template names
        :rtype: List[str]
        """
        return list(self._templates_dict.keys())

    def get_template_path(self, template_name: str) -> Optional[str]:
        """Get path to a specific template.
        
        :param template_name: Name of the template to find
        :type template_name: str
        :return: Path to template file, or None if not found
        :rtype: Optional[str]
        """
        return self._templates_dict.get(template_name)

    def load_template(self, template_name: str) -> ConfigManager:
        """Load a configuration template.
        
        :param template_name: Name of the template to load
        :type template_name: str
        :return: ConfigManager instance with loaded template
        :rtype: ConfigManager
        :raises ConfigError: If template is not found
        
        Example:
            >>> registry = ConfigRegistry()
            >>> config = registry.load_template('default')
        """
        template_path = self.get_template_path(template_name)
        if not template_path:
            raise ConfigError(
                f"Template '{template_name}' not found. "
                f"Available templates: {', '.join(self.list_templates())}"
            )

        return ConfigManager(template_path, self.schemas_dir)

    def create_custom_config(self, base_template: str = 'default',
                             overrides: Optional[Dict[str, Any]] = None) -> ConfigManager:
        """Create a custom configuration based on a template with overrides.
        
        :param base_template: Name of base template to use, defaults to 'default'
        :type base_template: str
        :param overrides: Dictionary of configuration overrides in
                         'section.key' or 'key' format
        :type overrides: Optional[Dict[str, Any]]
        :return: ConfigManager instance with custom configuration
        :rtype: ConfigManager
        :raises ConfigError: If base template is not found
        
        Example:
            >>> registry = ConfigRegistry()
            >>> config = registry.create_custom_config(
            ...     'default',
            ...     {'general_settings.max_iters': 5}
            ... )
        """
        # Load base template
        config_manager = self.load_template(base_template)

        # Apply overrides if provided
        if overrides:
            for section_key, value in overrides.items():
                if '.' in section_key:
                    section, key = section_key.split('.', 1)
                    config_manager.update_config(section, key, value)
                else:
                    # Default to general_settings for unqualified keys
                    config_manager.update_config('general_settings', section_key, value)

        return config_manager

    def validate_config(self, config_path: str) -> List[str]:
        """Validate a configuration file.
        
        :param config_path: Path to configuration file to validate
        :type config_path: str
        :return: List of validation errors (empty if valid)
        :rtype: List[str]
        """
        try:
            ConfigManager(config_path, self.schemas_dir)
            return []
        except (ConfigError, ConfigFileNotFoundError, ValueError, FileNotFoundError) as e:
            return [str(e)]

    def get_config_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined configuration profiles for different use cases.
        
        :return: Dictionary mapping profile names to their configurations
        :rtype: Dict[str, Dict[str, Any]]
        """
        return {
            'quick_test': {
                'description': 'Fast configuration for testing',
                'template': 'minimal',
                'overrides': {
                    'max_iters': 1,
                    'num_requests': 50,
                    'erlang_stop': 300
                }
            },
            'development': {
                'description': 'Development configuration with detailed logging',
                'template': 'default',
                'overrides': {
                    'print_step': 5,
                    'save_snapshots': True,
                    'snapshot_step': 10
                }
            },
            'production': {
                'description': 'Production configuration with optimized settings',
                'template': 'default',
                'overrides': {
                    'max_iters': 10,
                    'thread_erlangs': True,
                    'save_snapshots': False
                }
            },
            'rl_experiment': {
                'description': 'Reinforcement learning experiment setup',
                'template': 'rl_training',
                'overrides': {
                    'n_trials': 50,
                    'optimize_hyperparameters': True
                }
            },
            'benchmark': {
                'description': 'Benchmarking configuration',
                'template': 'default',
                'overrides': {
                    'max_iters': 20,
                    'num_requests': 2000,
                    'thread_erlangs': True,
                    'save_start_end_slots': True
                }
            }
        }

    def create_profile_config(self, profile_name: str,
                              additional_overrides: Optional[Dict[str, Any]] = None) -> ConfigManager:
        """Create configuration based on a predefined profile.
        
        :param profile_name: Name of the profile to use
        :type profile_name: str
        :param additional_overrides: Additional overrides to apply on top of profile
        :type additional_overrides: Optional[Dict[str, Any]]
        :return: ConfigManager instance with profile configuration
        :rtype: ConfigManager
        :raises ConfigError: If profile is not found
        
        Example:
            >>> registry = ConfigRegistry()
            >>> config = registry.create_profile_config(
            ...     'development',
            ...     {'num_requests': 100}
            ... )
        """
        profiles_dict = self.get_config_profiles()
        if profile_name not in profiles_dict:
            raise ConfigError(
                f"Profile '{profile_name}' not found. "
                f"Available profiles: {', '.join(profiles_dict.keys())}"
            )

        profile_dict = profiles_dict[profile_name]
        overrides_dict = profile_dict['overrides'].copy()

        # Merge additional overrides
        if additional_overrides:
            overrides_dict.update(additional_overrides)

        return self.create_custom_config(profile_dict['template'], overrides_dict)

    def export_config_template(self, config_manager: ConfigManager,
                               template_name: str, description: str = "") -> str:
        """Export a configuration as a new template.
        
        :param config_manager: ConfigManager instance to export
        :type config_manager: ConfigManager
        :param template_name: Name for the new template file
        :type template_name: str
        :param description: Optional description for the template
        :type description: str
        :return: Path to the exported template file
        :rtype: str
        """
        template_path = os.path.join(self.templates_dir, f"{template_name}.ini")

        # Description injection pending - see TODO.md for implementation details
        if description:
            # Description will be added when comment support is implemented
            _ = description  # Suppress unused variable warning

        config_manager.save_config(template_path, 'ini')

        # Refresh templates to include the newly exported template
        self._load_templates()

        return template_path
