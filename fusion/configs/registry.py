"""Configuration registry and factory for FUSION simulator."""

import os
import glob
from typing import Dict, Any, List, Optional
from pathlib import Path

from .config import ConfigManager
from .validate import SchemaValidator


class ConfigRegistry:
    """Registry for managing configuration templates and presets."""

    def __init__(self, templates_dir: Optional[str] = None, schemas_dir: Optional[str] = None):
        """Initialize configuration registry.
        
        Args:
            templates_dir: Directory containing configuration templates
            schemas_dir: Directory containing schema files
        """
        self.templates_dir = templates_dir or os.path.join(os.path.dirname(__file__), 'templates')
        self.schemas_dir = schemas_dir or os.path.join(os.path.dirname(__file__), 'schemas')
        self.validator = SchemaValidator(self.schemas_dir)
        self._templates: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all available configuration templates."""
        if not os.path.exists(self.templates_dir):
            return

        # Find all .ini files in templates directory
        template_pattern = os.path.join(self.templates_dir, '*.ini')
        for template_path in glob.glob(template_pattern):
            template_name = Path(template_path).stem
            self._templates[template_name] = template_path

    def list_templates(self) -> List[str]:
        """Get list of available configuration templates.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())

    def get_template_path(self, template_name: str) -> Optional[str]:
        """Get path to a specific template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Path to template file, or None if not found
        """
        return self._templates.get(template_name)

    def load_template(self, template_name: str) -> ConfigManager:
        """Load a configuration template.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            ConfigManager instance with loaded template
            
        Raises:
            ValueError: If template is not found
        """
        template_path = self.get_template_path(template_name)
        if not template_path:
            raise ValueError(f"Template '{template_name}' not found. Available: {self.list_templates()}")

        return ConfigManager(template_path, self.schemas_dir)

    def create_custom_config(self, base_template: str = 'default',
                             overrides: Optional[Dict[str, Any]] = None) -> ConfigManager:
        """Create a custom configuration based on a template with overrides.
        
        Args:
            base_template: Name of base template to use
            overrides: Dictionary of configuration overrides
            
        Returns:
            ConfigManager instance with custom configuration
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
                    # Assume it's a general setting
                    config_manager.update_config('general_settings', section_key, value)

        return config_manager

    def validate_config(self, config_path: str) -> List[str]:
        """Validate a configuration file.
        
        Args:
            config_path: Path to configuration file to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        try:
            ConfigManager(config_path, self.schemas_dir)
            return []
        except (ValueError, FileNotFoundError) as e:
            return [str(e)]

    def get_config_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined configuration profiles for different use cases.
        
        Returns:
            Dictionary of configuration profiles
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
        
        Args:
            profile_name: Name of the profile to use
            additional_overrides: Additional overrides to apply
            
        Returns:
            ConfigManager instance with profile configuration
            
        Raises:
            ValueError: If profile is not found
        """
        profiles = self.get_config_profiles()
        if profile_name not in profiles:
            raise ValueError(f"Profile '{profile_name}' not found. Available: {list(profiles.keys())}")

        profile = profiles[profile_name]
        overrides = profile['overrides'].copy()

        # Add any additional overrides
        if additional_overrides:
            overrides.update(additional_overrides)

        return self.create_custom_config(profile['template'], overrides)

    def export_config_template(self, config_manager: ConfigManager,
                               template_name: str, description: str = "") -> str:
        """Export a configuration as a new template.
        
        Args:
            config_manager: ConfigManager instance to export
            template_name: Name for the new template
            description: Optional description for the template
            
        Returns:
            Path to the exported template file
        """
        template_path = os.path.join(self.templates_dir, f"{template_name}.ini")

        # Add description as comment if provided
        if description:
            # Would need to implement comment injection in save_config
            pass

        config_manager.save_config(template_path, 'ini')

        # Reload templates to include the new one
        self._load_templates()

        return template_path
