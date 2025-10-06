"""
Configuration management system for FUSION.

This module provides comprehensive configuration management for the FUSION simulator,
including loading, validation, templates, and CLI integration.

Main components:
- ConfigManager: Core configuration loading and management
- SchemaValidator: JSON schema-based validation
- ConfigRegistry: Template and profile management
- CLIToConfigMapper: CLI argument mapping
- Error classes: Specific configuration exceptions
"""

# Local application imports
from .cli_to_config import CLIToConfigMapper
from .config import ConfigManager, SimulationConfig
from .errors import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigTypeConversionError,
    MissingRequiredOptionError,
)
from .registry import ConfigRegistry
from .validate import SchemaValidator, ValidationError

__all__ = [
    # Core classes
    'ConfigManager',
    'SimulationConfig',
    'SchemaValidator',
    'CLIToConfigMapper',
    'ConfigRegistry',
    # Error classes
    'ConfigError',
    'ConfigFileNotFoundError',
    'ConfigParseError',
    'ConfigTypeConversionError',
    'MissingRequiredOptionError',
    'ValidationError'
]
