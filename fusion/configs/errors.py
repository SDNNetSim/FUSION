"""Configuration-related exception classes for the FUSION CLI."""


class ConfigError(Exception):
    """Base exception for configuration errors.
    
    All configuration-related exceptions inherit from this class,
    allowing for broad exception handling when needed.
    """


class ConfigFileNotFoundError(ConfigError):
    """Raised when config file cannot be found.
    
    This exception is raised when attempting to load a configuration
    file that doesn't exist at the specified path.
    """


class ConfigParseError(ConfigError):
    """Raised when config file cannot be parsed.
    
    This exception is raised when a configuration file exists but
    contains invalid syntax or cannot be parsed in the expected format
    (INI, JSON, YAML).
    """


class MissingRequiredOptionError(ConfigError):
    """Raised when a required option is missing from config.
    
    This exception is raised during configuration validation when
    a required parameter is not present in the configuration.
    """


class ConfigTypeConversionError(ConfigError):
    """Raised when a config value cannot be converted to expected type.
    
    This exception is raised when a configuration value cannot be
    converted to its expected type (e.g., string to int conversion fails).
    """
