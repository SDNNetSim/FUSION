"""Configuration-related exception classes for the FUSION CLI."""


class ConfigError(Exception):
    """Base exception for configuration errors."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when config file cannot be found."""


class ConfigParseError(ConfigError):
    """Raised when config file cannot be parsed."""


class MissingRequiredOptionError(ConfigError):
    """Raised when a required option is missing from config."""


class ConfigTypeConversionError(ConfigError):
    """Raised when a config value cannot be converted to expected type."""
