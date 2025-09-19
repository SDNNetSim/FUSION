"""Custom exceptions for the RL plotting module."""


class PlottingError(Exception):
    """Base exception for plotting module errors."""


class DataLoadError(PlottingError):
    """Raised when data loading fails."""


class MetricProcessingError(PlottingError):
    """Raised when metric processing fails."""


class InvalidConfigurationError(PlottingError):
    """Raised when plotting configuration is invalid."""


class PlottingFileNotFoundError(PlottingError):
    """Raised when required files are not found."""


class InvalidDataFormatError(PlottingError):
    """Raised when data format is not as expected."""
