"""Domain-specific exceptions for the visualization system."""


class VisualizationDomainError(Exception):
    """Base exception for all visualization domain errors."""


class ValidationError(VisualizationDomainError):
    """Raised when data validation fails."""


class InvalidStateError(VisualizationDomainError):
    """Raised when an operation is attempted in an invalid state."""


class UnsupportedDataFormatError(VisualizationDomainError):
    """Raised when no adapter can handle the data format."""


class MetricExtractionError(VisualizationDomainError):
    """Raised when metric extraction from data fails."""


class PlotGenerationError(VisualizationDomainError):
    """Raised when plot generation fails."""


class InvalidMetricPathError(VisualizationDomainError):
    """Raised when a metric path is invalid or not found."""


class DataSourceNotFoundError(VisualizationDomainError):
    """Raised when a data source cannot be found."""


class InsufficientDataError(VisualizationDomainError):
    """Raised when insufficient data is available for processing."""


class RepositoryError(VisualizationDomainError):
    """Raised when repository operations fail."""


class ProcessingError(VisualizationDomainError):
    """Raised when data processing fails."""


class RenderError(VisualizationDomainError):
    """Raised when plot rendering fails."""


class RunDataNotFoundError(VisualizationDomainError):
    """Raised when run data cannot be found."""


class DataFormatError(VisualizationDomainError):
    """Raised when data format is invalid or cannot be parsed."""


class MetadataNotFoundError(VisualizationDomainError):
    """Raised when metadata cannot be found."""
