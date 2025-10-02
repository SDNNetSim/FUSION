"""Domain exceptions for the visualization system."""

from fusion.visualization.domain.exceptions.domain_exceptions import (
    DataFormatError,
    DataSourceNotFoundError,
    InsufficientDataError,
    InvalidMetricPathError,
    InvalidStateError,
    MetadataNotFoundError,
    MetricExtractionError,
    PlotGenerationError,
    ProcessingError,
    RenderError,
    RepositoryError,
    RunDataNotFoundError,
    UnsupportedDataFormatError,
    ValidationError,
    VisualizationDomainError,
)

__all__ = [
    "VisualizationDomainError",
    "ValidationError",
    "InvalidStateError",
    "UnsupportedDataFormatError",
    "MetricExtractionError",
    "PlotGenerationError",
    "InvalidMetricPathError",
    "DataSourceNotFoundError",
    "InsufficientDataError",
    "RepositoryError",
    "ProcessingError",
    "RenderError",
    "RunDataNotFoundError",
    "DataFormatError",
    "MetadataNotFoundError",
]
