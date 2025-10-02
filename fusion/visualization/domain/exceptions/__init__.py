"""Domain exceptions for the visualization system."""

from fusion.visualization.domain.exceptions.domain_exceptions import (
    VisualizationDomainError,
    ValidationError,
    InvalidStateError,
    UnsupportedDataFormatError,
    MetricExtractionError,
    PlotGenerationError,
    InvalidMetricPathError,
    DataSourceNotFoundError,
    InsufficientDataError,
    RepositoryError,
    ProcessingError,
    RenderError,
    RunDataNotFoundError,
    DataFormatError,
    MetadataNotFoundError,
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
