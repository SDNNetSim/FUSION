"""Data processors for transforming simulation data into plottable format."""

from fusion.visualization.infrastructure.processors.blocking_processor import (
    BlockingProbabilityProcessor,
)
from fusion.visualization.infrastructure.processors.multi_metric_processor import (
    MultiMetricProcessor,
)

# Alias for backward compatibility
BlockingProcessor = BlockingProbabilityProcessor

__all__ = [
    "BlockingProbabilityProcessor",
    "BlockingProcessor",
    "MultiMetricProcessor",
]
