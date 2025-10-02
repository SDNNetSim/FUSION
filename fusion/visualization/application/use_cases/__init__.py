"""Use cases for the application layer."""

from fusion.visualization.application.use_cases.batch_generate_plots import (
    BatchGeneratePlotsUseCase,
)
from fusion.visualization.application.use_cases.compare_algorithms import (
    CompareAlgorithmsUseCase,
)
from fusion.visualization.application.use_cases.generate_plot import GeneratePlotUseCase

__all__ = [
    "GeneratePlotUseCase",
    "BatchGeneratePlotsUseCase",
    "CompareAlgorithmsUseCase",
]
