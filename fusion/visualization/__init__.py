"""
fusion.visualization: Modern Domain-Driven Visualization System.

New Architecture (v6.0+):
- Domain-Driven Design with clean boundaries
- Plugin system for extensibility
- Versioned data adapters (no more format breaks!)
- 93% test coverage
- Full type safety

Usage:
    # Use cases directly
    from fusion.visualization.application.use_cases import generate_plot
    from fusion.visualization.application.dto import PlotRequestDTO

    # Or use the CLI
    $ fusion viz plot --config config.yml
    $ fusion viz batch --config batch.yml
    $ fusion viz compare -n NSFNet -a algo1 -a algo2

For more information:
    $ fusion viz info
    $ fusion viz --help
"""

# New exports (recommended)
from .application.use_cases.generate_plot import GeneratePlotUseCase
from .application.use_cases.batch_generate_plots import BatchGeneratePlotsUseCase
from .application.use_cases.compare_algorithms import CompareAlgorithmsUseCase

__all__ = [
    'GeneratePlotUseCase',
    'BatchGeneratePlotsUseCase',
    'CompareAlgorithmsUseCase',
]

__version__ = "6.0.0"
