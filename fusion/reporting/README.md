# Reporting Module

## Purpose
The reporting module provides utilities for formatting and reporting simulation results, separating presentation concerns from data collection in the FUSION simulation framework.

## Key Components
- `simulation_reporter.py`: Main SimulationReporter class for formatting and outputting simulation statistics
- `__init__.py`: Public API exports and module initialization

## Usage
```python
from fusion.reporting import SimulationReporter

# Initialize reporter
reporter = SimulationReporter(verbose=True)

# Report simulation progress
reporter.report_iteration_stats(
    iteration=0,
    max_iterations=100,
    erlang=50.0,
    blocking_list=[0.01, 0.02],
    print_flag=True
)

# Report simulation completion
reporter.report_simulation_complete(
    erlang=50.0,
    iterations_completed=100,
    confidence_interval=95.0
)
```

## Dependencies
### Internal Dependencies
- `fusion.utils.logging_config`: For logging configuration

### External Dependencies
- `logging`: Standard library logging
- `statistics`: Standard library statistics functions
- `typing`: Type annotations
