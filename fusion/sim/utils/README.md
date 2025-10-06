# Simulation Utils Module

## Purpose
This module provides utility functions for FUSION simulations, including network analysis, spectrum management, data processing, and file I/O operations.

## Key Components
- `network.py`: Path analysis, congestion metrics, and network calculations
- `spectrum.py`: Spectrum allocation, channel management, and fragmentation analysis
- `data.py`: Matrix operations, data transformations, and statistical calculations
- `formatting.py`: String formatting and conversion utilities
- `simulation.py`: Simulation setup, Erlang calculations, and study management
- `io.py`: File operations for YAML and JSON configurations

## Usage
```python
# Import specific functions
from fusion.sim.utils import find_path_length, get_super_channels

# Or import all utilities
from fusion.sim.utils import *
```

## Dependencies
- Internal: fusion.utils.logging_config
- External: numpy, networkx, yaml
