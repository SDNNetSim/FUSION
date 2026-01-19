# Spectrum Assignment Module

## Overview

The Spectrum Assignment module provides sophisticated algorithms for allocating spectrum resources in optical networks. This module implements multiple spectrum assignment strategies to optimize bandwidth utilization and minimize fragmentation in wavelength division multiplexing (WDM) and elastic optical networks (EON).

## Features

- **Multiple Assignment Algorithms**: First Fit, Best Fit, and Last Fit strategies
- **Multi-band Support**: Supports C-band, L-band, and other frequency bands
- **Multi-core Fiber Support**: Handles space division multiplexing (SDM)
- **Light Path Slicing**: Advanced segmentation for large bandwidth requests
- **Fragmentation Management**: Algorithms designed to minimize spectrum fragmentation
- **Dynamic Registry**: Pluggable algorithm architecture with runtime selection
- **Type Safety**: Full mypy compliance with modern Python type hints
- **Comprehensive Testing**: Extensive test coverage for all algorithms

## Module Structure

```
fusion/modules/spectrum/
├── __init__.py              # Module exports and public API
├── README.md                # This documentation
├── registry.py              # Algorithm registry and factory
├── utils.py                 # Helper utilities and shared functions
├── best_fit.py              # Best Fit algorithm implementation
├── first_fit.py             # First Fit algorithm implementation
├── last_fit.py              # Last Fit algorithm implementation
└── light_path_slicing.py    # Light path slicing manager
```

## Components

### Algorithm Registry (`registry.py`)

The `SpectrumRegistry` class provides a centralized registry for all spectrum assignment algorithms:

```python
from fusion.modules.spectrum import SpectrumRegistry, create_spectrum_algorithm

# Create algorithm instance
algorithm = create_spectrum_algorithm(
    name="best_fit",
    engine_props=engine_config,
    sdn_props=sdn_controller,
    route_props=routing_props
)

# List available algorithms
algorithms = list_spectrum_algorithms()
# ['first_fit', 'best_fit', 'last_fit']
```

### Spectrum Assignment Algorithms

#### First Fit Algorithm (`first_fit.py`)

Assigns spectrum by finding the first available contiguous set of slots:

```python
from fusion.modules.spectrum import FirstFitSpectrum

first_fit = FirstFitSpectrum(engine_props, sdn_props, route_props)
result = first_fit.assign(path=path_nodes, request=bandwidth_request)
```

**Characteristics:**
- Fast allocation time
- Simple implementation
- May cause fragmentation over time
- Good for scenarios with frequent allocation/deallocation

#### Best Fit Algorithm (`best_fit.py`)

Finds the smallest available contiguous set of slots that can accommodate the request:

```python
from fusion.modules.spectrum import BestFitSpectrum

best_fit = BestFitSpectrum(engine_props, sdn_props, route_props)
result = best_fit.assign(path=path_nodes, request=bandwidth_request)
```

**Characteristics:**
- Minimizes spectrum fragmentation
- Better resource utilization
- Slightly higher computational overhead
- Optimal for long-term network efficiency

#### Last Fit Algorithm (`last_fit.py`)

Assigns spectrum by finding the last (highest index) available contiguous slots:

```python
from fusion.modules.spectrum import LastFitSpectrum

last_fit = LastFitSpectrum(engine_props, sdn_props, route_props)
result = last_fit.assign(path=path_nodes, request=bandwidth_request)
```

**Characteristics:**
- Useful for specific allocation patterns
- Can help distribute spectrum usage
- Complements other algorithms in hybrid scenarios

### Spectrum Helpers (`utils.py`)

The `SpectrumHelpers` class provides utility functions for spectrum operations:

```python
from fusion.modules.spectrum.utils import SpectrumHelpers

helpers = SpectrumHelpers(engine_props, sdn_props, spectrum_props)

# Check spectrum availability
helpers.check_other_links()

# Find best core for allocation
best_core = helpers.find_best_core()

# Check super channels
can_allocate = helpers.check_super_channels(open_slots_matrix, "normal")
```

### Light Path Slicing (`light_path_slicing.py`)

Manages segmented allocation for large bandwidth requests:

```python
from fusion.modules.spectrum import LightPathSlicingManager

slicing_mgr = LightPathSlicingManager(engine_props, sdn_props, spectrum_obj)

# Static slicing
success = slicing_mgr.handle_static_slicing(path_list, forced_segments=-1)

# Dynamic slicing
slicing_mgr.handle_dynamic_slicing(path_list, path_index, forced_segments)
```

## Usage Examples

### Basic Spectrum Assignment

```python
from fusion.modules.spectrum import create_spectrum_algorithm

# Initialize algorithm
algorithm = create_spectrum_algorithm(
    "best_fit",
    engine_props,
    sdn_props,
    route_props
)

# Define path and request
path = [1, 2, 3, 4]  # Node IDs
request = BandwidthRequest(bandwidth=100, modulation="QPSK")

# Assign spectrum
assignment = algorithm.assign(path, request)

if assignment:
    print(f"Assigned slots {assignment['start_slot']}-{assignment['end_slot']}")
    print(f"Core: {assignment['core_number']}, Band: {assignment['band']}")
```

### Multi-band Assignment

```python
# Configure engine for multi-band operation
engine_props = {
    "band_list": ["c", "l"],  # C-band and L-band
    "cores_per_link": 7,      # Multi-core fiber
    "guard_slots": 1,         # Guard band
    "allocation_method": "best_fit"
}

# Algorithm automatically handles multi-band
assignment = algorithm.assign(path, request)
```

### Dynamic Algorithm Selection

```python
from fusion.modules.spectrum import get_multiband_spectrum_algorithms

# Get algorithms supporting multi-band
multiband_algos = get_multiband_spectrum_algorithms()

# Select algorithm based on network conditions
if network_fragmentation > 0.7:
    algo_name = "best_fit"  # Minimize fragmentation
else:
    algo_name = "first_fit"  # Faster allocation

algorithm = create_spectrum_algorithm(algo_name, engine_props, sdn_props, route_props)
```

## Algorithm Parameters

### Engine Properties

```python
engine_props = {
    "cores_per_link": 7,           # Number of cores per fiber
    "band_list": ["c", "l"],       # Supported frequency bands
    "guard_slots": 1,              # Guard band slots
    "slots_per_gbps": 1,           # Slots per Gbps of bandwidth
    "allocation_method": "best_fit", # Default allocation strategy
    "max_segments": 10,            # Maximum slicing segments
    "fixed_grid": True             # Fixed or flexible grid
}
```

### Spectrum Properties

```python
spectrum_props = SpectrumProps()
spectrum_props.slots_needed = 4
spectrum_props.forced_core = None      # Optional core constraint
spectrum_props.forced_band = None      # Optional band constraint
spectrum_props.forced_index = None     # Optional slot constraint
spectrum_props.path_list = [1, 2, 3]   # Network path
```

## Performance Metrics

All algorithms provide performance metrics:

```python
metrics = algorithm.get_metrics()
print(f"Algorithm: {metrics['algorithm']}")
print(f"Assignments made: {metrics['assignments_made']}")
print(f"Total slots assigned: {metrics['total_slots_assigned']}")
print(f"Average slots per assignment: {metrics['average_slots_per_assignment']}")
print(f"Supports multiband: {metrics['supports_multiband']}")
```

## Fragmentation Analysis

```python
# Calculate fragmentation for a path
fragmentation = algorithm.get_fragmentation_metric(path)
print(f"Path fragmentation: {fragmentation:.3f}")

# Lower values indicate better spectrum efficiency
```

## Dependencies

- **numpy**: Numerical operations and array processing
- **fusion.core.properties**: Core property classes
- **fusion.interfaces.spectrum**: Abstract base classes
- **fusion.sim.utils**: Simulation utilities
- **fusion.utils.logging_config**: Logging configuration

## Type Safety

This module is fully typed with mypy compliance:

```python
from typing import Any, Generator
from fusion.modules.spectrum import AbstractSpectrumAssigner

# All algorithms implement the interface
def process_algorithm(algo: AbstractSpectrumAssigner) -> dict[str, Any]:
    return algo.assign(path=[1, 2, 3], request=mock_request)
```

## Testing

Run the spectrum module tests:

```bash
# Run all spectrum tests
pytest tests/modules/spectrum/

# Run specific algorithm tests
pytest tests/modules/spectrum/test_best_fit.py
pytest tests/modules/spectrum/test_first_fit.py
pytest tests/modules/spectrum/test_last_fit.py
```

## Contributing

When contributing to the spectrum module:

1. **Follow Type Hints**: All functions must have proper type annotations
2. **Docstring Format**: Use Sphinx-style docstrings (`:param:`, `:return:`)
3. **Algorithm Interface**: Implement `AbstractSpectrumAssigner` for new algorithms
4. **Registry Registration**: Register new algorithms in the registry
5. **Testing**: Add comprehensive tests for new algorithms
6. **Performance**: Include fragmentation metrics and performance analysis

## Related Modules

- **[SNR Module](../snr/README.md)**: Signal-to-noise ratio calculations
- **[Routing Module](../routing/README.md)**: Path computation algorithms
- **[Modulation Module](../modulation/README.md)**: Modulation format selection

## References

1. Christodoulopoulos, K., et al. "Elastic Bandwidth Allocation in Flexible OFDM-Based Optical Networks." Journal of Lightwave Technology, 2011.
2. Wang, Y., et al. "Spectrum Assignment in Elastic Optical Networks." IEEE/OSA Journal of Optical Communications and Networking, 2013.
3. Klinkowski, M., et al. "Survey of Resource Allocation Schemes and Algorithms in Spectrally-Spatially Flexible Optical Networking." Computer Networks, 2018.
