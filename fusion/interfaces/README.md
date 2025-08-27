# FUSION Interfaces

This directory contains the abstract base classes (interfaces) that define the contracts for all pluggable components in the FUSION architecture.

## Overview

The interfaces ensure that all algorithm implementations follow a consistent API, making them interchangeable and testable. This design pattern enables:

- **Pluggability**: Easy swapping of different algorithm implementations
- **Testability**: Clear contracts for unit testing
- **Maintainability**: Consistent API across all modules
- **Extensibility**: Easy addition of new algorithms

## Available Interfaces

### 1. AbstractRoutingAlgorithm (`router.py`)

Base class for all routing algorithms. Implementations must provide:

- `algorithm_name`: Unique identifier for the algorithm
- `supported_topologies`: List of network topologies the algorithm supports
- `validate_environment()`: Check if algorithm can work with given topology
- `route()`: Find a path from source to destination
- `get_paths()`: Get k shortest paths
- `update_weights()`: Update edge weights dynamically
- `get_metrics()`: Return performance metrics

**Example Implementation**: See `fusion/modules/routing/k_shortest_path.py`

### 2. AbstractSpectrumAssigner (`spectrum.py`)

Base class for spectrum assignment algorithms. Implementations must provide:

- `algorithm_name`: Unique identifier for the algorithm
- `supports_multiband`: Whether algorithm supports multi-band assignment
- `assign()`: Assign spectrum resources along a path
- `check_spectrum_availability()`: Verify spectrum slots are free
- `allocate_spectrum()`: Reserve spectrum resources
- `deallocate_spectrum()`: Release spectrum resources
- `get_fragmentation_metric()`: Calculate fragmentation level
- `get_metrics()`: Return performance metrics

### 3. AbstractSNRMeasurer (`snr.py`)

Base class for SNR measurement algorithms. Implementations must provide:

- `algorithm_name`: Unique identifier for the algorithm
- `supports_multicore`: Whether algorithm supports multi-core fibers
- `calculate_snr()`: Calculate SNR for a path
- `calculate_link_snr()`: Calculate SNR for a single link
- `calculate_crosstalk()`: Compute crosstalk noise
- `calculate_nonlinear_noise()`: Compute nonlinear noise components
- `get_required_snr_threshold()`: Get SNR requirement for modulation
- `is_snr_acceptable()`: Check if SNR meets requirements
- `update_link_state()`: Update state after allocation
- `get_metrics()`: Return performance metrics

### 4. AgentInterface (`agent.py`)

Base class for reinforcement learning agents. Implementations must provide:

- `algorithm_name`: Unique identifier for the algorithm
- `action_space_type`: 'discrete' or 'continuous'
- `observation_space_shape`: Dimensions of observations
- `act()`: Select action given observation
- `train()`: Train the agent
- `learn_from_experience()`: Update from single experience
- `save()/load()`: Model persistence
- `get_reward()`: Calculate reward for transitions
- `update_exploration_params()`: Adjust exploration over time
- `get_config()/set_config()`: Configuration management
- `get_metrics()`: Return performance metrics

## Implementation Guidelines

### 1. Creating a New Implementation

```python
from fusion.interfaces.router import AbstractRoutingAlgorithm

class MyCustomRouter(AbstractRoutingAlgorithm):
    def __init__(self, engine_props: dict, sdn_props: object):
        super().__init__(engine_props, sdn_props)
        # Initialize your algorithm-specific attributes
        
    @property
    def algorithm_name(self) -> str:
        return "my_custom_router"
        
    # Implement all other required methods...
```

### 2. Registering Your Implementation

After creating your implementation, register it in the appropriate registry:

```python
# In fusion/modules/routing/registry.py
from .my_custom_router import MyCustomRouter

ROUTING_ALGORITHMS = {
    'my_custom': MyCustomRouter,
    # ... other algorithms
}
```

### 3. Testing Interface Compliance

Use the provided test utilities to ensure your implementation follows the interface:

```python
# Run interface compliance tests
python -m pytest tests/test_interfaces.py
```

## Best Practices

1. **Type Hints**: Always use proper type hints as defined in the interfaces
2. **Documentation**: Document your implementation thoroughly
3. **Error Handling**: Handle edge cases gracefully
4. **State Management**: Use the `reset()` method to clear internal state
5. **Metrics**: Implement meaningful metrics in `get_metrics()`
6. **Validation**: Always validate inputs in your implementations

## Migration Guide

For existing code that needs to be migrated to use interfaces:

1. Identify the appropriate interface for your module
2. Update the class to inherit from the interface
3. Implement all required abstract methods
4. Update method signatures to match the interface
5. Add proper type hints
6. Update the module registry
7. Add unit tests for interface compliance