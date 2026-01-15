# Interfaces

## Purpose

Defines abstract base classes and Protocol classes for all pluggable components in FUSION, ensuring consistent APIs across algorithm implementations. Supports both legacy ABC-based interfaces and modern Protocol-based pipeline definitions.

## Key Components

### Legacy Abstract Base Classes
- `router.py`: `AbstractRoutingAlgorithm` - Base class for routing algorithms
- `spectrum.py`: `AbstractSpectrumAssigner` - Base class for spectrum assignment algorithms
- `snr.py`: `AbstractSNRMeasurer` - Base class for SNR measurement algorithms
- `agent.py`: `AgentInterface` - Base class for reinforcement learning agents
- `factory.py`: `AlgorithmFactory` and `SimulationPipeline` - Factory classes for creating algorithm instances

### Pipeline Protocols
- `pipelines.py`: Protocol definitions for pipeline architecture
  - `RoutingPipeline`: Find candidate routes between nodes
  - `SpectrumPipeline`: Find available spectrum along paths
  - `GroomingPipeline`: Pack requests onto existing lightpaths
  - `SNRPipeline`: Validate signal quality
  - `SlicingPipeline`: Divide requests into smaller allocations

### Control Policy Protocol
- `control_policy.py`: `ControlPolicy` protocol for unified path selection
  - Supports heuristics, RL policies, and supervised/unsupervised learning policies
  - `PolicyAction` type alias for action results

## Usage

### Legacy Abstract Base Classes
```python
from fusion.interfaces import AbstractRoutingAlgorithm

class MyRouter(AbstractRoutingAlgorithm):
    @property
    def algorithm_name(self) -> str:
        return "my_router"
    # Implement required methods...
```

### Pipeline Protocols
```python
from fusion.interfaces import RoutingPipeline

# Type hint a parameter as accepting any routing implementation
def process_request(
    router: RoutingPipeline,
    network_state: NetworkState,
) -> RouteResult:
    return router.find_routes("A", "B", 100, network_state)

# Check if object implements protocol (runtime_checkable)
if isinstance(my_router, RoutingPipeline):
    result = my_router.find_routes(...)
```

### Control Policy Protocol
```python
from fusion.interfaces import ControlPolicy

class MyPolicy:
    def select_action(self, request, options, network_state) -> int:
        return 0  # Always select first option

    def update(self, request, action, reward) -> None:
        pass  # No learning

    def get_name(self) -> str:
        return "MyPolicy"

policy = MyPolicy()
isinstance(policy, ControlPolicy)  # True
```

## Dependencies

### Internal Dependencies
- `fusion.domain.network_state`: NetworkState for pipeline protocols
- `fusion.domain.request`: Request objects
- `fusion.domain.results`: Result dataclasses (RouteResult, SpectrumResult, etc.)
- `fusion.modules.rl.adapter`: PathOption for control policy

### External Dependencies
- `abc`: Abstract base classes
- `typing`: Type hints and Protocol classes

## Testing

Unit tests are located in `tests/` directory:
```bash
# Run all interface tests
pytest fusion/interfaces/tests/

# Run specific test file
pytest fusion/interfaces/tests/test_router.py

# Run with coverage
pytest --cov=fusion.interfaces fusion/interfaces/tests/
```

## API Reference

### Legacy Classes
- `AbstractRoutingAlgorithm`: Base class for routing algorithms
- `AbstractSpectrumAssigner`: Base class for spectrum assignment
- `AbstractSNRMeasurer`: Base class for SNR measurement
- `AgentInterface`: Base class for RL agents
- `AlgorithmFactory`: Factory for creating algorithm instances
- `SimulationPipeline`: Complete processing pipeline

### Pipeline Protocols
- `RoutingPipeline`: Route finding interface
- `SpectrumPipeline`: Spectrum assignment interface
- `GroomingPipeline`: Traffic grooming interface
- `SNRPipeline`: SNR validation interface
- `SlicingPipeline`: Request slicing interface

### Control Policy
- `ControlPolicy`: Unified path selection interface
- `PolicyAction`: Type alias for action results (int)

## Notes

### Design Decisions
- Legacy ABCs use explicit inheritance for interface enforcement
- V5 Protocols use structural typing (duck typing) for flexibility
- `@runtime_checkable` decorator enables isinstance() checks on protocols
- Control policy protocol unifies heuristic and RL-based path selection
