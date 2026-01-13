# Interfaces Module

## Purpose
Defines abstract base classes for all pluggable components in FUSION, ensuring consistent APIs across algorithm implementations.

## Key Components
- `router.py`: Base class for routing algorithms
- `spectrum.py`: Base class for spectrum assignment algorithms
- `snr.py`: Base class for SNR measurement algorithms
- `agent.py`: Base class for reinforcement learning agents
- `factory.py`: Factory classes for creating algorithm instances

## Usage
```python
from fusion.interfaces import AbstractRoutingAlgorithm

class MyRouter(AbstractRoutingAlgorithm):
    @property
    def algorithm_name(self) -> str:
        return "my_router"
    # Implement required methods...
```

## Dependencies
- ABC module for abstract base classes
- typing module for type hints
- Module registries in fusion.modules.*
