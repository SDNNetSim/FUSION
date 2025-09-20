# RL Module Configuration Constants

This directory contains configuration constants and argument definitions for the FUSION RL module.

## File Structure

### Core Files

- **`general_args.py`** - Algorithm definitions and strategy constants
  - `VALID_PATH_ALGORITHMS` - Algorithms for path selection
  - `VALID_CORE_ALGORITHMS` - Algorithms for core decisions
  - `VALID_DRL_ALGORITHMS` - Deep RL algorithms (abbreviated for compatibility)
  - `VALID_DEEP_REINFORCEMENT_LEARNING_ALGORITHMS` - Full name alias
  - `EPISODIC_STRATEGIES` - Exploration strategy options

- **`observation_args.py`** - Observation space configurations
  - `OBS_DICT` - Observation space definitions
  - `OBSERVATION_SPACE_DEFINITIONS` - Clearer alias for OBS_DICT
  - `VALID_OBSERVATION_FEATURES` - Set of all valid features

- **`registry_args.py`** - Algorithm registration mappings
  - `ALGORITHM_REGISTRY` - Maps algorithm names to setup/load/class

### New Additions

- **`constants.py`** - Comprehensive constants with enums (for new code)
  - Provides type-safe enums and better organization
  - Includes validation sets and default configurations
  
- **`__init__.py`** - Package exports for convenient imports

## Usage Examples

### Traditional Import (Backward Compatible)
```python
from fusion.modules.rl.args.general_args import VALID_PATH_ALGORITHMS
from fusion.modules.rl.args.observation_args import OBS_DICT
```

### Package Import (Recommended)
```python
from fusion.modules.rl.args import (
    VALID_PATH_ALGORITHMS,
    OBSERVATION_SPACE_DEFINITIONS,
    ALGORITHM_REGISTRY
)
```

### Using New Constants (For New Code)
```python
from fusion.modules.rl.args.constants import (
    AlgorithmType,
    ObservationSpaceTemplates,
    DefaultConfigurations
)

# Type-safe algorithm checking
if algorithm == AlgorithmType.PPO.value:
    # Handle PPO specific logic
    pass
```

## Important Notes

1. **Backward Compatibility**: Dictionary keys like `paths_cong` and `OBS_DICT` are maintained for compatibility but documented to indicate their full meaning.

2. **Variable Naming**: While we use descriptive names in new code, abbreviated keys in data structures are preserved to avoid breaking existing functionality.

3. **Type Annotations**: All files now include proper type hints following the coding standards.

4. **Documentation**: Each constant is documented with inline comments explaining its purpose.