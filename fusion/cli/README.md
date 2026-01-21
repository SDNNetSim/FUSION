# CLI Module

## Purpose
Command-line interface entry points and argument parsing for the FUSION optical network simulator. Provides a clean CLI architecture with modern Python practices, comprehensive error handling, and maintainable code organization.

**Example**: *This module handles CLI entry points for simulation and training operations with centralized argument parsing and proper error handling.*

## Key Components

### Core Files
- `run_sim.py`: Main network simulation runner with multiprocessing support
- `run_train.py`: Machine learning and reinforcement learning agent training entry point
- `main_parser.py`: Centralized parser construction with modern naming conventions
- `config_setup.py`: Configuration management with proper error handling and validation
- `constants.py`: Shared CLI constants including exit codes and settings
- `utils.py`: Common patterns and boilerplate code for CLI entry points

### Optional Files
- `parameters/`: Modular argument definitions and centralized registry system

## Usage

### Basic Usage
```python
from fusion.cli import build_main_argument_parser, setup_config_from_cli

# Create main parser with all subcommands
parser = build_main_argument_parser()
args = parser.parse_args()

# Set up configuration from CLI arguments
config = setup_config_from_cli(args)
```

### Entry Point Usage
```bash
# Run simulation
python -m fusion.cli.run_sim run_sim --config_path config.ini

# Train an agent
python -m fusion.cli.run_train --agent_type rl --config_path config.ini
```

## Dependencies

### Internal Dependencies
- `fusion.utils.logging_config`: Standardized logging setup
- `fusion.configs`: Configuration management and validation
- `fusion.sim`: Simulation and training pipeline modules

### External Dependencies
- `argparse`: Command-line argument parsing (built-in)
- `multiprocessing`: Process management for simulations (built-in)
- `pathlib`: Modern path handling (built-in)

## Configuration

### Required Configuration
```python
# CLI uses standard FUSION configuration format
config = {
    "config_path": "path/to/config.ini",
    "debug": False,
    "output_dir": "results/"
}
```

### Environment Variables
- `FUSION_DEBUG`: Enable debug mode for CLI operations
- `FUSION_CONFIG_PATH`: Default configuration file path

## Testing

Unit tests are located in `tests/` directory:
```bash
# Run CLI tests
pytest fusion/cli/tests/

# Run with coverage
pytest --cov=fusion.cli fusion/cli/tests/
```

## API Reference

### Main Functions
- `build_main_argument_parser()`: Build the main CLI argument parser with all subcommands
- `create_training_argument_parser()`: Create and parse arguments for training simulations
- `setup_config_from_cli()`: Set up configuration from command line input

### Key Classes
- `ConfigManager`: Centralized configuration management with validation and error handling

### Entry Points
- `run_sim.main()`: Entry point for running network simulations
- `run_train.main()`: Entry point for training ML/RL agents

## Examples

### Example 1: Basic Simulation
```python
from fusion.cli.run_sim import main
import multiprocessing

# Run simulation with default settings
stop_flag = multiprocessing.Event()
exit_code = main(stop_flag)
```

### Example 2: Configuration Management
```python
from fusion.cli import ConfigManager

# Create config manager from CLI arguments
config_manager = ConfigManager.from_args(args)

# Access configuration values
simulation_config = config_manager.get('s1')  # Default thread
debug_mode = config_manager.get_value('debug', default=False)
```

## Notes

### Design Decisions
- **Minimal Entry Points**: Entry scripts contain no business logic, delegating to pipeline modules
- **Centralized Argument Registry**: Consistent argument parsing across all CLI operations
- **Modern Error Handling**: User-friendly error messages with logging for debugging
- **Backward Compatibility**: Legacy function names maintained through compatibility aliases

---

## Architecture

The CLI module follows a clean architecture pattern:

1. **Entry Points** (`run_*.py`): Minimal scripts that handle argument parsing and delegate to pipelines
2. **Argument Management** (`main_parser.py`, `parameters/`): Centralized argument definition and parsing
3. **Configuration** (`config_setup.py`): Robust configuration loading with validation and error handling
4. **Utilities** (`utils.py`, `constants.py`): Shared patterns and constants for consistency

## Error Handling

The CLI provides comprehensive error handling with:
- **User-Friendly Messages**: Clear error descriptions with emoji indicators and helpful suggestions
- **Structured Logging**: Detailed error information logged for debugging
- **Proper Exit Codes**: Standard Unix exit codes for script integration
- **Graceful Interruption**: Clean handling of Ctrl+C and other interruptions

## Integration

This CLI module integrates with:
- **Simulation Engine**: Through `fusion.sim` pipeline modules
- **Configuration System**: Using `fusion.configs` for validation and loading
- **Logging System**: Via `fusion.utils.logging_config` for consistent logging
