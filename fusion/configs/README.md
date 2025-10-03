# FUSION Configuration System

## Purpose
Modular configuration management system for FUSION simulator with validation, templates, and CLI integration.

## Quick Start

### Step 1: Use a Template
```bash
# List templates
ls fusion/configs/templates/
# default.ini, minimal.ini, rl_training.ini, cross_platform.ini

# Run simulation with template
python -m fusion.cli.run_sim run_sim --config_path fusion/configs/templates/minimal.ini --run_id test
```

### Step 2: Create Custom Configuration
```ini
[general_settings]
erlang_start = 300
erlang_stop = 1200
num_requests = 1000

[topology_settings]
network = NSFNet
cores_per_link = 3

[spectrum_settings]
c_band = 320
```

### Step 3: Validation
```python
from fusion.configs import ConfigManager
config = ConfigManager('my_config.ini')  # Auto-validates
```

## Architecture

```
configs/
├── Core Components
│   ├── config.py            # ConfigManager - loads, validates, manages configs
│   ├── validate.py          # SchemaValidator - JSON schema validation
│   ├── registry.py          # ConfigRegistry - templates and profiles
│   └── cli_to_config.py     # CLIToConfigMapper - CLI argument mapping
├── Support Files
│   ├── errors.py            # Custom exception classes
│   ├── constants.py         # Module constants
│   └── schema.py            # Configuration schemas
├── schemas/                 # JSON validation schemas
└── templates/               # Pre-built configurations
    ├── default.ini          # Full-featured baseline
    ├── minimal.ini          # Quick testing
    ├── rl_training.ini      # RL experiments
    └── cross_platform.ini   # OS-agnostic
```

## Key Components

### ConfigManager
Central configuration management class that:
- Loads configuration from INI, JSON, or YAML files
- Validates configuration against schemas
- Provides structured access to configuration sections
- Supports configuration merging and CLI argument integration

### SchemaValidator
JSON schema-based validation system that:
- Validates configuration structure and types
- Enforces required fields and value constraints
- Provides detailed error reporting
- Generates default configurations from schemas

### ConfigRegistry
Template and profile management system that:
- Manages configuration templates
- Provides predefined configuration profiles
- Supports custom configuration creation
- Enables configuration template export

### CLIToConfigMapper
Maps CLI arguments to configuration structure:
- Converts command-line arguments to config sections
- Supports argument overrides of base configurations
- Maintains consistency between CLI and config file options

## Usage

### Basic Loading
```python
from fusion.configs import ConfigManager
config = ConfigManager('config.ini').get_config()
print(config.general['erlang_start'])  # 300
```

### Templates & Profiles
```python
from fusion.configs import ConfigRegistry

registry = ConfigRegistry()

# Load template
config = registry.load_template('minimal')

# Use profile (quick_test, development, production, rl_experiment, benchmark)
config = registry.create_profile_config('quick_test')

# Custom config
config = registry.create_custom_config(
    'default',
    overrides={'erlang_start': 500}
)
```

### CLI Override
```python
config_manager = ConfigManager('base.ini')
config_manager.merge_cli_args({'max_iters': 5})
```

## Configuration Profiles

| Profile | Description | Key Settings |
|---------|-------------|-------------|
| `quick_test` | Fast testing | max_iters=1, num_requests=50 |
| `development` | Debug mode | print_step=5, save_snapshots=true |
| `production` | Optimized | max_iters=10, thread_erlangs=true |
| `rl_experiment` | RL training | n_trials=50, optimize_hyperparameters=true |
| `benchmark` | Performance | max_iters=20, num_requests=2000 |

## Templates

| Template | Use Case | Features |
|----------|----------|----------|
| `default.ini` | Production baseline | All parameters, balanced settings |
| `minimal.ini` | Quick tests | Only required params, fast execution |
| `rl_training.ini` | RL experiments | Epsilon-greedy, medium loads |
| `cross_platform.ini` | CI/CD | OS-agnostic paths, simple config |

## Migration Notes
- Legacy INI files work without modification
- New features (validation, templates) are opt-in
- CLI integration unchanged

## CLI Integration

```bash
# Override config values from CLI
python -m fusion.cli.run_sim run_sim \
  --config_path config.ini \
  --erlang_start 500 \
  --max_iters 5
```

## Parameter Reference

### Required Parameters

**[general_settings]**
- `erlang_start/stop/step`: Traffic load range (≥0)
- `max_iters`: Iterations per load (≥1)
- `num_requests`: Requests per iteration (≥1)
- `holding_time`: Connection duration (≥0)
- `route_method`: `k_shortest_path`, `xt_aware`
- `allocation_method`: `first_fit`, `priority_first`

**[topology_settings]**
- `network`: `NSFNet`, `Pan-European`, `USbackbone60`
- `bw_per_slot`: Bandwidth per slot in GHz
- `cores_per_link`: Fiber cores per link (≥1)

**[spectrum_settings]**
- `c_band`: Number of C-band slots (≥1)

### Key Optional Parameters

**[snr_settings]**
- `snr_type`: `None` or `xt_calculation`
- `xt_type`: `with_length` or `without_length`

**[rl_settings]**
- `path_algorithm`: `dqn`, `epsilon_greedy_bandit`, `first_fit`
- `device`: `cpu`, `cuda`, `mps`
- `n_trials`: Training iterations

**[ml_settings]**
- `deploy_model`: Enable ML model
- `ml_model`: `decision_tree`, `random_forest`

*See `fusion/configs/schemas/main.json` for complete parameter list*

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ValidationError: erlang_start is required` | Add missing field to config |
| `'MyNetwork' is not one of ['NSFNet'...]` | Use supported network name |
| `'abc' is not of type 'integer'` | Use correct data type |
| `Configuration file not found` | Check file path and extension |

## Dependencies
- **configparser**: INI file parsing
- **json**: JSON file handling
- **yaml** (optional): YAML file support
- **dataclasses**: Configuration data structures
