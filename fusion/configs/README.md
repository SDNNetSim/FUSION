# FUSION Configuration System

This directory contains the modular configuration management system for the FUSION simulator, implementing the architecture plan requirements for scalable and validated configuration handling.

## 🚀 Quick Start Tutorial

New to FUSION? Follow this comprehensive tutorial to get started with the configuration system.

### Step 1: Understanding Configuration Files

FUSION uses INI configuration files to control simulation parameters. These files are organized into sections:

```ini
[general_settings]
erlang_start = 300       # Starting traffic load
erlang_stop = 1200       # Ending traffic load
num_requests = 1000      # Number of connection requests

[topology_settings]  
network = NSFNet         # Network topology to use
cores_per_link = 3       # Number of fiber cores per link

[spectrum_settings]
c_band = 320            # Number of spectral slots in C-band
```

### Step 2: Using Configuration Templates

Instead of writing configurations from scratch, use our pre-built templates:

```bash
# List available templates
ls fusion/configs/templates/

# Templates available:
# - default.ini          # Full-featured default configuration
# - minimal.ini           # Minimal setup for quick testing  
# - rl_training.ini       # Optimized for RL training
# - cross_platform.ini    # Cross-platform compatible paths
```

### Step 3: Running Your First Simulation

```bash
# Option 1: Use a template directly
python -m fusion.cli.run_sim run_sim --config_path fusion/configs/templates/minimal.ini --run_id my_first_sim

# Option 2: Copy and modify a template
cp fusion/configs/templates/default.ini my_config.ini
# Edit my_config.ini with your preferred values
python -m fusion.cli.run_sim run_sim --config_path my_config.ini --run_id custom_sim
```

### Step 4: Understanding Schemas

Schemas define what configuration options are valid and ensure your configs are correct:

```python
# The schema defines:
# - Required fields (like erlang_start, network)
# - Valid values (network must be "NSFNet", "Pan-European", or "USbackbone60")  
# - Data types (num_requests must be integer ≥ 1)
# - Value ranges (cores_per_link must be ≥ 1)
```

### Step 5: Configuration Validation

Your configuration is automatically validated when loaded:

```python
from fusion.configs import ConfigManager

# This will validate against schema and show helpful error messages
config_manager = ConfigManager('my_config.ini')

# Example validation error:
# ValidationError: topology_settings.network: 'InvalidNetwork' is not one of ['NSFNet', 'Pan-European', 'USbackbone60']
```

## Directory Structure

```
configs/
├── __init__.py              # Main configuration exports
├── config.py                # Core ConfigManager class
├── validate.py              # Schema validation system
├── cli_to_config.py         # CLI argument mapping
├── registry.py              # Configuration templates and profiles
├── schemas/                 # JSON schema files for validation
│   └── main.json           # Main configuration schema
├── templates/               # Configuration templates
│   ├── default.ini         # Default configuration
│   ├── minimal.ini         # Minimal test configuration
│   ├── rl_training.ini     # RL-optimized configuration
│   └── *.ini               # Other template files
└── README.md               # This file
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

## Usage Examples

### Basic Configuration Loading
```python
from fusion.configs import ConfigManager

# Load configuration file
config_manager = ConfigManager('path/to/config.ini')
config = config_manager.get_config()

# Access configuration sections
general_settings = config.general
topology_settings = config.topology
```

### Using Configuration Registry
```python
from fusion.configs import ConfigRegistry

# Initialize registry
registry = ConfigRegistry()

# List available templates
templates = registry.list_templates()

# Load a template
config_manager = registry.load_template('rl_training')

# Create custom configuration
custom_config = registry.create_custom_config(
    base_template='default',
    overrides={
        'erlang_start': 500,
        'max_iters': 10
    }
)
```

### Using Configuration Profiles
```python
# Get predefined profiles
profiles = registry.get_config_profiles()

# Create configuration from profile
dev_config = registry.create_profile_config('development')
prod_config = registry.create_profile_config('production')

# Profile with additional overrides
test_config = registry.create_profile_config(
    'quick_test',
    additional_overrides={'num_requests': 25}
)
```

### CLI Integration
```python
from fusion.configs import ConfigManager, CLIToConfigMapper

# Load base configuration
config_manager = ConfigManager('base_config.ini')

# Map CLI arguments and merge
cli_args = {'erlang_start': 400, 'max_iters': 5}
config_manager.merge_cli_args(cli_args)
```

## Configuration Profiles

The system includes predefined profiles for common use cases:

- **quick_test**: Fast configuration for testing (minimal resources)
- **development**: Development setup with detailed logging
- **production**: Optimized production configuration
- **rl_experiment**: Reinforcement learning experiment setup
- **benchmark**: High-resource benchmarking configuration

## Schema Validation

Configuration files are validated against JSON schemas in the `schemas/` directory:

- `main.json`: Primary configuration schema with all sections
- Additional schemas can be added for specific modules or use cases

Validation includes:
- Type checking (string, number, boolean, object, array)
- Required field validation
- Value range constraints
- Enumerated value validation

## Template System

Templates in the `templates/` directory provide:

- **default.ini**: Comprehensive default configuration with all sections
- **minimal.ini**: Minimal configuration for quick testing
- **rl_training.ini**: Optimized for reinforcement learning training
- **Legacy templates**: Existing configurations moved from `ini/` directory

## Migration from Legacy System

The new configuration system maintains backward compatibility with existing INI files while providing enhanced features:

1. **Legacy INI files** are automatically supported
2. **Existing configurations** can be loaded without modification
3. **New features** (validation, templates, profiles) are opt-in
4. **CLI integration** remains consistent

## 📚 Advanced Configuration Guide

### Using Configuration Profiles

Profiles provide preset configurations for common scenarios:

```python
from fusion.configs import ConfigRegistry

registry = ConfigRegistry()

# Available profiles:
# - quick_test: Fast testing with minimal resources
# - development: Development setup with detailed logging  
# - production: Optimized production configuration
# - rl_experiment: Reinforcement learning setup
# - benchmark: High-resource benchmarking

# Load a profile
config_manager = registry.create_profile_config('quick_test')

# Customize a profile  
custom_config = registry.create_profile_config(
    'rl_experiment',
    additional_overrides={
        'max_iters': 20,
        'device': 'cuda'
    }
)
```

### CLI Integration and Overrides

Override configuration values directly from the command line:

```bash
# Override specific configuration values
python -m fusion.cli.run_sim run_sim \
  --config_path my_config.ini \
  --run_id test_run \
  --erlang_start 500 \
  --erlang_stop 2000 \
  --max_iters 5
```

### Schema Reference

#### Required Configuration Sections

**`[general_settings]`** - Core simulation parameters
- `erlang_start` (number): Starting Erlang load, must be ≥ 0
- `erlang_stop` (number): Ending Erlang load, must be ≥ 0  
- `num_requests` (integer): Number of requests, must be ≥ 1

**`[topology_settings]`** - Network topology configuration
- `network` (string): Must be one of: "NSFNet", "Pan-European", "USbackbone60"

**`[spectrum_settings]`** - Spectral resource configuration
- `c_band` (integer): Number of C-band slots, must be ≥ 1

#### Optional Configuration Sections

**`[snr_settings]`** - Signal-to-noise ratio parameters
**`[rl_settings]`** - Reinforcement learning parameters (when using RL agents)

For complete schema details, see `fusion/configs/schemas/main.json`

## 🛠️ Troubleshooting Guide

### Common Configuration Errors

**❌ Missing Required Fields**
```
ValidationError: general_settings.erlang_start is required
```
**✅ Solution**: Add the missing field to your configuration:
```ini
[general_settings]
erlang_start = 300
```

**❌ Invalid Network Name**
```
ValidationError: topology_settings.network: 'MyNetwork' is not one of ['NSFNet', 'Pan-European', 'USbackbone60']
```
**✅ Solution**: Use one of the supported network topologies:
```ini
[topology_settings]
network = NSFNet
```

**❌ Invalid Data Types**
```
ValidationError: general_settings.num_requests: 'abc' is not of type 'integer'
```
**✅ Solution**: Use correct data types:
```ini
[general_settings]
num_requests = 1000  # Integer, not string
```

### Configuration File Not Found

If you get a "Configuration file not found" error:

1. Check the file path is correct
2. Use absolute paths or paths relative to your current working directory
3. Ensure the file has `.ini` extension

### Template vs Custom Configurations

**Use Templates When:**
- Starting a new project
- Need a working baseline configuration
- Want to follow best practices

**Create Custom Configurations When:**
- Need specific parameter combinations
- Running specialized experiments
- Have unique topology requirements

## 🔧 Developer Guide

### Adding New Configuration Sections

1. **Update the schema** (`schemas/main.json`):
```json
{
  "properties": {
    "my_new_section": {
      "type": "object", 
      "properties": {
        "my_parameter": {"type": "string"}
      },
      "required": ["my_parameter"]
    }
  }
}
```

2. **Update ConfigManager** to handle the new section
3. **Add CLI arguments** in `fusion/cli/args/` if needed
4. **Create template** with the new section populated

### Testing Configurations

Always validate new configurations:

```python
from fusion.configs import ConfigManager

# Test configuration loading
try:
    config_manager = ConfigManager('new_config.ini')
    print("✅ Configuration is valid!")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

## Best Practices

1. **Use templates** as starting points for new configurations
2. **Validate configurations** before running simulations
3. **Use profiles** for common scenarios instead of manual configuration
4. **Override specific values** rather than duplicating entire configurations
5. **Add new schemas** when introducing configuration sections
6. **Test configurations** in development before production use
7. **Use meaningful run_ids** for tracking simulation results
8. **Keep configurations in version control** for reproducibility