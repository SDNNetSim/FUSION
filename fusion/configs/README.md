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

Templates in the `templates/` directory provide pre-configured setups for different use cases:

### Available Templates

#### **default.ini** - Comprehensive Default Configuration
Complete configuration with all sections and reasonable defaults for production use.

**Key Features:**
- Full spectrum of parameters across all sections
- Balanced settings for general-purpose simulations
- Comprehensive RL settings for machine learning experiments
- Well-documented parameter choices

**Best For:** Production simulations, baseline experiments, comprehensive testing

#### **minimal.ini** - Quick Testing Configuration  
Stripped-down configuration with only essential parameters for rapid testing.

**Key Features:**
- Only required parameters included
- Fast execution with reduced complexity
- Small traffic loads (300-500 Erlang)
- Limited iterations (max_iters = 2)

**Best For:** Unit testing, development debugging, quick validation

#### **runtime_config.ini** - Runtime Optimized Configuration
Optimized for runtime performance with epsilon-greedy algorithms.

**Key Features:**
- Epsilon-greedy bandit path algorithm for faster convergence
- Medium traffic loads (300-500 Erlang)
- Balanced between speed and accuracy
- Thread section example (s2) included

**Best For:** Development testing, algorithm comparison, runtime benchmarks

#### **xtar_example_config.ini** - Cross-Talk Aware Configuration
Specialized configuration for cross-talk aware simulations with XTAR assumptions.

**Key Features:**
- XTAR modulation assumptions enabled
- Pan-European network topology
- Cross-talk aware routing (xt_aware)
- Priority-first allocation method
- High traffic loads (2000+ Erlang)
- Extended holding times (3600s)

**Best For:** Cross-talk research, advanced optical simulations, European network studies

#### **cross_platform.ini** - Cross-Platform Compatible Configuration
Standardized configuration ensuring compatibility across different operating systems.

**Key Features:**
- Platform-agnostic file paths
- Single request size distribution (100 Gbps only)
- Simplified parameter set
- Conservative resource usage

**Best For:** CI/CD pipelines, multi-platform testing, containerized deployments

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

## 📋 Complete Parameter Reference

### `[general_settings]` - Core Simulation Parameters

#### **Traffic Load Parameters**
- `erlang_start` (number, ≥0): Starting traffic load in Erlangs (e.g., 300)
- `erlang_stop` (number, ≥0): Ending traffic load in Erlangs (e.g., 1200)  
- `erlang_step` (number, ≥0): Step size for traffic load increments (e.g., 300)

#### **Simulation Control**
- `max_iters` (integer, ≥1): Maximum simulation iterations per Erlang point (e.g., 4)
- `num_requests` (integer, ≥1): Number of connection requests per iteration (e.g., 500)
- `holding_time` (number, ≥0): Average connection holding time in seconds (e.g., 0.2)

#### **Routing & Spectrum Assignment**
- `route_method` (string): Routing algorithm
  - `k_shortest_path`: K-shortest path routing
  - `xt_aware`: Cross-talk aware routing
- `allocation_method` (string): Spectrum allocation method
  - `first_fit`: First-fit allocation
  - `priority_first`: Priority-based allocation  
- `k_paths` (integer, ≥1): Number of candidate paths to consider (e.g., 4)

#### **Request Generation**
- `request_distribution` (JSON object): Distribution of request sizes in Gbps
  ```ini
  request_distribution = {"25": 0.1, "50": 0.1, "100": 0.5, "200": 0.2, "400": 0.1}
  ```

#### **Modulation & Grid Settings**
- `mod_assumption` (string): Modulation assumption type
  - `DEFAULT`: Standard modulation formats
  - `XTAR_ASSUMPTIONS`: Cross-talk aware modulation
- `mod_assumption_path` (string): Path to modulation format definitions
- `fixed_grid` (boolean): Enable fixed grid mode (true/false)
- `guard_slots` (integer, ≥0): Number of guard slots between channels

#### **Performance Optimization**
- `thread_erlangs` (boolean): Enable parallel processing of Erlang points
- `dynamic_lps` (boolean): Enable dynamic lightpath sizing
- `pre_calc_mod_selection` (boolean): Pre-calculate modulation selection
- `max_segments` (integer, ≥1): Maximum segments per lightpath

#### **Output & Monitoring**
- `save_snapshots` (boolean): Save network snapshots during simulation
- `snapshot_step` (integer, ≥1): Frequency of snapshot saves
- `print_step` (integer, ≥1): Frequency of progress output
- `save_step` (integer, ≥1): Frequency of result saves
- `save_start_end_slots` (boolean): Save spectrum slot allocation details

### `[topology_settings]` - Network Configuration

#### **Network Topology**
- `network` (string): Network topology name
  - `NSFNet`: US National Science Foundation Network
  - `Pan-European`: European research network
  - `USbackbone60`: 60-node US backbone network

#### **Physical Parameters**
- `bw_per_slot` (number, >0): Bandwidth per spectrum slot in GHz (e.g., 12.5)
- `cores_per_link` (integer, ≥1): Number of fiber cores per link (e.g., 3)
- `multi_fiber` (boolean): Enable multiple fiber support

#### **Link Properties**
- `const_link_weight` (boolean): Use constant link weights
- `is_only_core_node` (boolean): Restrict connections to core nodes only

### `[spectrum_settings]` - Spectral Resources

- `c_band` (integer, ≥1): Number of spectrum slots in C-band (e.g., 320)
- `o_band` (integer, ≥0): Number of spectrum slots in O-band (optional)
- `e_band` (integer, ≥0): Number of spectrum slots in E-band (optional)  
- `s_band` (integer, ≥0): Number of spectrum slots in S-band (optional)
- `l_band` (integer, ≥0): Number of spectrum slots in L-band (optional)

### `[snr_settings]` - Signal Quality Parameters

#### **SNR Calculation**
- `snr_type` (string): SNR calculation method
  - `None`: Disable SNR calculations
  - `xt_calculation`: Enable cross-talk aware SNR

#### **Cross-Talk Parameters**
- `xt_type` (string): Cross-talk calculation type
  - `without_length`: Length-independent cross-talk
  - `with_length`: Length-dependent cross-talk
- `xt_noise` (boolean): Enable cross-talk noise modeling
- `requested_xt` (JSON object): Target cross-talk levels per modulation
  ```ini
  requested_xt = {"QPSK": -26.19, "16-QAM": -36.69, "64-QAM": -41.69}
  ```

#### **Physical Layer Parameters**
- `input_power` (number, >0): Optical input power in Watts (e.g., 0.001)
- `beta` (number, ≥0): Fiber parameter beta
- `theta` (number): Angle parameter for calculations
- `phi` (JSON object): Modulation-specific phi values
- `egn_model` (boolean): Enable EGN noise model
- `bi_directional` (boolean): Enable bidirectional transmission

### `[rl_settings]` - Reinforcement Learning

#### **Training Configuration**
- `is_training` (boolean): Enable training mode
- `n_trials` (integer, ≥1): Number of training trials
- `device` (string): Computation device (`cpu`, `cuda`, `mps`)
- `optimize_hyperparameters` (boolean): Enable hyperparameter optimization
- `optuna_trials` (integer, ≥1): Number of Optuna optimization trials

#### **Algorithm Selection**
- `path_algorithm` (string): Path selection algorithm
  - `dqn`: Deep Q-Network
  - `epsilon_greedy_bandit`: Epsilon-greedy bandit
  - `first_fit`: First-fit heuristic
- `core_algorithm` (string): Core selection algorithm  
- `spectrum_algorithm` (string): Spectrum allocation algorithm

#### **Learning Parameters**
- `gamma` (number, 0-1): Discount factor for future rewards
- `alpha_start` (number, >0): Initial learning rate
- `alpha_end` (number, >0): Final learning rate  
- `alpha_update` (string): Learning rate decay method
- `epsilon_start` (number, 0-1): Initial exploration rate
- `epsilon_end` (number, 0-1): Final exploration rate
- `epsilon_update` (string): Exploration decay method

#### **Network Architecture**
- `feature_extractor` (string): Feature extraction method
- `gnn_type` (string): Graph neural network type
- `layers` (integer, ≥1): Number of network layers
- `emb_dim` (integer, ≥1): Embedding dimension
- `heads` (integer, ≥1): Number of attention heads

#### **Reward Structure**
- `reward` (integer): Positive reward value for successful allocations
- `penalty` (integer): Negative penalty for blocking/failures
- `dynamic_reward` (boolean): Enable dynamic reward scaling

### `[ml_settings]` - Machine Learning

- `deploy_model` (boolean): Deploy trained ML model
- `ml_training` (boolean): Enable ML training mode
- `ml_model` (string): ML algorithm type (`decision_tree`, `random_forest`, etc.)
- `train_file_path` (string): Path to training data
- `test_size` (number, 0-1): Fraction of data for testing
- `output_train_data` (boolean): Save training data for analysis

### `[file_settings]` - Output Configuration

- `file_type` (string): Output file format (`json`, `csv`, `pickle`)

### Thread Sections `[s1], [s2], ...` - Multi-Threading

Thread-specific overrides for parallel simulation:
```ini
[s2]
k_paths = 3  # Override k_paths for thread 2
```

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