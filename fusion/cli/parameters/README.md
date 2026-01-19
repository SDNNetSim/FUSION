# CLI Parameters Module

## Purpose

The CLI Parameters module provides a centralized, modular system for managing command-line arguments across all FUSION commands. It implements a registry-based architecture that organizes arguments into logical groups, reduces duplication, and ensures consistent argument handling throughout the application.

**IMPORTANT: CLI arguments override INI configuration values for ALL processes in a simulation.** When you specify a CLI argument, it will be applied to every process (s1, s2, etc.) regardless of what is defined in the INI file.

## Key Components

### Core Files
- `registry.py`: Central ArgumentRegistry coordinator for managing argument groups
- `shared.py`: Common arguments used across multiple commands (config, debug, output, plotting)
- `__init__.py`: Module exports and public API with comprehensive documentation

### Specialized Argument Groups
- `analysis.py`: Statistics collection, plotting, export, comparison, and filtering arguments
- `network.py`: Network topology, link configuration, node settings, and spectrum band arguments
- `policy.py`: SDN Orchestrator policy selection, heuristic parameters, and protection settings (v6.0.0)
- `routing.py`: Legacy routing algorithms, spectrum allocation, and SDN configuration arguments
- `snr.py`: Signal-to-noise ratio calculations and modulation format selection arguments
- `traffic.py`: Traffic generation, Erlang load configuration, and simulation control arguments
- `training.py`: Reinforcement learning (RL), supervised learning (SL), feature extraction, and optimization arguments

### Legacy Compatibility
- `simulation.py`: Compatibility module that imports and combines arguments from other modules

## Architecture

The module follows a **registry pattern** with modular argument groups:

```
CLI Parameters Module Architecture
├── ArgumentRegistry (registry.py)
│   ├── Central coordinator for all argument groups
│   └── Manages argument registration and organization
├── Shared Arguments (shared.py)
│   ├── Core configuration (config_path, run_id)
│   ├── Debug settings (verbose, debug)
│   ├── Output control (save_results, output_dir)
│   └── Plot formatting (plot_format)
├── Specialized Groups
│   ├── Analysis (analysis.py)
│   ├── Network (network.py)
│   ├── Policy (policy.py) - v6.0.0 Orchestrator
│   ├── Routing (routing.py) - Legacy
│   ├── SNR (snr.py)
│   ├── Traffic (traffic.py)
│   └── Training (training.py)
└── Legacy Compatibility
    └── Simulation (simulation.py) - imports others
```

## Routing Systems: Legacy vs Orchestrator (v6.0.0)

FUSION has two distinct path selection systems that coexist. Understanding the difference is important for choosing the right CLI arguments.

### Legacy Routing System (`fusion/modules/routing/`)

The legacy system tightly couples path computation and spectrum allocation in a single step.

**CLI Arguments**: `--route_method`, `--k_paths`, `--allocation_method`

```
Request → KShortestPath.route() → [computes paths + assigns spectrum] → Result
```

- **Classes**: `KShortestPath`, `LeastCongestedRouting`, `CongestionAwareRouting`, `XTAwareRouting`
- **Interface**: `AbstractRoutingAlgorithm`
- **Used by**: Legacy simulation pipeline (`run_simulation_pipeline`)

Example:
```bash
fusion-sim --config_path config.ini --route_method k_shortest_path --k_paths 3
```

### SDN Orchestrator Policy System (`fusion/policies/`) - v6.0.0

The new orchestrator decouples the process into three stages:

```
Request → Orchestrator computes K paths → Policy selects one → Orchestrator allocates spectrum → Result
```

**CLI Arguments**: `--policy-type`, `--policy-name`, `--policy-k-paths`, `--protection-enabled`

1. **Path computation**: Orchestrator computes K candidate paths
2. **Path selection**: ControlPolicy (heuristic/SL/RL) decides which path to use
3. **Spectrum allocation**: Orchestrator handles assignment

- **Classes**: `FirstFeasiblePolicy`, `ShortestFeasiblePolicy`, `LeastCongestedPolicy`, `LoadBalancedPolicy`
- **Interface**: `ControlPolicy` protocol
- **Used by**: SDN Orchestrator pipeline (`SDNOrchestrator`)

Example:
```bash
fusion-sim --config_path config.ini --policy-type heuristic --policy-name least_congested
```

### Key Differences

| Aspect | Legacy (`--route_method`) | Orchestrator (`--policy-*`) |
|--------|---------------------------|----------------------------|
| Architecture | Tightly coupled | Decoupled (3 stages) |
| Path selection | Baked into routing algorithm | Separate ControlPolicy |
| SL/RL support | Limited | Native support |
| Protection | Separate module | Integrated (`--protection-enabled`) |

**Important**: `LeastCongestedRouting` (legacy) and `LeastCongestedPolicy` (orchestrator) are **different**:
- Legacy: Computes paths with congestion awareness baked in
- Orchestrator: Selects from pre-computed paths based on congestion metric

### Which System to Use?

- **Legacy system**: For backward compatibility with existing configurations and scripts
- **Orchestrator system**: For new experiments, especially those involving SL/RL policies or 1+1 protection

The two systems do not interfere with each other. The pipeline you run determines which arguments are used.

## Usage

### Basic Usage
```python
from fusion.cli.parameters import (
    args_registry,
    add_config_args,
    add_network_args,
    add_routing_args
)

# Create parser and add argument groups
parser = argparse.ArgumentParser()
add_config_args(parser)        # Core configuration
add_network_args(parser)       # Network topology
add_routing_args(parser)       # Routing algorithms
```

### Registry-Based Usage
```python
from fusion.cli.parameters import args_registry, ArgumentRegistry

# Use the global registry
registry = args_registry
registry.register_argument_group("network", add_network_args)

# Or create a custom registry
custom_registry = ArgumentRegistry()
custom_registry.register_argument_group("custom", my_custom_args)
```

### Command-Specific Usage
```python
# For simulation commands
from fusion.cli.parameters import register_run_sim_args

# Register complete simulation argument set
subparsers = parser.add_subparsers()
register_run_sim_args(subparsers)
```

## Complete Parameter Reference

### Required Arguments
| Parameter | Type | Description | Source |
|-----------|------|-------------|--------|
| `--config_path` | str | Path to INI configuration file | shared.py |
| `--run_id` | str | Unique identifier for this simulation run | shared.py |

### Core Configuration Arguments (shared.py)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--verbose` | bool | False | Enable verbose output |
| `--debug` | bool | False | Enable debug mode |
| `--output_dir` | str | None | Directory to save output files |
| `--save_results` | bool | False | Save simulation results to file |
| `--save_snapshots` | bool | False | Save simulation snapshots |
| `--snapshot_step` | int | None | Step interval for saving snapshots |
| `--print_step` | int | None | Step interval for printing progress |
| `--save_step` | int | None | Step interval for saving results |
| `--save_start_end_slots` | bool | False | Save start and end slots information |
| `--file_type` | str | None | Output file format type |
| `--filter_mods` | bool | False | Enable modulation filtering |
| `--plot_format` | str | "png" | Output format for plots (png/pdf/svg/eps) |

### Network Configuration Arguments (network.py)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--network` | str | None | Network topology name (e.g., 'NSFNet', 'USbackbone60') |
| `--cores_per_link` | int | 1 | Number of cores per fiber link |
| `--bw_per_slot` | float | None | Bandwidth per spectral slot in GHz |
| `--const_link_weight` | bool | False | Use constant link weights for routing |
| `--bi_directional` | bool | False | Enable bidirectional links |
| `--multi_fiber` | bool | False | Enable multi-fiber links |
| `--is_only_core_node` | bool | False | Only allow core nodes to send requests |
| `--c_band` | int | 96 | Number of spectral slots in C-band (1530-1565nm) |
| `--l_band` | int | 0 | Number of spectral slots in L-band (1565-1625nm) |
| `--s_band` | int | 0 | Number of spectral slots in S-band (1460-1530nm) |
| `--e_band` | int | 0 | Number of spectral slots in E-band (1360-1460nm) |
| `--o_band` | int | 0 | Number of spectral slots in O-band (1260-1360nm) |

### Routing Configuration Arguments (routing.py)
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--route_method` | str | None | Routing algorithm method | shortest_path, k_shortest_path |
| `--k_paths` | int | 3 | Number of candidate paths for k-shortest path routing | - |
| `--allocation_method` | str | None | Spectrum allocation method | first_fit, best_fit, last_fit |
| `--guard_slots` | int | 1 | Number of guard slots between allocations | - |
| `--spectrum_priority` | str | None | Priority order for multi-band allocation | BSC, CSB |
| `--dynamic_lps` | bool | False | Enable SDN dynamic lightpath switching | - |
| `--single_core` | bool | False | Force single-core allocation per request | - |

### SNR and Modulation Arguments (snr.py)
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--mod_assumption` | str | None | Modulation format selection strategy | fixed, adaptive, precalculated, DEFAULT, CUSTOM, slicing_dissertation, ARASH_MOD_ASSUMPTIONS, ARASH, SNR_ASSUMPTIONS, XTAR_ASSUMPTIONS |
| `--mod_assumption_path` | str | None | Path to modulation format configuration file | - |
| `--snr_type` | str | None | SNR calculation method | linear, nonlinear, egn |
| `--input_power` | float | 1e-3 | Input power in Watts | - |
| `--egn_model` | bool | False | Enable Enhanced Gaussian Noise (EGN) model | - |

### Traffic Generation Arguments (traffic.py)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--erlang_start` | float | None | Starting Erlang load |
| `--erlang_stop` | float | None | Ending Erlang load |
| `--erlang_step` | float | None | Erlang load increment |
| `--holding_time` | float | None | Average holding time for requests |
| `--num_requests` | int | None | Total number of requests to generate |
| `--max_iters` | int | 3 | Maximum number of simulation iterations |
| `--thread_erlangs` | bool | False | Enable multi-threaded Erlang processing |

### Training Arguments (training.py)

#### Reinforcement Learning (RL) Configuration
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--path_algorithm` | str | None | Path selection RL algorithm | dqn, ppo, a2c, q_learning, bandits, epsilon_greedy_bandit |
| `--core_algorithm` | str | None | Core selection RL algorithm | dqn, ppo, a2c, q_learning, bandits, epsilon_greedy_bandit, first_fit |
| `--spectrum_algorithm` | str | None | Spectrum allocation RL algorithm | dqn, ppo, a2c, q_learning, bandits, epsilon_greedy_bandit, first_fit |
| `--path_model` | str | None | Path to pre-trained path selection model | - |
| `--core_model` | str | None | Path to pre-trained core selection model | - |
| `--spectrum_model` | str | None | Path to pre-trained spectrum allocation model | - |
| `--is_training` | bool | False | Enable training mode (vs. inference mode) | - |
| `--learn_rate` | float | 0.001 | Learning rate for RL algorithms | - |
| `--gamma` | float | 0.99 | Discount factor for future rewards | - |
| `--epsilon_start` | float | 1.0 | Initial epsilon value for epsilon-greedy exploration | - |
| `--epsilon_end` | float | 0.01 | Final epsilon value for epsilon-greedy exploration | - |
| `--epsilon_update` | str | "linear" | Epsilon decay strategy | linear, exponential, step, linear_decay, exp_decay |
| `--reward` | float | 1.0 | Reward value for successful actions | - |
| `--penalty` | float | -1.0 | Penalty value for unsuccessful actions | - |
| `--dynamic_reward` | bool | False | Enable dynamic reward calculation | - |

#### Feature Extraction Configuration
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--feature_extractor` | str | None | Feature extraction method | graphormer, path_gnn |
| `--gnn_type` | str | None | Graph Neural Network architecture type | gcn, gat, sage, graphconv |
| `--layers` | int | 3 | Number of layers in neural network | - |
| `--emb_dim` | int | 64 | Embedding dimension for neural networks | - |
| `--heads` | int | 8 | Number of attention heads (for attention-based models) | - |
| `--obs_space` | str | None | Observation space representation | graph, vector, matrix, hybrid |

#### Supervised Learning (SL) Configuration
<!-- TODO (v6.1.0): Rename ml_* arguments to sl_* for consistency (e.g., ml_training -> sl_training, ml_model -> sl_model) -->
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--ml_training` | bool | False | Enable SL training mode | - |
| `--ml_model` | str | None | Supervised learning model type | random_forest, svm, linear_regression, neural_network, decision_tree |
| `--train_file_path` | str | None | Path to training data file | - |
| `--test_size` | float | 0.2 | Fraction of data to use for testing (0.0-1.0) | - |
| `--output_train_data` | bool | False | Save training data to file | - |
| `--deploy_model` | bool | False | Deploy trained model for inference | - |

#### Optimization Configuration
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--optimize` | bool | False | Enable hyperparameter optimization | - |
| `--optimize_hyperparameters` | bool | False | Enable automated hyperparameter tuning | - |
| `--optuna_trials` | int | 100 | Number of optimization trials for Optuna | - |
| `--n_trials` | int | 10 | Number of trials for grid search or random search | - |
| `--device` | str | "auto" | Computing device for training (cpu/gpu) | cpu, cuda, mps, auto |

### Analysis and Plotting Arguments (analysis.py)

#### Statistics Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save_snapshots` | bool | False | Save simulation state snapshots during execution |
| `--snapshot_step` | int | 100 | Number of requests between snapshots |
| `--save_step` | int | 1000 | Number of requests between saving results |
| `--print_step` | int | 1000 | Number of requests between progress updates |
| `--save_start_end_slots` | bool | False | Save detailed slot allocation information |

#### Plotting Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--plot_results` | bool | False | Generate plots from simulation results |
| `--plot_dpi` | int | 300 | Resolution (DPI) for generated plots |
| `--show_plots` | bool | False | Display plots interactively (in addition to saving) |

#### Export Configuration
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--file_type` | str | "json" | Default file format for data export | json, csv, excel, tsv |

#### Filtering Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--filter_mods` | bool | False | Filter results by modulation format |
| `--min_erlang` | float | None | Minimum Erlang load to include in analysis |
| `--max_erlang` | float | None | Maximum Erlang load to include in analysis |

#### Comparison Configuration
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--compare_runs` | str (nargs="+") | None | List of run IDs to compare | - |
| `--baseline_run` | str | None | Run ID to use as baseline for comparison | - |
| `--metrics` | str (nargs="+") | None | Metrics to include in comparison | blocking_probability, path_length, execution_time, resource_utilization |
| `--significance_test` | str | None | Statistical test for comparing results | t_test, wilcoxon, mann_whitney |

### Orchestrator Policy Arguments (policy.py) - v6.0.0

These arguments configure the SDN Orchestrator's path selection system. See "Routing Systems" section above for when to use these vs legacy routing arguments.

#### Policy Configuration
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--policy-type` | str | heuristic | Type of policy for path selection | heuristic, sl, rl |
| `--policy-name` | str | first_feasible | Heuristic policy name | first_feasible, shortest, shortest_feasible, least_congested, random, random_feasible, load_balanced |
| `--policy-model-path` | str | None | Path to SL/RL model file for sl/rl policy types | - |
| `--policy-fallback` | str | first_feasible | Fallback policy name when SL/RL fails | - |
| `--policy-k-paths` | int | 3 | Number of candidate paths for policy | - |
| `--policy-seed` | int | None | Random seed for policy (affects random policies) | - |
| `--policy-algorithm` | str | None | RL algorithm name for rl policy type | PPO, MaskablePPO, DQN, A2C |
| `--policy-device` | str | cpu | Device for SL/RL inference | cpu, cuda, auto |

#### Heuristic Policy Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--heuristic-alpha` | float | 0.5 | Alpha parameter for LoadBalancedPolicy (0.0=congestion, 1.0=length) |
| `--heuristic-seed` | int | None | Random seed for RandomFeasiblePolicy |

#### Protection Configuration (Orchestrator)
| Parameter | Type | Default | Description | Choices |
|-----------|------|---------|-------------|---------|
| `--protection-enabled` | bool | False | Enable 1+1 protection | - |
| `--disjointness-type` | str | link | Type of path disjointness | link, node |
| `--protection-switchover-ms` | float | 50.0 | Protection switchover latency in milliseconds | - |
| `--restoration-latency-ms` | float | 100.0 | Restoration latency in milliseconds | - |

## Dependencies

### Internal Dependencies
- `argparse`: Python standard library for argument parsing
- No other internal FUSION dependencies (self-contained module)

### External Dependencies
- **None**: This module only uses Python standard library components

## Configuration

### Basic Configuration Example
```python
import argparse
from fusion.cli.parameters import add_config_args, add_network_args

parser = argparse.ArgumentParser(description='FUSION Simulation')
add_config_args(parser)     # Required: --config_path, --run_id
add_network_args(parser)    # Optional: network topology settings

args = parser.parse_args()
```

### Complete Simulation Configuration
```python
from fusion.cli.parameters import register_run_sim_args

parser = argparse.ArgumentParser(description='FUSION')
subparsers = parser.add_subparsers(dest='command')

# Register complete run_sim command with all argument groups
register_run_sim_args(subparsers)

args = parser.parse_args()
```

## API Reference

### Main Functions

#### Core Registration Functions
- `register_run_sim_args(subparsers)`: Register complete run_sim subcommand parser
- `add_run_sim_args(parser)`: Add consolidated run_sim arguments to parser
- `add_simulation_args(parser)`: Add comprehensive simulation arguments (compatibility)

#### Shared Arguments
- `add_config_args(parser)`: Add core configuration arguments (config_path, run_id)
- `add_debug_args(parser)`: Add debug and verbose output arguments
- `add_output_args(parser)`: Add output file and save configuration arguments
- `add_plot_format_args(parser)`: Add plot formatting arguments

#### Specialized Argument Groups
- `add_network_args(parser)`: Add network topology and physical layer arguments
- `add_all_network_args(parser)`: Add all network-related argument groups
- `add_routing_args(parser)`: Add legacy routing algorithm arguments
- `add_all_routing_args(parser)`: Add all legacy routing-related argument groups
- `add_policy_args(parser)`: Add orchestrator policy selection arguments (v6.0.0)
- `add_all_policy_args(parser)`: Add all orchestrator policy argument groups (v6.0.0)
- `add_traffic_args(parser)`: Add traffic generation and simulation control arguments
- `add_snr_args(parser)`: Add SNR calculation and modulation arguments
- `add_all_training_args(parser)`: Add all training-related argument groups
- `add_all_analysis_args(parser)`: Add all analysis-related argument groups

#### Registry System
- `ArgumentRegistry`: Central coordinator class for argument management
- `args_registry`: Global registry instance for shared argument management

## Examples

### Example 1: Basic Network Simulation Setup
```python
import argparse
from fusion.cli.parameters import (
    add_config_args,
    add_network_args,
    add_routing_args,
    add_traffic_args
)

parser = argparse.ArgumentParser(description='Network Simulation')

# Add required core arguments
add_config_args(parser)

# Add simulation-specific arguments
add_network_args(parser)    # Network topology
add_routing_args(parser)    # Routing algorithms
add_traffic_args(parser)    # Traffic generation

args = parser.parse_args()

# Example usage:
# python sim.py --config_path config.ini --run_id sim001
#               --network NSFNet --route_method shortest_path
#               --erlang_start 50 --erlang_stop 200
```

### Example 2: SL Training Configuration
```python
import argparse
from fusion.cli.parameters import (
    add_config_args,
    add_network_args,
    add_all_training_args
)

parser = argparse.ArgumentParser(description='SL Training')
add_config_args(parser)
add_network_args(parser)
add_all_training_args(parser)

args = parser.parse_args()

# Example usage:
# python train.py --config_path config.ini --run_id train001
#                 --network USbackbone60 --ml_training
#                 --ml_model random_forest --test_size 0.3
```

### Example 3: SDN Orchestrator with Policy (v6.0.0)
```python
import argparse
from fusion.cli.parameters import (
    add_config_args,
    add_network_args,
    add_all_policy_args
)

parser = argparse.ArgumentParser(description='Orchestrator Simulation')
add_config_args(parser)
add_network_args(parser)
add_all_policy_args(parser)

args = parser.parse_args()

# Example usage with heuristic policy:
# python sim.py --config_path config.ini --run_id orch001
#               --network NSFNet --policy-type heuristic
#               --policy-name load_balanced --heuristic-alpha 0.7

# Example usage with RL policy and protection:
# python sim.py --config_path config.ini --run_id orch002
#               --network USbackbone60 --policy-type rl
#               --policy-model-path models/ppo_policy.zip
#               --protection-enabled --disjointness-type node
```

### Example 4: Analysis and Plotting
```python
import argparse
from fusion.cli.parameters import (
    add_config_args,
    add_all_analysis_args
)

parser = argparse.ArgumentParser(description='Results Analysis')
add_config_args(parser)
add_all_analysis_args(parser)

args = parser.parse_args()

# Example usage:
# python analyze.py --config_path config.ini --run_id analysis001
#                   --plot_results --plot_format pdf --plot_dpi 600
#                   --compare_runs sim001 sim002 --metrics blocking_probability
```

## Notes

### Design Decisions
- **Registry Pattern**: Enables modular argument organization and reduces code duplication
- **Argument Groups**: Logical separation improves CLI usability and code maintainability
- **Backward Compatibility**: Legacy modules (simulation.py) maintained for existing code
- **Type Annotations**: Full type hints throughout for better IDE support and code clarity
- **Comprehensive Documentation**: Every parameter documented with type, default, and description

### Module Organization Principles
- **Single Responsibility**: Each file handles one logical argument category
- **Shared Components**: Common arguments centralized in shared.py
- **Extensibility**: Easy to add new argument groups without modifying existing code
- **Import Flexibility**: Support both individual function imports and registry-based usage

### Integration Notes
- **CLI Commands**: Used by all FUSION CLI commands (run_sim, run_train, etc.)
- **Configuration System**: Integrates with INI-based configuration management
- **Logging System**: Debug arguments integrate with FUSION logging infrastructure
- **Output Management**: File and directory arguments coordinate with result saving systems
