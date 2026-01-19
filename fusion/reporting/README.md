# Reporting Module

## Purpose
The reporting module provides utilities for formatting and reporting simulation results, separating presentation concerns from data collection in the FUSION simulation framework. Includes comprehensive support for survivability experiments with metrics collection, dataset logging, and multi-seed aggregation.

## Key Components
- `simulation_reporter.py`: Main SimulationReporter class for formatting and outputting simulation statistics
- `dataset_logger.py`: Offline RL dataset logging to JSONL format
- `aggregation.py`: Multi-seed result aggregation with statistical analysis
- `csv_export.py`: CSV export utilities for batch results
- `__init__.py`: Public API exports and module initialization

## Standard Reporting

### Basic Usage
```python
from fusion.reporting import SimulationReporter

# Initialize reporter
reporter = SimulationReporter(verbose=True)

# Report simulation progress
reporter.report_iteration_stats(
    iteration=0,
    max_iterations=100,
    erlang=50.0,
    blocking_list=[0.01, 0.02],
    print_flag=True
)

# Report simulation completion
reporter.report_simulation_complete(
    erlang=50.0,
    iterations_completed=100,
    confidence_interval=95.0
)
```

## Survivability Features

### Dataset Logging

Log offline RL training data in JSONL format:

```python
from fusion.reporting import DatasetLogger

# Initialize logger
logger = DatasetLogger(
    output_path='datasets/training_data.jsonl',
    engine_props={'seed': 42, 'dataset_logging': {'epsilon_mix': 0.1}}
)

# Log transitions
logger.log_transition(
    state={'src': 0, 'dst': 13, 'k_paths': [[0, 1, 2], [0, 3, 2]]},
    action=0,
    reward=1.0,
    next_state=None,
    action_mask=[True, False],
    meta={'request_id': 123, 't': 456.78}
)

logger.close()
```

**Dataset Format (JSONL):**
```json
{
  "t": 456.78,
  "seed": 42,
  "state": {"src": 0, "dst": 13, "k_paths": [[0, 1, 2], [0, 3, 2]]},
  "action": 0,
  "reward": 1.0,
  "next_state": null,
  "action_mask": [true, false],
  "meta": {"request_id": 123, "t": 456.78}
}
```

### Multi-Seed Aggregation

Aggregate results across multiple random seeds:

```python
from fusion.reporting import aggregate_seed_results

# Results from multiple seeds
results = [
    {'bp_overall': 0.05, 'recovery_time_mean_ms': 52.3, 'seed': 42},
    {'bp_overall': 0.06, 'recovery_time_mean_ms': 51.8, 'seed': 43},
    {'bp_overall': 0.055, 'recovery_time_mean_ms': 53.1, 'seed': 44}
]

# Aggregate with statistics
metrics = ['bp_overall', 'recovery_time_mean_ms']
aggregated = aggregate_seed_results(results, metrics)

print(aggregated['bp_overall'])
# Output: {'mean': 0.055, 'std': 0.004, 'ci95_lower': 0.050, 'ci95_upper': 0.060, 'n': 3}
```

### CSV Export

Export results to CSV for analysis:

```python
from fusion.reporting import export_results_to_csv

# Export batch results
results = [stats1.to_csv_row(), stats2.to_csv_row(), stats3.to_csv_row()]
export_results_to_csv(results, 'results/survivability_results.csv')
```

**CSV columns include:**
- Experiment parameters: `topology`, `load`, `failure_type`, `k_paths`, `policy`, `seed`
- Standard metrics: `bp_overall`
- Survivability metrics: `recovery_time_mean_ms`, `recovery_time_p95_ms`, `frag_proxy_mean`, `decision_time_mean_ms`
- Failure window metrics: `bp_window_fail_mean`, `bp_window_fail_p95`

## Configuration

Enable survivability reporting in configuration:

```ini
[dataset_logging]
log_offline_dataset = true
dataset_output_path = datasets/offline_data.jsonl
epsilon_mix = 0.1

[reporting]
export_csv = true
csv_output_path = results/survivability_results.csv
aggregate_seeds = true
seed_list = [42, 43, 44, 45, 46]
```

## Dependencies
### Internal Dependencies
- `fusion.utils.logging_config`: For logging configuration
- `fusion.core.metrics`: SimStats with survivability metrics

### External Dependencies
- `logging`: Standard library logging
- `statistics`: Standard library statistics functions
- `typing`: Type annotations
- `numpy`: Numerical operations and statistics
- `json`: JSONL serialization for dataset logging
- `csv`: CSV export functionality
- `pathlib`: File path handling
