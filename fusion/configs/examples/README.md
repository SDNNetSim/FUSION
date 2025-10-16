# Example Configurations

This directory contains example configurations for common survivability experiment scenarios.

## Available Examples

### 1. Link Failure with KSP-FF Baseline (`link_failure_ksp_ff.ini`)

**Purpose**: Test basic link failure handling with KSP-FF routing policy.

**Use Case**: Baseline comparison for single link failures.

**Key Features**:
- Single link failure (F1)
- KSP-FF routing policy
- No protection mechanisms
- Measures impact of single link failure on blocking probability

**Run**:
```bash
python -m fusion.cli.run_sim --config_path fusion/configs/examples/link_failure_ksp_ff.ini
```

---

### 2. Geographic Failure with 1+1 Protection (`geo_failure_protection.ini`)

**Purpose**: Test network survivability under regional disasters with protection.

**Use Case**: Evaluate 1+1 disjoint path protection effectiveness.

**Key Features**:
- Geographic failure (F4) with hop-radius=2
- 1+1 disjoint path protection enabled
- Fast protection switchover (50ms)
- Automatic reversion to primary path after repair

**Run**:
```bash
python -m fusion.cli.run_sim --config_path fusion/configs/examples/geo_failure_protection.ini
```

---

### 3. RL Policy Evaluation (`rl_policy_eval.ini`)

**Purpose**: Evaluate offline RL policy (Behavior Cloning) performance.

**Use Case**: Compare RL policy against baseline under SRLG failures.

**Key Features**:
- SRLG failure (F3) - multiple correlated links
- Behavior Cloning (BC) policy
- Action masking with fallback to KSP-FF
- Dataset logging for analysis
- Multi-seed aggregation

**Run**:
```bash
python -m fusion.cli.run_sim --config_path fusion/configs/examples/rl_policy_eval.ini
```

**Note**: Requires trained BC model at `models/bc_model.pt`.

---

### 4. Dataset Generation (`dataset_generation.ini`)

**Purpose**: Generate offline RL training datasets with epsilon-greedy exploration.

**Use Case**: Collect high-quality data for training BC/IQL policies.

**Key Features**:
- Extended simulation (5000 requests)
- Higher epsilon for exploration (ε=0.2)
- Geographic failure for diverse scenarios
- Comprehensive dataset logging

**Run**:
```bash
python -m fusion.cli.run_sim --config_path fusion/configs/examples/dataset_generation.ini
```

**Output**: JSONL dataset at `datasets/training_dataset.jsonl`

---

## Configuration Parameters Quick Reference

### Failure Types

| Parameter | Values | Description |
|-----------|--------|-------------|
| `failure_type` | `none`, `link`, `node`, `srlg`, `geo` | Type of failure to inject |
| `failed_link_src` | Node ID | Source node of failed link (F1) |
| `failed_link_dst` | Node ID | Destination node of failed link (F1) |
| `srlg_links` | List of tuples | Links in SRLG (F3) |
| `geo_center_node` | Node ID | Center of geographic failure (F4) |
| `geo_hop_radius` | Integer | Radius in hops (F4) |

### Protection Settings

| Parameter | Values | Description |
|-----------|--------|-------------|
| `protection_mode` | `none`, `1plus1` | Protection mechanism |
| `protection_switchover_ms` | Float | Switchover latency (ms) |
| `restoration_latency_ms` | Float | Re-routing latency (ms) |
| `revert_to_primary` | Boolean | Revert after repair |

### Policy Settings

| Parameter | Values | Description |
|-----------|--------|-------------|
| `policy_type` | `ksp_ff`, `one_plus_one`, `bc`, `iql` | Path selection policy |
| `bc_model_path` | Path | BC model file (if using BC) |
| `iql_model_path` | Path | IQL model file (if using IQL) |
| `fallback_policy` | `ksp_ff`, `one_plus_one` | Fallback when all actions masked |
| `device` | `cpu`, `cuda`, `mps` | Compute device for RL inference |

### Dataset Logging

| Parameter | Values | Description |
|-----------|--------|-------------|
| `log_offline_dataset` | Boolean | Enable dataset logging |
| `dataset_output_path` | Path | Output JSONL file path |
| `epsilon_mix` | Float [0,1] | Exploration probability |

## Customization Guide

### Running with Different Seeds

```bash
python -m fusion.cli.run_sim \
  --config_path fusion/configs/examples/link_failure_ksp_ff.ini \
  --seed 43
```

### Changing Failure Parameters

```bash
python -m fusion.cli.run_sim \
  --config_path fusion/configs/examples/geo_failure_protection.ini \
  --geo_center_node 7 \
  --geo_hop_radius 3
```

### Using Different Topologies

Edit the configuration file:
```ini
[topology_settings]
network = USNET  # or COST239, DT, etc.
```

## Batch Experiments

Run multiple configurations with different seeds for statistical significance:

```bash
#!/bin/bash
for seed in 42 43 44 45 46; do
  python -m fusion.cli.run_sim \
    --config_path fusion/configs/examples/rl_policy_eval.ini \
    --seed $seed
done

# Aggregate results
python -m fusion.analysis.aggregate_results --input results/*.csv --output aggregated.csv
```

## Output Files

All examples produce:
- **CSV Results**: Comprehensive metrics in `results/*.csv`
- **Log Files**: Detailed simulation logs
- **Dataset Files**: JSONL format (if enabled) in `datasets/*.jsonl`

## Further Reading

- [Survivability v1 Documentation](../../../docs/survivability-v1/README.md)
- [Main Configuration Template](../templates/survivability_experiment.ini)
- [Failures Module](../../../fusion/modules/failures/README.md)
- [RL Policies Module](../../../fusion/modules/rl/policies/README.md)
