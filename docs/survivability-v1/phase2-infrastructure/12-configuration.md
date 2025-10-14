# Phase 2: Core Infrastructure

## 12 - Configuration System Integration

**Section Reference**: 1.8 - Configuration System Integration

**Purpose**: Extend FUSION's configuration system to support survivability experiments with failure injection, protection mechanisms, RL policies, and dataset logging.

**Location**: `fusion/configs/`

**Estimated Effort**: 0.5 days

---

## Overview

This module extends FUSION's existing configuration infrastructure to support survivability features while maintaining backward compatibility with existing configurations. New configuration sections are added for:
- Failure injection settings
- Protection mechanisms (1+1)
- RL policy selection
- Dataset logging
- Recovery timing parameters

---

## 1. Configuration Template

### New Template: `survivability_experiment.ini`

**Location**: `fusion/configs/templates/survivability_experiment.ini`

```ini
# ============================================================================
# FUSION Survivability Experiment Configuration
# For offline RL + protection + failure testing
# ============================================================================

[general_settings]
# Load range for experiments
erlang_start = 100
erlang_stop = 300
erlang_step = 100

# Simulation parameters
max_iters = 5
num_requests = 2000
holding_time = 3.0
thread_num = s1

# Random seed (set to specific value for reproducibility)
seed = 42

[topology_settings]
# Network topology
network = NSFNet
cores_per_link = 7
bw_per_slot = 12.5

[spectrum_settings]
# C-band spectrum allocation
c_band = 320
guard_slots = 1
allocation_method = first_fit

[routing_settings]
# K-path candidate generation
route_method = k_shortest_path
k_paths = 4
path_ordering = hops
precompute_paths = true

[failure_settings]
# Failure type: none, link, node, srlg, geo
failure_type = none

# Failure timing
# -1 = uniform_mid (midpoint of simulation)
# Otherwise: specific arrival index
t_fail_arrival_index = -1
t_repair_after_arrivals = 1000

# Link failure (F1) parameters
failed_link_src = 0
failed_link_dst = 1

# Node failure (F2) parameters (not in v1, but structure prepared)
failed_node_id = 0

# SRLG failure (F3) parameters
# List of link tuples: [(0,1), (2,3)]
srlg_links = []

# Geographic failure (F4) parameters
geo_center_node = 5
geo_hop_radius = 2

[protection_settings]
# Protection mode: none, 1plus1
protection_mode = none

# 1+1 protection timing (milliseconds)
protection_switchover_ms = 50.0
restoration_latency_ms = 100.0

# Whether to revert to primary after repair
revert_to_primary = false

[offline_rl_settings]
# Policy selection: ksp_ff, one_plus_one, bc, iql
policy_type = ksp_ff

# Model paths (required if using bc or iql)
bc_model_path = models/bc_model.pt
iql_model_path = models/iql_model.pt

# Compute device: cpu, cuda, mps
device = cpu

# Fallback policy when all actions masked
fallback_policy = ksp_ff

[dataset_logging]
# Enable offline dataset logging for training
log_offline_dataset = false
dataset_output_path = datasets/offline_data.jsonl

# Epsilon-mix probability (0.0 = no exploration)
epsilon_mix = 0.1

[recovery_timing]
# Recovery time modeling parameters (milliseconds)
protection_switchover_ms = 50.0
restoration_latency_ms = 100.0

# Failure window size (number of arrivals after failure)
failure_window_size = 1000

[reporting]
# CSV export settings
export_csv = true
csv_output_path = results/survivability_results.csv

# Multi-seed aggregation
aggregate_seeds = true
seed_list = [42, 43, 44, 45, 46]

[snr_settings]
# SNR/QoT settings (not used in v1)
snr_type = None
```

---

## 2. Configuration Schema

### Schema Extension: `fusion/configs/schemas/survivability.json`

**Location**: `fusion/configs/schemas/survivability.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "FUSION Survivability Configuration Schema",
  "description": "Configuration schema for survivability experiments",
  "type": "object",
  "properties": {
    "failure_settings": {
      "type": "object",
      "description": "Failure injection configuration",
      "properties": {
        "failure_type": {
          "type": "string",
          "enum": ["none", "link", "node", "srlg", "geo"],
          "default": "none",
          "description": "Type of failure to inject"
        },
        "t_fail_arrival_index": {
          "type": "integer",
          "default": -1,
          "description": "Arrival index when failure occurs (-1 = midpoint)"
        },
        "t_repair_after_arrivals": {
          "type": "integer",
          "minimum": 1,
          "default": 1000,
          "description": "Number of arrivals until repair"
        },
        "failed_link_src": {
          "type": "integer",
          "minimum": 0,
          "description": "Source node of failed link (F1)"
        },
        "failed_link_dst": {
          "type": "integer",
          "minimum": 0,
          "description": "Destination node of failed link (F1)"
        },
        "failed_node_id": {
          "type": "integer",
          "minimum": 0,
          "description": "Node ID for node failure (F2)"
        },
        "srlg_links": {
          "type": "array",
          "items": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "integer"}
          },
          "default": [],
          "description": "List of link tuples in SRLG (F3)"
        },
        "geo_center_node": {
          "type": "integer",
          "minimum": 0,
          "description": "Center node for geographic failure (F4)"
        },
        "geo_hop_radius": {
          "type": "integer",
          "minimum": 1,
          "default": 2,
          "description": "Hop radius for geographic failure (F4)"
        }
      },
      "required": ["failure_type"]
    },
    "protection_settings": {
      "type": "object",
      "description": "Protection mechanism configuration",
      "properties": {
        "protection_mode": {
          "type": "string",
          "enum": ["none", "1plus1"],
          "default": "none",
          "description": "Protection mechanism to use"
        },
        "protection_switchover_ms": {
          "type": "number",
          "minimum": 0,
          "default": 50.0,
          "description": "1+1 switchover latency (ms)"
        },
        "restoration_latency_ms": {
          "type": "number",
          "minimum": 0,
          "default": 100.0,
          "description": "Restoration compute + signaling latency (ms)"
        },
        "revert_to_primary": {
          "type": "boolean",
          "default": false,
          "description": "Revert to primary path after repair"
        }
      }
    },
    "offline_rl_settings": {
      "type": "object",
      "description": "RL policy configuration",
      "properties": {
        "policy_type": {
          "type": "string",
          "enum": ["ksp_ff", "one_plus_one", "bc", "iql"],
          "default": "ksp_ff",
          "description": "Policy to use for path selection"
        },
        "bc_model_path": {
          "type": "string",
          "description": "Path to Behavior Cloning model (.pt)"
        },
        "iql_model_path": {
          "type": "string",
          "description": "Path to IQL model (.pt)"
        },
        "device": {
          "type": "string",
          "enum": ["cpu", "cuda", "mps"],
          "default": "cpu",
          "description": "Compute device for model inference"
        },
        "fallback_policy": {
          "type": "string",
          "enum": ["ksp_ff", "one_plus_one"],
          "default": "ksp_ff",
          "description": "Fallback when all actions masked"
        }
      }
    },
    "dataset_logging": {
      "type": "object",
      "description": "Offline dataset logging configuration",
      "properties": {
        "log_offline_dataset": {
          "type": "boolean",
          "default": false,
          "description": "Enable offline dataset logging"
        },
        "dataset_output_path": {
          "type": "string",
          "default": "datasets/offline_data.jsonl",
          "description": "Output path for JSONL dataset"
        },
        "epsilon_mix": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "default": 0.1,
          "description": "Probability of selecting second-best path"
        }
      }
    },
    "recovery_timing": {
      "type": "object",
      "description": "Recovery time modeling configuration",
      "properties": {
        "protection_switchover_ms": {
          "type": "number",
          "minimum": 0,
          "default": 50.0
        },
        "restoration_latency_ms": {
          "type": "number",
          "minimum": 0,
          "default": 100.0
        },
        "failure_window_size": {
          "type": "integer",
          "minimum": 1,
          "default": 1000,
          "description": "Number of arrivals for failure window BP"
        }
      }
    },
    "reporting": {
      "type": "object",
      "description": "Results reporting configuration",
      "properties": {
        "export_csv": {
          "type": "boolean",
          "default": true,
          "description": "Export results to CSV"
        },
        "csv_output_path": {
          "type": "string",
          "default": "results/survivability_results.csv",
          "description": "CSV output path"
        },
        "aggregate_seeds": {
          "type": "boolean",
          "default": true,
          "description": "Aggregate results across multiple seeds"
        },
        "seed_list": {
          "type": "array",
          "items": {"type": "integer"},
          "default": [42, 43, 44, 45, 46],
          "description": "List of seeds for multi-seed runs"
        }
      }
    }
  }
}
```

---

## 3. Configuration Validation

### Extension to `fusion/configs/validator.py`

```python
"""
Configuration validator for survivability settings.
"""

import json
from pathlib import Path
from typing import Any
import jsonschema


def validate_survivability_config(config: dict[str, Any]) -> None:
    """
    Validate survivability-specific configuration.

    :param config: Configuration dictionary
    :type config: dict[str, Any]
    :raises jsonschema.ValidationError: If validation fails
    :raises ValueError: If logical constraints violated

    Example:
        >>> config = load_config('survivability_experiment.ini')
        >>> validate_survivability_config(config)
    """
    # Load schema
    schema_path = Path(__file__).parent / 'schemas' / 'survivability.json'
    with open(schema_path) as f:
        schema = json.load(f)

    # Validate against schema
    jsonschema.validate(config, schema)

    # Additional logical validations
    _validate_failure_config(config)
    _validate_protection_config(config)
    _validate_rl_policy_config(config)


def _validate_failure_config(config: dict[str, Any]) -> None:
    """Validate failure settings."""
    failure_settings = config.get('failure_settings', {})
    failure_type = failure_settings.get('failure_type', 'none')

    # Type-specific validations
    if failure_type == 'link':
        if 'failed_link_src' not in failure_settings:
            raise ValueError("Link failure requires 'failed_link_src'")
        if 'failed_link_dst' not in failure_settings:
            raise ValueError("Link failure requires 'failed_link_dst'")

    elif failure_type == 'node':
        if 'failed_node_id' not in failure_settings:
            raise ValueError("Node failure requires 'failed_node_id'")

    elif failure_type == 'srlg':
        srlg_links = failure_settings.get('srlg_links', [])
        if not srlg_links:
            raise ValueError("SRLG failure requires non-empty 'srlg_links'")

    elif failure_type == 'geo':
        if 'geo_center_node' not in failure_settings:
            raise ValueError("Geographic failure requires 'geo_center_node'")
        if 'geo_hop_radius' not in failure_settings:
            raise ValueError("Geographic failure requires 'geo_hop_radius'")


def _validate_protection_config(config: dict[str, Any]) -> None:
    """Validate protection settings."""
    protection_settings = config.get('protection_settings', {})
    protection_mode = protection_settings.get('protection_mode', 'none')

    if protection_mode == '1plus1':
        # Ensure routing settings compatible
        routing_settings = config.get('routing_settings', {})
        if routing_settings.get('k_paths', 1) < 2:
            raise ValueError("1+1 protection requires k_paths >= 2")


def _validate_rl_policy_config(config: dict[str, Any]) -> None:
    """Validate RL policy settings."""
    rl_settings = config.get('offline_rl_settings', {})
    policy_type = rl_settings.get('policy_type', 'ksp_ff')

    # Validate model paths for RL policies
    if policy_type == 'bc':
        if 'bc_model_path' not in rl_settings:
            raise ValueError("BC policy requires 'bc_model_path'")

        model_path = Path(rl_settings['bc_model_path'])
        if not model_path.exists():
            raise ValueError(f"BC model not found: {model_path}")

    elif policy_type == 'iql':
        if 'iql_model_path' not in rl_settings:
            raise ValueError("IQL policy requires 'iql_model_path'")

        model_path = Path(rl_settings['iql_model_path'])
        if not model_path.exists():
            raise ValueError(f"IQL model not found: {model_path}")
```

---

## 4. CLI Integration

### Extension to `fusion/cli/run_sim.py`

```python
"""
CLI argument extensions for survivability experiments.
"""

import argparse


def add_survivability_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add survivability-specific CLI arguments.

    :param parser: Argument parser
    :type parser: argparse.ArgumentParser
    """
    # Failure settings
    failure_group = parser.add_argument_group('Failure Settings')
    failure_group.add_argument(
        '--failure_type',
        type=str,
        choices=['none', 'link', 'node', 'srlg', 'geo'],
        help='Type of failure to inject'
    )
    failure_group.add_argument(
        '--failed_link_src',
        type=int,
        help='Source node of failed link (F1)'
    )
    failure_group.add_argument(
        '--failed_link_dst',
        type=int,
        help='Destination node of failed link (F1)'
    )
    failure_group.add_argument(
        '--geo_center_node',
        type=int,
        help='Center node for geographic failure (F4)'
    )
    failure_group.add_argument(
        '--geo_hop_radius',
        type=int,
        help='Hop radius for geographic failure (F4)'
    )

    # Protection settings
    protection_group = parser.add_argument_group('Protection Settings')
    protection_group.add_argument(
        '--protection_mode',
        type=str,
        choices=['none', '1plus1'],
        help='Protection mechanism to use'
    )

    # RL policy settings
    rl_group = parser.add_argument_group('RL Policy Settings')
    rl_group.add_argument(
        '--policy_type',
        type=str,
        choices=['ksp_ff', 'one_plus_one', 'bc', 'iql'],
        help='Policy to use for path selection'
    )
    rl_group.add_argument(
        '--bc_model_path',
        type=str,
        help='Path to BC model (.pt)'
    )
    rl_group.add_argument(
        '--iql_model_path',
        type=str,
        help='Path to IQL model (.pt)'
    )

    # Dataset logging
    dataset_group = parser.add_argument_group('Dataset Logging')
    dataset_group.add_argument(
        '--log_offline_dataset',
        action='store_true',
        help='Enable offline dataset logging'
    )
    dataset_group.add_argument(
        '--dataset_output_path',
        type=str,
        help='Output path for JSONL dataset'
    )
    dataset_group.add_argument(
        '--epsilon_mix',
        type=float,
        help='Probability of selecting second-best path'
    )

    # Multi-seed runs
    seed_group = parser.add_argument_group('Multi-Seed Runs')
    seed_group.add_argument(
        '--seed_list',
        type=int,
        nargs='+',
        help='List of seeds for multi-seed runs'
    )
```

---

## 5. Example Configurations

### Example 1: Link Failure with KSP-FF

```ini
[failure_settings]
failure_type = link
failed_link_src = 3
failed_link_dst = 9
t_fail_arrival_index = -1
t_repair_after_arrivals = 1000

[offline_rl_settings]
policy_type = ksp_ff
```

### Example 2: Geographic Failure with 1+1 Protection

```ini
[failure_settings]
failure_type = geo
geo_center_node = 5
geo_hop_radius = 2

[protection_settings]
protection_mode = 1plus1
protection_switchover_ms = 50.0

[offline_rl_settings]
policy_type = one_plus_one
```

### Example 3: SRLG Failure with BC Policy

```ini
[failure_settings]
failure_type = srlg
srlg_links = [(0,1), (2,3), (5,6)]

[offline_rl_settings]
policy_type = bc
bc_model_path = models/bc_srlg_model.pt
fallback_policy = ksp_ff

[dataset_logging]
log_offline_dataset = false
```

---

## 6. Testing Requirements

### Unit Tests

```python
import pytest
from fusion.configs.validator import validate_survivability_config


def test_survivability_template_loads():
    """Test that template loads without validation errors."""
    from fusion.configs import load_config

    config = load_config('templates/survivability_experiment.ini')
    validate_survivability_config(config)


def test_failure_settings_validated():
    """Test that invalid failure_type rejected by schema."""
    config = {
        'failure_settings': {
            'failure_type': 'invalid_type'
        }
    }

    with pytest.raises(Exception):  # jsonschema.ValidationError
        validate_survivability_config(config)


def test_policy_type_validated():
    """Test that invalid policy_type rejected."""
    config = {
        'offline_rl_settings': {
            'policy_type': 'invalid_policy'
        }
    }

    with pytest.raises(Exception):
        validate_survivability_config(config)


def test_config_override_from_cli():
    """Test that CLI args override config file values."""
    from fusion.configs import load_config, merge_cli_args

    config = load_config('templates/survivability_experiment.ini')

    # CLI args
    cli_args = {
        'failure_type': 'geo',
        'geo_center_node': 7,
        'policy_type': 'bc'
    }

    # Merge
    merged = merge_cli_args(config, cli_args)

    # Verify overrides
    assert merged['failure_settings']['failure_type'] == 'geo'
    assert merged['failure_settings']['geo_center_node'] == 7
    assert merged['offline_rl_settings']['policy_type'] == 'bc'
```

---

## 7. Acceptance Criteria

- [x] `test_survivability_template_loads`: Template loads without validation errors
- [x] `test_failure_settings_validated`: Invalid failure_type rejected by schema
- [x] `test_policy_type_validated`: Invalid policy_type rejected
- [x] `test_config_override_from_cli`: CLI args override config file values
- [x] All survivability settings documented with descriptions
- [x] Backward compatibility maintained with existing configs

---

## Notes

- **Backward Compatibility**: Existing configurations without survivability sections work unchanged
- **Validation**: Schema validation happens at config load time, with clear error messages
- **CLI Priority**: CLI arguments override config file values
- **Extensibility**: New configuration sections can be added by extending the schema

---

**Related Documents**:
- [10-failure-module.md](10-failure-module.md) (Failure configuration)
- [20-protection.md](../phase3-protection/20-protection.md) (Protection configuration)
- [30-rl-policies.md](../phase4-rl-integration/30-rl-policies.md) (Policy configuration)
- [51-documentation.md](../phase6-quality/51-documentation.md) (Documentation standards)
