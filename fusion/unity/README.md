# Unity Module

## Purpose
The Unity module provides cluster management functionality for FUSION simulations, including job submission, manifest generation, and result synchronization from remote cluster storage.

## Key Components
- `make_manifest.py`: Generate job manifests from specification files
- `submit_manifest.py`: Submit manifests as SLURM array jobs to the cluster
- `fetch_results.py`: Synchronize simulation results from remote storage
- `constants.py`: Module-wide constants for paths and configurations
- `errors.py`: Custom exceptions for Unity operations
- `scripts/`: Shell scripts for cluster operations

## Usage
```python
# Generate a manifest from a specification
python -m fusion.unity.make_manifest spec_name

# Submit jobs to the cluster
python -m fusion.unity.submit_manifest experiments/0123/1456/network1 run_sim.sh

# Fetch results from cluster
python -m fusion.unity.fetch_results
```

## Dependencies
Internal dependencies:
- `fusion.configs.schema` (for simulation parameter types)
- `fusion.utils.logging_config` (for logging setup)

External dependencies:
- `pyyaml` (optional, for YAML spec files)
- Standard library: `pathlib`, `subprocess`, `csv`, `json`
