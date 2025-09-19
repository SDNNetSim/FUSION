# General-Purpose Scripts

This directory contains general-purpose scripts that can be used across different computing environments. While these scripts are primarily used on the Unity cluster (our main cluster), they are designed to be portable and reusable.

## Job Execution Scripts

### `run_rl_sim.sh`
**Purpose**: Execute reinforcement learning simulations via SLURM job arrays

**Usage**: Called automatically by SLURM array jobs

**Features**:
- SLURM array job execution for RL workloads
- Manifest-based parameter processing
- Virtual environment management
- RL environment registration
- Multi-partition support (arm-gpu, cpu, gpu)

### `run_sim.sh`
**Purpose**: Execute general simulations via SLURM job arrays

**Usage**: Called automatically by SLURM array jobs

**Features**:
- SLURM array job execution for general simulation workloads
- Manifest-based parameter processing
- Virtual environment management
- Result tracking and indexing

### `register_rl_env.sh`
**Purpose**: Register custom reinforcement learning environments

**Usage**: `./register_rl_env.sh <algorithm> <environment_name>`

**Example**: `./register_rl_env.sh PPO SimEnv`

**Features**:
- Registers environments in Stable Baselines 3 and Gymnasium
- Supports multiple RL algorithms
- Virtual environment integration

## Monitoring and Administration Scripts

### `check_memory.sh`
**Purpose**: Monitor memory usage and count running instances of processes

**Usage**: `./check_memory.sh <script_path> <process_name>`

**Example**: `./check_memory.sh /path/to/script.py simulation`

**Features**:
- Calculates total memory usage of matching processes
- Counts number of running instances
- Supports full command line matching

### `kill_script.sh`
**Purpose**: Terminate all running instances of a script

**Usage**: `./kill_script.sh <script_path>`

**Example**: `./kill_script.sh /path/to/script.py`

**Features**:
- Validates script path before termination
- Uses process name matching for safe termination
- Provides confirmation of termination

## Requirements

- SLURM workload manager (for job execution scripts)
- Standard Unix utilities (ps, awk, grep, pkill)
- Python 3.11+ (for simulation scripts)
- Stable Baselines 3 and Gymnasium (for RL scripts)

## Notes

These scripts are designed to be general-purpose and portable across different cluster environments. While they are primarily used on the Unity cluster at UMass Amherst, they can be adapted for use on other SLURM-based computing systems.

**For Unity cluster-specific infrastructure scripts**, see `fusion/unity/scripts/` which contains:
- Unity-specific user group management
- Unity-specific SLURM queue and priority tools
- Unity cluster virtual environment setup