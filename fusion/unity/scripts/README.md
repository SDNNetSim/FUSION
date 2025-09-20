# Unity Cluster Infrastructure Scripts

This directory contains bash scripts specifically designed for Unity cluster infrastructure management at UMass Amherst. These scripts handle Unity-specific administration, monitoring, and environment setup tasks that are tied to the Unity cluster's specific configuration and user accounts.

## Unity Cluster Virtual Environment Management

### `make_unity_venv.sh`
**Purpose**: Create Python virtual environments specifically for Unity cluster jobs

**Usage**: `./make_unity_venv.sh <target_directory> <python_version>`

**Example**: `./make_unity_venv.sh /work/venvs/unity_env python3.11`

**Features**:
- Unity cluster-optimized virtual environment creation
- Integration with Unity cluster module system
- Python version validation against Unity cluster available versions
- Automatic cleanup of existing environments
- Enhanced error handling and status reporting

## Unity Cluster User and Resource Management

### `group_jobs.sh`
**Purpose**: Analyze Unity cluster resource usage for specific research group members

**Usage**: `./group_jobs.sh <start_date>`

**Example**: `./group_jobs.sh '7 days ago'`

**Features**:
- Reports resource usage for Unity cluster user group members
- Hardcoded Unity-specific user accounts (ryan_mccann_student_uml_edu, etc.)
- Covers both historical and current job consumption
- Unity-specific user account integration
- CPU, memory, and node usage analysis

### `priority.sh`
**Purpose**: Check Unity cluster SLURM job priorities and queue status

**Usage**: `./priority.sh <username> <partition>`

**Example**: `./priority.sh ryan_mccann_student_uml_edu gpu`

**Features**:
- Unity cluster partition-specific priority analysis
- User priority comparison within Unity cluster queues
- Queue position reporting for Unity cluster jobs
- Unity-specific partition knowledge (gpu, cpu, arm-gpu)

## Unity Cluster Information

- **Cluster Name**: Unity (UMass Amherst)
- **Scheduler**: SLURM
- **Partitions**: gpu, cpu, arm-gpu
- **User Group**: Specific UMass research group accounts
- **Module System**: Environment modules for Python and other software
- **Storage**: Shared work directories and user-specific spaces

## Requirements

- Unity cluster account and permissions
- SLURM environment (squeue, sacct, sbatch commands)
- Unity cluster module system access
- Appropriate permissions for user group resource monitoring
- For resource analysis: bc (basic calculator)

## Integration with FUSION Unity Module

These scripts support the Unity cluster infrastructure that the Python modules in the parent directory depend on:
- `make_manifest.py` - Creates job manifests for Unity cluster submission
- `submit_manifest.py` - Submits SLURM jobs to Unity cluster
- `fetch_results.py` - Retrieves results from Unity cluster storage

## Notes

These scripts are specifically designed for the Unity cluster infrastructure and contain:
- Unity-specific user account references
- Unity cluster partition and resource configurations
- Unity cluster module system integration
- Unity-specific storage and directory structures

**For general-purpose scripts** that can be used across different environments, see `tools/scripts/` which contains:
- Portable job execution scripts (run_rl_sim.sh, run_sim.sh)
- General monitoring tools (check_memory.sh)
- Generic administration utilities (kill_script.sh)
- Portable RL environment setup (register_rl_env.sh)