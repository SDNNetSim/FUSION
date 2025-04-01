# unity_venv

This directory contains the `unity_venv` virtual environment, specifically set up for running projects on the Unity
cluster managed by UMass Amherst. The Unity cluster utilizes the SLURM job scheduler for managing computational tasks.

## Contents

- `unity_venv/`: Virtual environment directory for the Unity cluster.

## Purpose

The `unity_venv` environment is tailored for running jobs on the Unity cluster at UMass Amherst. It includes all
necessary dependencies and configurations required for seamless integration with the SLURM job scheduler.

## Usage

### Creating the Environment

The environment can be automatically created, using the bash script: `bash_scripts/make_unity_venv.sh`

### Activating the Environment

Before running any scripts or jobs, you need to activate the virtual environment. You can do this by navigating to
the `unity_venv` directory and using the following command:

```bash
source venvs/unity_venv/bin/activate
