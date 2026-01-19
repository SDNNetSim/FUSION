"""
Constants for the Unity cluster management module.

This module defines all constants used across the Unity module for managing
cluster simulations, job submissions, and result fetching.
"""

# Directory and path constants
UNITY_BASE_DIR = "unity"
EXPERIMENTS_DIR = "experiments"
BASH_SCRIPTS_DIR = "bash_scripts"
CLUSTER_RESULTS_DIR = "cluster_results"
TEMP_CONFIG_DIR = ".tmp_config"
CONFIG_FILE_PATH = "configs/config.yml"

# File naming patterns
MANIFEST_FILE = "manifest.csv"
MANIFEST_META_FILE = "manifest_meta.json"
RUNS_INDEX_FILE = "runs_index.json"
SLURM_OUTPUT_PATTERN = "slurm_%A_%a.out"
SIM_INPUT_PATTERN = "sim_input_s*.json"

# Resource keys for SLURM job submission
RESOURCE_KEYS = {"partition", "time", "mem", "cpus", "gpus", "nodes"}

# Path algorithm types
RL_ALGORITHMS = {
    "ppo",
    "qr_dqn",
    "a2c",
    "dqn",
    "epsilon_greedy_bandit",
    "ucb_bandit",
    "q_learning",
}

# Boolean string values
BOOL_TRUE_VALUES = {"true", "yes", "1"}

# Remote synchronization settings
RSYNC_DEFAULT_OPTIONS = ["-avP", "--compress"]
SYNC_DELAY_SECONDS = 3.0

# Path segment counts
OUTPUT_TO_INPUT_SEGMENTS = 4  # Number of path segments to extract for sync
