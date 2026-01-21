"""Configuration constants for the FUSION CLI."""

import os

# Project paths
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG_PATH: str = os.path.join(PROJECT_ROOT, "ini", "run_ini", "config.ini")

# Configuration file handling
CONFIG_DIR_PATH: str = os.path.join("ini", "run_ini")
REQUIRED_SECTION: str = "general_settings"

# Process configuration patterns
# TODO (v6.1): Rename these variables to use "process" instead of "thread" for clarity.
# FUSION uses multiprocessing, not threading. Variable names should reflect this:
#   - DEFAULT_THREAD_NAME -> DEFAULT_PROCESS_NAME
#   - THREAD_SECTION_PATTERN -> PROCESS_SECTION_PATTERN
# This requires updating all usages across the codebase.
DEFAULT_THREAD_NAME: str = "s1"
THREAD_SECTION_PATTERN: str = r"^s\d"

# Parameters that accept dictionary values in configuration
DICT_PARAM_OPTIONS: list[str] = ["request_distribution", "requested_xt", "phi"]
