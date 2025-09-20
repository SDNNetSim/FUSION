"""Configuration constants for the FUSION CLI."""

import os

# Project paths
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG_PATH: str = os.path.join(PROJECT_ROOT, "ini", "run_ini", "config.ini")

# Configuration file handling
CONFIG_DIR_PATH: str = os.path.join("ini", "run_ini")
REQUIRED_SECTION: str = "general_settings"

# Thread configuration patterns
DEFAULT_THREAD_NAME: str = "s1"
THREAD_SECTION_PATTERN: str = r"^s\d"

# Parameters that accept dictionary values in configuration
DICT_PARAM_OPTIONS_LIST: list[str] = ["request_distribution", "requested_xt", "phi"]
