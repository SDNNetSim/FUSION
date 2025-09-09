"""Configuration constants for the FUSION CLI."""

import os

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'ini', 'run_ini', 'config.ini')

# Configuration constants
DEFAULT_THREAD_NAME = 's1'
THREAD_SECTION_PATTERN = r'^s\d'
DICT_PARAM_OPTIONS = ['request_distribution', 'requested_xt', 'phi']
REQUIRED_SECTION = 'general_settings'
CONFIG_DIR_PATH = os.path.join('ini', 'run_ini')
