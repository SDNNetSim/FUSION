"""
Constants for the gymnasium environment module.

This module defines configuration constants and default values used
by the simulation environment implementation.
"""

# Configuration keys
DEFAULT_SIMULATION_KEY: str = 's1'
DEFAULT_SAVE_SIMULATION: bool = False

# Supported spectral bands
SUPPORTED_SPECTRAL_BANDS: list = ['c']  # Currently only C-band supported

# Configuration dictionary keys for arrival parameters
ARRIVAL_DICT_KEYS: dict = {
    'start': 'erlang_start',
    'stop': 'erlang_stop',
    'step': 'erlang_step'
}

# Environment setup constants
DEFAULT_ITERATION: int = 0
DEFAULT_ARRIVAL_COUNT: int = 0
