"""
Observation space definitions for reinforcement learning environments.

This module defines the features included in different observation space
configurations used by RL agents for decision making.
"""
from typing import Dict, List, Set

# Observation space definitions mapping observation keys to their features
# Note: Dictionary keys (e.g., 'obs_1') are kept for backward compatibility
# but should be considered as 'observation_space_1' etc. in documentation
OBS_DICT: Dict[str, List[str]] = {
    # Basic routing information only
    "obs_1": ["source", "destination"],

    # Routing with bandwidth requirement
    "obs_2": ["source", "destination", "request_bandwidth"],

    # Routing with time constraint
    "obs_3": ["source", "destination", "holding_time"],

    # Routing with bandwidth and time constraints
    "obs_4": ["source", "destination", "request_bandwidth", "holding_time"],

    # Extended routing with path information
    "obs_5": ["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths"],

    # Extended routing with congestion information
    # Note: 'paths_cong' key maintained for backward compatibility (represents path congestion)
    "obs_6": ["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths",
              "paths_cong"],

    # Full routing information with resource availability
    "obs_7": ["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths",
              "paths_cong", "available_slots"],

    # Complete observation space with feasibility indicator
    "obs_8": ["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths",
              "paths_cong", "available_slots", "is_feasible"],
}

# Backward compatibility mapping
OBSERVATION_SPACE_DEFINITIONS: Dict[str, List[str]] = OBS_DICT

# Set of all valid observation features for validation
VALID_OBSERVATION_FEATURES: Set[str] = {
    "source",
    "destination",
    "request_bandwidth",
    "holding_time",
    "slots_needed",
    "path_lengths",
    "paths_cong",  # Path congestion levels
    "available_slots",
    "is_feasible"
}
