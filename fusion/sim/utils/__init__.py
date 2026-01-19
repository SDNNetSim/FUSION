"""
Utility functions for FUSION simulations.

This module provides various utility functions for network analysis,
path calculations, spectrum management, and data processing used
throughout the FUSION simulation framework.
"""

# Network utilities
# Data processing utilities
from .data import (
    calculate_matrix_statistics,
    dict_to_list,
    min_max_scale,
    sort_dict_keys,
    sort_nested_dict_values,
    update_dict_from_list,
    update_matrices,
)

# Formatting utilities
from .formatting import (
    int_to_string,
    list_to_title,
    snake_to_title,
)

# I/O utilities
from .io import (
    modify_multiple_json_values,
    parse_yaml_file,
)
from .network import (
    classify_congestion,
    find_core_congestion,
    find_core_fragmentation_congestion,
    find_max_path_length,
    find_path_congestion,
    find_path_fragmentation,
    find_path_length,
    get_path_modulation,
)

# Simulation utilities
from .simulation import (
    get_erlang_values,
    get_start_time,
    log_message,
    run_simulation_for_erlangs,
    save_study_results,
)

# Spectrum utilities
from .spectrum import (
    combine_and_one_hot,
    find_free_channels,
    find_free_slots,
    find_taken_channels,
    get_channel_overlaps,
    get_shannon_entropy_fragmentation,
    get_super_channels,
)

# Define public API
__all__ = [
    # Network functions
    "classify_congestion",
    "find_core_congestion",
    "find_core_fragmentation_congestion",
    "find_max_path_length",
    "find_path_congestion",
    "find_path_fragmentation",
    "find_path_length",
    "get_path_modulation",
    # Spectrum functions
    "combine_and_one_hot",
    "find_free_channels",
    "find_free_slots",
    "find_taken_channels",
    "get_channel_overlaps",
    "get_shannon_entropy_fragmentation",
    "get_super_channels",
    # Data processing functions
    "calculate_matrix_statistics",
    "dict_to_list",
    "min_max_scale",
    "sort_dict_keys",
    "sort_nested_dict_values",
    "update_dict_from_list",
    "update_matrices",
    # Formatting functions
    "int_to_string",
    "list_to_title",
    "snake_to_title",
    # Simulation functions
    "get_erlang_values",
    "get_start_time",
    "log_message",
    "run_simulation_for_erlangs",
    "save_study_results",
    # I/O functions
    "modify_multiple_json_values",
    "parse_yaml_file",
    # Backward compatibility aliases
    "find_path_len",
    "get_path_mod",
    "find_path_cong",
    "find_path_frag",
    "sort_nested_dict_vals",
    "find_core_cong",
    "find_core_frag_cong",
    "get_erlang_vals",
    "get_hfrag",
    "classify_cong",
    "calc_matrix_stats",
    "find_max_path_len",
]

# Backward compatibility aliases
find_path_len = find_path_length
get_path_mod = get_path_modulation
find_path_cong = find_path_congestion
find_path_frag = find_path_fragmentation
sort_nested_dict_vals = sort_nested_dict_values
find_core_cong = find_core_congestion
find_core_frag_cong = find_core_fragmentation_congestion
get_erlang_vals = get_erlang_values
get_hfrag = get_shannon_entropy_fragmentation
classify_cong = classify_congestion
calc_matrix_stats = calculate_matrix_statistics
find_max_path_len = find_max_path_length
