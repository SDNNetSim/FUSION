"""
Configuration schema definitions for FUSION simulator.

This module defines the required and optional configuration options
for the FUSION simulator, including their expected types and conversion
functions. These schemas are used for validation and type conversion
during configuration loading.
"""

# Standard library imports
from collections.abc import Callable
from typing import Any

# Third-party imports
# None
# Local application imports
from fusion.utils.config import str_to_bool

# Required configuration options for simulation
SIM_REQUIRED_OPTIONS_DICT: dict[str, dict[str, Callable[..., Any]]] = {
    "general_settings": {
        "erlang_start": float,
        "erlang_stop": float,
        "erlang_step": float,
        "mod_assumption": str,
        "mod_assumption_path": str,
        "holding_time": float,
        "thread_erlangs": str_to_bool,
        "guard_slots": int,
        "num_requests": int,
        "max_iters": int,
        "dynamic_lps": str_to_bool,
        "fixed_grid": str_to_bool,
        "pre_calc_mod_selection": str_to_bool,
        "max_segments": int,
        "save_snapshots": str_to_bool,
        "snapshot_step": int,
        "print_step": int,
        "spectrum_priority": str,
        "save_step": int,
        "save_start_end_slots": str_to_bool,
        "is_grooming_enabled": str_to_bool,
        "can_partially_serve": str_to_bool,
        "transponder_usage_per_node": str_to_bool,
        "blocking_type_ci": str_to_bool,
    },
    "topology_settings": {
        "network": str,
        "bw_per_slot": float,
        "cores_per_link": int,
        "const_link_weight": str_to_bool,
        "is_only_core_node": str_to_bool,
        "multi_fiber": str_to_bool,
    },
    "spectrum_settings": {
        "c_band": int,
    },
    "snr_settings": {
        "snr_type": str,
        "input_power": float,
        "egn_model": str_to_bool,
        "beta": float,
        "theta": float,
        "xt_type": str,
        "xt_noise": str_to_bool,
        "requested_xt": str,
        "phi": str,
        "snr_recheck": str_to_bool,
        "recheck_adjacent_cores": str_to_bool,
        "recheck_crossband": str_to_bool,
    },
    "file_settings": {
        "file_type": str,
    },
    "ml_settings": {
        "deploy_model": str_to_bool,
    },
}

# Optional configuration options with their type converters
OPTIONAL_OPTIONS_DICT: dict[str, dict[str, Callable[..., Any]]] = {
    "general_settings": {
        "filter_mods": bool,
        "request_distribution": str,
        "fragmentation_metrics": str,
        "frag_calc_step": int,
    },
    "topology_settings": {
        "bi_directional": str_to_bool,
        "is_only_core_node": str_to_bool,
    },
    "spectrum_settings": {
        "o_band": int,
        "e_band": int,
        "s_band": int,
        "l_band": int,
        "guard_slots": int,
        "allocation_method": str,
    },
    "file_settings": {
        "run_id": str,
    },
    "rl_settings": {
        "obs_space": str,
        "n_trials": int,
        "device": str,
        "optimize_hyperparameters": str_to_bool,
        "optuna_trials": int,
        "is_training": str_to_bool,
        "path_algorithm": str,
        "path_model": str,
        "core_algorithm": str,
        "core_model": str,
        "spectrum_algorithm": str,
        "spectrum_model": str,
        "render_mode": str,
        "super_channel_space": int,
        "alpha_start": float,
        "alpha_end": float,
        "alpha_update": str,
        "gamma": float,
        "epsilon_start": float,
        "epsilon_end": float,
        "epsilon_update": str,
        "path_levels": int,
        "decay_rate": float,
        "feature_extractor": str,
        "gnn_type": str,
        "layers": int,
        "emb_dim": int,
        "heads": int,
        "conf_param": int,
        "cong_cutoff": float,
        "reward": int,
        "penalty": int,
        "dynamic_reward": str_to_bool,
        "core_beta": float,
    },
    "ml_settings": {
        "deploy_model": str_to_bool,
        "output_train_data": str_to_bool,
        "ml_training": str_to_bool,
        "ml_model": str,
        "train_file_path": str,
        "test_size": float,
    },
    "dataset_logging_settings": {
        "log_offline_dataset": str_to_bool,
        "dataset_output_path": str,
        "epsilon_mix": float,
    },
    "offline_rl_settings": {
        "policy_type": str,
        "fallback_policy": str,
        "device": str,
    },
    "recovery_timing_settings": {
        "protection_switchover_ms": float,
        "restoration_latency_ms": float,
        "failure_window_size": int,
    },
    "protection_settings": {
        "revert_to_primary": str_to_bool,
    },
    "routing_settings": {
        "route_method": str,
        "k_paths": int,
        "path_ordering": str,
        "precompute_paths": str_to_bool,
    },
    "failure_settings": {
        "failure_type": str,
        "geo_center_node": int,
        "geo_hop_radius": int,
        "t_fail_arrival_index": int,
        "t_repair_after_arrivals": int,
        "failed_link_src": int,
        "failed_link_dst": int,
    },
    "reporting_settings": {
        "export_csv": str_to_bool,
        "csv_output_path": str,
    },
}
