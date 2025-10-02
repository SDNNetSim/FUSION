"""CLI argument to configuration mapping for FUSION simulator."""

import argparse
from typing import Any

from fusion.configs.errors import ConfigTypeConversionError


class CLIToConfigMapper:
    """
    Maps CLI arguments to configuration structure.

    This class provides utilities to map command-line arguments to the
    hierarchical configuration structure used by FUSION. It maintains a
    mapping dictionary that associates CLI argument names with their
    corresponding configuration sections and keys.
    """

    def __init__(self) -> None:
        """
        Initialize CLI to config mapper.

        Sets up the internal mapping dictionary that defines how CLI
        arguments correspond to configuration sections and keys.
        """
        # Define mapping from CLI argument names to config sections and keys
        self.arg_mapping: dict[str, tuple[str, str]] = {
            # General settings
            "holding_time": ("general_settings", "holding_time"),
            "mod_assumption": ("general_settings", "mod_assumption"),
            "mod_assumption_path": ("general_settings", "mod_assumption_path"),
            "erlang_start": ("general_settings", "erlang_start"),
            "erlang_stop": ("general_settings", "erlang_stop"),
            "erlang_step": ("general_settings", "erlang_step"),
            "max_iters": ("general_settings", "max_iters"),
            "guard_slots": ("general_settings", "guard_slots"),
            "max_segments": ("general_settings", "max_segments"),
            "thread_erlangs": ("general_settings", "thread_erlangs"),
            "dynamic_lps": ("general_settings", "dynamic_lps"),
            "fixed_grid": ("general_settings", "fixed_grid"),
            "pre_calc_mod_selection": ("general_settings", "pre_calc_mod_selection"),
            "spectrum_priority": ("general_settings", "spectrum_priority"),
            "num_requests": ("general_settings", "num_requests"),
            "request_distribution": ("general_settings", "request_distribution"),
            "allocation_method": ("general_settings", "allocation_method"),
            "k_paths": ("general_settings", "k_paths"),
            "route_method": ("general_settings", "route_method"),
            "save_snapshots": ("general_settings", "save_snapshots"),
            "snapshot_step": ("general_settings", "snapshot_step"),
            "print_step": ("general_settings", "print_step"),
            "save_step": ("general_settings", "save_step"),
            "save_start_end_slots": ("general_settings", "save_start_end_slots"),
            # Topology settings
            "network": ("topology_settings", "network"),
            "bw_per_slot": ("topology_settings", "bw_per_slot"),
            "cores_per_link": ("topology_settings", "cores_per_link"),
            "const_link_weight": ("topology_settings", "const_link_weight"),
            "is_only_core_node": ("topology_settings", "is_only_core_node"),
            "multi_fiber": ("topology_settings", "multi_fiber"),
            # Spectrum settings
            "c_band": ("spectrum_settings", "c_band"),
            # SNR settings
            "snr_type": ("snr_settings", "snr_type"),
            "xt_type": ("snr_settings", "xt_type"),
            "beta": ("snr_settings", "beta"),
            "theta": ("snr_settings", "theta"),
            "input_power": ("snr_settings", "input_power"),
            "egn_model": ("snr_settings", "egn_model"),
            "phi": ("snr_settings", "phi"),
            "bi_directional": ("snr_settings", "bi_directional"),
            "xt_noise": ("snr_settings", "xt_noise"),
            "requested_xt": ("snr_settings", "requested_xt"),
            # RL settings
            "obs_space": ("rl_settings", "obs_space"),
            "n_trials": ("rl_settings", "n_trials"),
            "device": ("rl_settings", "device"),
            "optimize_hyperparameters": ("rl_settings", "optimize_hyperparameters"),
            "optuna_trials": ("rl_settings", "optuna_trials"),
            "is_training": ("rl_settings", "is_training"),
            "path_algorithm": ("rl_settings", "path_algorithm"),
            "path_model": ("rl_settings", "path_model"),
            "core_algorithm": ("rl_settings", "core_algorithm"),
            "core_model": ("rl_settings", "core_model"),
            "spectrum_algorithm": ("rl_settings", "spectrum_algorithm"),
            "spectrum_model": ("rl_settings", "spectrum_model"),
            "render_mode": ("rl_settings", "render_mode"),
            "super_channel_space": ("rl_settings", "super_channel_space"),
            "alpha_start": ("rl_settings", "alpha_start"),
            "alpha_end": ("rl_settings", "alpha_end"),
            "alpha_update": ("rl_settings", "alpha_update"),
            "gamma": ("rl_settings", "gamma"),
            "epsilon_start": ("rl_settings", "epsilon_start"),
            "epsilon_end": ("rl_settings", "epsilon_end"),
            "epsilon_update": ("rl_settings", "epsilon_update"),
            "path_levels": ("rl_settings", "path_levels"),
            "decay_rate": ("rl_settings", "decay_rate"),
            "feature_extractor": ("rl_settings", "feature_extractor"),
            "gnn_type": ("rl_settings", "gnn_type"),
            "layers": ("rl_settings", "layers"),
            "emb_dim": ("rl_settings", "emb_dim"),
            "heads": ("rl_settings", "heads"),
            "conf_param": ("rl_settings", "conf_param"),
            "cong_cutoff": ("rl_settings", "cong_cutoff"),
            "reward": ("rl_settings", "reward"),
            "penalty": ("rl_settings", "penalty"),
            "dynamic_reward": ("rl_settings", "dynamic_reward"),
            "core_beta": ("rl_settings", "core_beta"),
            # ML settings
            "deploy_model": ("ml_settings", "deploy_model"),
            "output_train_data": ("ml_settings", "output_train_data"),
            "ml_training": ("ml_settings", "ml_training"),
            "ml_model": ("ml_settings", "ml_model"),
            "train_file_path": ("ml_settings", "train_file_path"),
            "test_size": ("ml_settings", "test_size"),
            # File settings
            "file_type": ("file_settings", "file_type"),
        }

    def map_args_to_config(self, args: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Map CLI arguments to configuration structure.

        Takes a flat dictionary of CLI arguments and transforms it into a
        hierarchical configuration dictionary organized by sections.

        :param args: Dictionary of CLI arguments where keys are argument names
                    and values are the argument values
        :type args: Dict[str, Any]
        :return: Configuration dictionary organized by sections where each
                section contains its respective configuration parameters
        :rtype: Dict[str, Dict[str, Any]]

        Example:
            >>> mapper = CLIToConfigMapper()
            >>> cli_args = {'holding_time': 10, 'network': 'nsfnet'}
            >>> config = mapper.map_args_to_config(cli_args)
            >>> print(config)
            {'general_settings': {'holding_time': 10},
             'topology_settings': {'network': 'nsfnet'}}
        """
        config: dict[str, dict[str, Any]] = {}

        for arg_name, value in args.items():
            if value is None:
                continue

            try:
                if arg_name in self.arg_mapping:
                    section, key = self.arg_mapping[arg_name]

                    if section not in config:
                        config[section] = {}

                    config[section][key] = value
                else:
                    # Handle unmapped arguments - put them in general_settings
                    if "general_settings" not in config:
                        config["general_settings"] = {}
                    config["general_settings"][arg_name] = value
            except Exception as e:
                raise ConfigTypeConversionError(
                    f"Failed to map argument '{arg_name}' with value "
                    f"'{value}': {str(e)}"
                ) from e

        return config

    def map_namespace_to_config(
        self, args: argparse.Namespace
    ) -> dict[str, dict[str, Any]]:
        """
        Map argparse Namespace to configuration structure.

        Convenience method that converts an argparse.Namespace object to a
        configuration dictionary.

        :param args: argparse.Namespace object containing parsed CLI arguments
        :type args: argparse.Namespace
        :return: Configuration dictionary organized by sections
        :rtype: Dict[str, Dict[str, Any]]
        :raises AttributeError: If args is not a valid Namespace object
        """
        try:
            return self.map_args_to_config(vars(args))
        except AttributeError as e:
            raise AttributeError(
                f"Invalid argparse.Namespace object provided: {str(e)}"
            ) from e

    def get_cli_override_config(
        self, cli_args: dict[str, Any], base_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get configuration with CLI arguments overriding base config.

        Merges CLI arguments with a base configuration, where CLI arguments
        take precedence over base configuration values.

        :param cli_args: CLI arguments dictionary to override base config
        :type cli_args: Dict[str, Any]
        :param base_config: Base configuration dictionary to be overridden
        :type base_config: Dict[str, Any]
        :return: Merged configuration with CLI overrides applied
        :rtype: Dict[str, Any]

        Example:
            >>> base = {'general_settings': {'holding_time': 5}}
            >>> cli = {'holding_time': 10}
            >>> merged = mapper.get_cli_override_config(cli, base)
            >>> print(merged['general_settings']['holding_time'])
            10
        """
        # Validate inputs
        if not isinstance(base_config, dict):
            raise TypeError(
                f"base_config must be a dictionary, got {type(base_config).__name__}"
            )

        # Start with deep copy of base config to avoid modifying original
        result: dict[str, Any] = {}
        for key, value in base_config.items():
            if isinstance(value, dict):
                result[key] = value.copy()
            else:
                result[key] = value

        # Map CLI args to config format
        cli_config = self.map_args_to_config(cli_args)

        # Merge CLI overrides into base config
        for section, values in cli_config.items():
            if section not in result:
                result[section] = {}
            result[section].update(values)

        return result

    def get_reverse_mapping(self) -> dict[str, str]:
        """Get reverse mapping from config path to CLI argument name.

        Creates a reverse lookup dictionary that maps configuration paths
        (in the format 'section.key') to their corresponding CLI argument names.

        :return: Dictionary mapping config paths to CLI argument names
        :rtype: Dict[str, str]

        Example:
            >>> mapper = CLIToConfigMapper()
            >>> reverse_map = mapper.get_reverse_mapping()
            >>> print(reverse_map['general_settings.holding_time'])
            'holding_time'
        """
        reverse_map: dict[str, str] = {}
        for cli_arg, (section, key) in self.arg_mapping.items():
            reverse_map[f"{section}.{key}"] = cli_arg
        return reverse_map
