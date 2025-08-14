"""CLI argument to configuration mapping for FUSION simulator."""

from typing import Dict, Any
import argparse


class CLIToConfigMapper:
    """Maps CLI arguments to configuration structure."""

    def __init__(self):
        """Initialize CLI to config mapper."""
        # Define mapping from CLI argument names to config sections and keys
        self.arg_mapping = {
            # General settings
            'holding_time': ('general_settings', 'holding_time'),
            'mod_assumption': ('general_settings', 'mod_assumption'),
            'mod_assumption_path': ('general_settings', 'mod_assumption_path'),
            'erlang_start': ('general_settings', 'erlang_start'),
            'erlang_stop': ('general_settings', 'erlang_stop'),
            'erlang_step': ('general_settings', 'erlang_step'),
            'max_iters': ('general_settings', 'max_iters'),
            'guard_slots': ('general_settings', 'guard_slots'),
            'max_segments': ('general_settings', 'max_segments'),
            'thread_erlangs': ('general_settings', 'thread_erlangs'),
            'dynamic_lps': ('general_settings', 'dynamic_lps'),
            'fixed_grid': ('general_settings', 'fixed_grid'),
            'pre_calc_mod_selection': ('general_settings', 'pre_calc_mod_selection'),
            'spectrum_priority': ('general_settings', 'spectrum_priority'),
            'num_requests': ('general_settings', 'num_requests'),
            'request_distribution': ('general_settings', 'request_distribution'),
            'allocation_method': ('general_settings', 'allocation_method'),
            'k_paths': ('general_settings', 'k_paths'),
            'route_method': ('general_settings', 'route_method'),
            'save_snapshots': ('general_settings', 'save_snapshots'),
            'snapshot_step': ('general_settings', 'snapshot_step'),
            'print_step': ('general_settings', 'print_step'),
            'save_step': ('general_settings', 'save_step'),
            'save_start_end_slots': ('general_settings', 'save_start_end_slots'),

            # Topology settings
            'network': ('topology_settings', 'network'),
            'bw_per_slot': ('topology_settings', 'bw_per_slot'),
            'cores_per_link': ('topology_settings', 'cores_per_link'),
            'const_link_weight': ('topology_settings', 'const_link_weight'),
            'is_only_core_node': ('topology_settings', 'is_only_core_node'),
            'multi_fiber': ('topology_settings', 'multi_fiber'),

            # Spectrum settings
            'c_band': ('spectrum_settings', 'c_band'),

            # SNR settings
            'snr_type': ('snr_settings', 'snr_type'),
            'xt_type': ('snr_settings', 'xt_type'),
            'beta': ('snr_settings', 'beta'),
            'theta': ('snr_settings', 'theta'),
            'input_power': ('snr_settings', 'input_power'),
            'egn_model': ('snr_settings', 'egn_model'),
            'phi': ('snr_settings', 'phi'),
            'bi_directional': ('snr_settings', 'bi_directional'),
            'xt_noise': ('snr_settings', 'xt_noise'),
            'requested_xt': ('snr_settings', 'requested_xt'),

            # RL settings
            'obs_space': ('rl_settings', 'obs_space'),
            'n_trials': ('rl_settings', 'n_trials'),
            'device': ('rl_settings', 'device'),
            'optimize_hyperparameters': ('rl_settings', 'optimize_hyperparameters'),
            'optuna_trials': ('rl_settings', 'optuna_trials'),
            'is_training': ('rl_settings', 'is_training'),
            'path_algorithm': ('rl_settings', 'path_algorithm'),
            'path_model': ('rl_settings', 'path_model'),
            'core_algorithm': ('rl_settings', 'core_algorithm'),
            'core_model': ('rl_settings', 'core_model'),
            'spectrum_algorithm': ('rl_settings', 'spectrum_algorithm'),
            'spectrum_model': ('rl_settings', 'spectrum_model'),
            'render_mode': ('rl_settings', 'render_mode'),
            'super_channel_space': ('rl_settings', 'super_channel_space'),
            'alpha_start': ('rl_settings', 'alpha_start'),
            'alpha_end': ('rl_settings', 'alpha_end'),
            'alpha_update': ('rl_settings', 'alpha_update'),
            'gamma': ('rl_settings', 'gamma'),
            'epsilon_start': ('rl_settings', 'epsilon_start'),
            'epsilon_end': ('rl_settings', 'epsilon_end'),
            'epsilon_update': ('rl_settings', 'epsilon_update'),
            'path_levels': ('rl_settings', 'path_levels'),
            'decay_rate': ('rl_settings', 'decay_rate'),
            'feature_extractor': ('rl_settings', 'feature_extractor'),
            'gnn_type': ('rl_settings', 'gnn_type'),
            'layers': ('rl_settings', 'layers'),
            'emb_dim': ('rl_settings', 'emb_dim'),
            'heads': ('rl_settings', 'heads'),
            'conf_param': ('rl_settings', 'conf_param'),
            'cong_cutoff': ('rl_settings', 'cong_cutoff'),
            'reward': ('rl_settings', 'reward'),
            'penalty': ('rl_settings', 'penalty'),
            'dynamic_reward': ('rl_settings', 'dynamic_reward'),
            'core_beta': ('rl_settings', 'core_beta'),

            # ML settings
            'deploy_model': ('ml_settings', 'deploy_model'),
            'output_train_data': ('ml_settings', 'output_train_data'),
            'ml_training': ('ml_settings', 'ml_training'),
            'ml_model': ('ml_settings', 'ml_model'),
            'train_file_path': ('ml_settings', 'train_file_path'),
            'test_size': ('ml_settings', 'test_size'),

            # File settings
            'file_type': ('file_settings', 'file_type'),
        }

    def map_args_to_config(self, args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Map CLI arguments to configuration structure.
        
        Args:
            args: Dictionary of CLI arguments
            
        Returns:
            Configuration dictionary organized by sections
        """
        config = {}

        for arg_name, value in args.items():
            if value is None:
                continue

            if arg_name in self.arg_mapping:
                section, key = self.arg_mapping[arg_name]

                if section not in config:
                    config[section] = {}

                config[section][key] = value
            else:
                # Handle unmapped arguments - put them in general_settings
                if 'general_settings' not in config:
                    config['general_settings'] = {}
                config['general_settings'][arg_name] = value

        return config

    def map_namespace_to_config(self, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
        """Map argparse Namespace to configuration structure.
        
        Args:
            args: argparse.Namespace object
            
        Returns:
            Configuration dictionary organized by sections
        """
        return self.map_args_to_config(vars(args))

    def get_cli_override_config(self, cli_args: Dict[str, Any],
                                base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration with CLI arguments overriding base config.
        
        Args:
            cli_args: CLI arguments dictionary
            base_config: Base configuration dictionary
            
        Returns:
            Merged configuration with CLI overrides
        """
        # Start with base config
        result = base_config.copy()

        # Map CLI args to config format
        cli_config = self.map_args_to_config(cli_args)

        # Merge CLI overrides into base config
        for section, values in cli_config.items():
            if section not in result:
                result[section] = {}
            result[section].update(values)

        return result

    def get_reverse_mapping(self) -> Dict[str, str]:
        """Get reverse mapping from config path to CLI argument name.
        
        Returns:
            Dictionary mapping (section, key) tuples to CLI argument names
        """
        reverse_map = {}
        for cli_arg, (section, key) in self.arg_mapping.items():
            reverse_map[f"{section}.{key}"] = cli_arg
        return reverse_map
