import unittest
import argparse

from fusion.configs.schema import SIM_REQUIRED_OPTIONS, OPTIONAL_OPTIONS
from fusion.cli.args.registry import args_registry


class TestConfigArgs(unittest.TestCase):
    """
    Test config_args.py script.
    """

    def test_command_line_and_config_options(self):  # pylint: disable=too-many-nested-blocks
        """ Test if command line params have all config options. """
        config_keys = self._get_config_keys()
        cli_params = self._extract_cli_params()
        # Ignore known special-case CLI args not handled via config
        ignored_keys = {
            'optimize', 'config_path', 'run_id', 'verbose', 'debug', 'output_dir', 'save_results', 'mode',
            'request_distribution',
            # RL/ML specific parameters - config-only
            'gamma', 'device', 'render_mode', 'path_model', 'heads', 'dynamic_reward', 'alpha_end', 'decay_rate',
            'feature_extractor', 'epsilon_update', 'core_beta', 'reward', 'conf_param', 'path_levels', 'penalty',
            'super_channel_space', 'core_algorithm', 'n_trials', 'optuna_trials', 'cong_cutoff', 'spectrum_model',
            'spectrum_algorithm', 'alpha_start', 'obs_space', 'layers', 'gnn_type', 'is_training', 'core_model',
            'alpha_update', 'path_algorithm', 'optimize_hyperparameters', 'epsilon_end', 'epsilon_start',
            'xt_noise', 'requested_xt', 'beta', 'xt_type', 'phi', 'theta',
            'max_segments', 'fixed_grid', 'pre_calc_mod_selection', 'emb_dim'
        }
        missing_in_cli = config_keys - cli_params - ignored_keys
        self.assertFalse(missing_in_cli, f"These config options are missing in "
                                         f"command line parameters: {missing_in_cli}")

    def _get_config_keys(self):
        """Extract all configuration keys from config dictionaries."""
        config_keys = set()
        for option_group in [SIM_REQUIRED_OPTIONS, OPTIONAL_OPTIONS]:
            for _, options in option_group.items():
                config_keys.update(options.keys())
        return config_keys

    def _extract_cli_params(self):
        """Extract CLI parameter names from the parser."""
        # Create a parser with all run_sim related argument groups
        parser = args_registry.create_main_parser()
        cli_params = set()

        # Get the run_sim subparser
        # pylint: disable=protected-access
        for action in parser._subparsers._actions:
            if isinstance(action, argparse._SubParsersAction):
                cli_params = self._extract_subparser_params(action)
                break
        return cli_params

    def _extract_subparser_params(self, subparsers_action):
        """Extract parameters from the run_sim subparser."""
        cli_params = set()
        for subparser_name, subparser in subparsers_action.choices.items():
            if subparser_name == 'run_sim':
                # Extract all arguments from the run_sim subparser
                # pylint: disable=protected-access
                for action in subparser._actions:
                    if hasattr(action, 'dest') and action.dest != 'help':
                        cli_params.add(action.dest)
                break
        return cli_params


if __name__ == '__main__':
    unittest.main()
