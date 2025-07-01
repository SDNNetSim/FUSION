import unittest

from arg_scripts.config_args import SIM_REQUIRED_OPTIONS, OTHER_OPTIONS, COMMAND_LINE_PARAMS


class TestConfigArgs(unittest.TestCase):
    """
    Test config_args.py script.
    """

    def test_command_line_and_config_options(self):
        """ Test if command line params have all config options. """
        config_keys = set()
        for option_group in [SIM_REQUIRED_OPTIONS, OTHER_OPTIONS]:
            for _, options in option_group.items():
                config_keys.update(options.keys())

        cli_params = {param[0] for param in COMMAND_LINE_PARAMS}
        missing_in_cli = config_keys - cli_params
        self.assertFalse(missing_in_cli, f"These config options are missing in "
                                         f"command line parameters: {missing_in_cli}")


if __name__ == '__main__':
    unittest.main()
