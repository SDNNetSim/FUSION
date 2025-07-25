import unittest
import os
from unittest.mock import patch
from config_scripts.setup_config import read_config
from config_scripts.parse_args import parse_args
from arg_scripts.config_args import SIM_REQUIRED_OPTIONS, OTHER_OPTIONS


class TestReadConfig(unittest.TestCase):
    """
    Tests setup_config.py.
    """

    def setUp(self):
        base = os.path.dirname(__file__)
        self.valid_conf = os.path.join(base, 'fixtures', 'valid_config.ini')
        self.invalid_conf = os.path.join(base, 'fixtures', 'invalid_config.ini')
        self.mock_args = ['program_name']

        os.makedirs('tests/ini', exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """
        Removes previously created directory.
        """
        try:
            os.rmdir('tests/ini')
        except FileNotFoundError:
            pass
        except OSError:
            for root, dirs, files in os.walk('tests/ini', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir('tests/ini')

    @patch('sys.argv', ['program_name'])
    def test_successful_config_read(self):
        """Test successful configuration parsing from valid_config.ini."""
        args_dict = {}
        for option_group in [SIM_REQUIRED_OPTIONS, OTHER_OPTIONS]:
            for _, options in option_group.items():
                args_dict.update({key: None for key in options})

        config_result = read_config(args_dict, self.valid_conf)

        assert 's1' in config_result
        assert isinstance(config_result['s1'], dict)
        self.assertEqual(config_result['s1']['erlang_start'], 300)
        self.assertEqual(config_result['s1']['holding_time'], 0.2)

    @patch('sys.argv', ['program_name'])
    def test_missing_config_file(self):
        """
        Test handling of a missing configuration file.
        """
        args_obj = parse_args()
        with self.assertRaises(ValueError):
            read_config(args_obj, 'None')

    @patch('sys.argv', ['program_name'])
    def test_invalid_config_read(self):
        """
        Test handling of an invalid configuration file.
        """
        args_obj = parse_args()
        with self.assertRaises(ValueError) as context:
            read_config(args_obj, self.invalid_conf)
        self.assertIn("Missing 'erlang_start' in the general_settings section", str(context.exception))


    @patch('sys.argv', ['program_name'])
    def test_command_line_input(self):
        """
        Test overriding configuration values with command line input.
        """
        with patch('sys.argv', ['program_name', '--holding_time', '1000']):
            args_obj = parse_args()
        config_dict = read_config(args_obj, self.valid_conf)
        self.assertEqual(config_dict['s1']['holding_time'], 1000.0)

    @patch('sys.argv', ['program_name'])
    def test_config_with_default_values(self):
        """
        Test that default values are set correctly when options are not specified.
        """
        args_obj = parse_args()
        config_dict = read_config(args_obj, self.valid_conf)
        self.assertIsNone(config_dict['s1'].get('some_optional_parameter'))

    @patch('sys.argv', ['program_name'])
    def test_multiple_simulation_threads(self):
        """
        Test configuration with multiple simulation threads.
        """
        args_obj = parse_args()
        config_dict = read_config(args_obj, self.valid_conf)
        self.assertIn('s2', config_dict)
        self.assertIn('s3', config_dict)
        self.assertIsInstance(config_dict['s2'], dict)
        self.assertIsInstance(config_dict['s3'], dict)


if __name__ == '__main__':
    unittest.main()
