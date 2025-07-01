import unittest
import os
from unittest.mock import patch
from config_scripts.setup_config import read_config
from config_scripts.parse_args import parse_args


class TestReadConfig(unittest.TestCase):
    """
    Tests setup_config.py.
    """

    def setUp(self):
        self.valid_conf = os.path.join('tests', 'fixtures', 'valid_config.ini')
        self.invalid_conf = os.path.join('tests', 'fixtures', 'invalid_config.ini')
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
        """
        Test successful configuration file.
        """
        args_obj = parse_args()
        config_dict = read_config(args_obj, self.valid_conf)
        self.assertIsNotNone(config_dict)
        self.assertIn('s1', config_dict)
        self.assertIsInstance(config_dict['s1'], dict)

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
        self.assertIn("Missing 'mod_assumption_path' in the general_settings section", str(context.exception))

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
