import os
import unittest
from unittest.mock import patch
from fusion.cli.main_parser import build_parser
from fusion.cli.config_setup import load_config, ConfigManager


class TestConfigManager(unittest.TestCase):
    """
    Unit tests for the FUSION ConfigManager and config loading behavior.
    """

    def setUp(self):
        base = os.path.dirname(__file__)
        self.valid_config = os.path.join(base, 'fixtures', 'valid_config.ini')
        self.invalid_config = os.path.join(base, 'fixtures', 'invalid_config.ini')

    @patch('sys.argv', ['prog', 'run_sim', '--config_path', 'tests/fixtures/valid_config.ini', '--run_id', 'test', '--network', 'NSFNet'])
    def test_successful_config_load(self):
        """
        Test successful config loading.
        """
        parser = build_parser()
        args = parser.parse_args()
        config = ConfigManager.from_args(args)
        config_dict = config.as_dict()

        self.assertIn("s1", config_dict)
        # CLI defaults override config file values due to argparse default handling
        self.assertEqual(config_dict["s1"]["erlang_start"], 100.0)  # CLI default
        self.assertEqual(config_dict["s1"]["holding_time"], 1.0)    # CLI default

    @patch('sys.argv', ['prog', 'run_sim', '--config_path', 'tests/fixtures/invalid_config.ini', '--run_id', 'test', '--network', 'NSFNet'])
    def test_invalid_config_read(self):
        """
        Test invalid config reading.
        """
        parser = build_parser()
        args = parser.parse_args()
        config = load_config(self.invalid_config, vars(args))
        self.assertEqual(config, {})  # load_config fails gracefully and returns {}

    @patch('sys.argv', ['prog', 'run_sim', '--config_path', 'non_existent_config.ini', '--run_id', 'test', '--network', 'NSFNet'])
    def test_missing_config_file(self):
        """
        Test missing config file.
        """
        parser = build_parser()
        args = parser.parse_args()
        config = load_config("non_existent_config.ini", vars(args))
        self.assertEqual(config, {})  # Should not raise, should return {}

    @patch('sys.argv', ['prog', 'run_sim', '--config_path', 'tests/fixtures/valid_config.ini', '--run_id', 'test', '--network', 'NSFNet', '--holding_time', '1.5'])
    def test_command_line_override(self):
        """
        Test command line override.
        """
        parser = build_parser()
        args = parser.parse_args()
        config = ConfigManager.from_args(args)
        config_dict = config.as_dict()
        self.assertEqual(config_dict["s1"]["holding_time"], 1.5)

    @patch('sys.argv', ['prog', 'run_sim', '--config_path', 'tests/fixtures/valid_config.ini', '--run_id', 'test', '--network', 'NSFNet'])
    def test_default_values_handled(self):
        """
        Test default values handled.
        """
        parser = build_parser()
        args = parser.parse_args()
        config = ConfigManager.from_args(args)
        config_dict = config.as_dict()
        self.assertIn("s1", config_dict)
        self.assertIsNone(config_dict["s1"].get("some_nonexistent_param"))

    @patch('sys.argv', ['prog', 'run_sim', '--config_path', 'tests/fixtures/valid_config.ini', '--run_id', 'test', '--network', 'NSFNet'])
    def test_multi_thread_sections(self):
        """
        Test multi-thread sections.
        """
        parser = build_parser()
        args = parser.parse_args()
        config = ConfigManager.from_args(args)
        config_dict = config.as_dict()
        self.assertIn("s2", config_dict)
        self.assertIn("s3", config_dict)
        self.assertIsInstance(config_dict["s2"], dict)
        self.assertIsInstance(config_dict["s3"], dict)


if __name__ == '__main__':
    unittest.main()
