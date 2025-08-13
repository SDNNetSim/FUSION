import unittest
from unittest import mock
from fusion.cli.main_parser import build_parser, get_train_args, get_gui_args


class TestMainParser(unittest.TestCase):
    """
    Tests for FUSION CLI main_parser.py logic, including subcommands and argument validation.
    """

    def test_run_sim_basic(self):
        """
        Test basic run-sim command.
        """
        test_args = [
            "run_sim",
            "--run_id", "test123"
        ]
        parser = build_parser()
        args = parser.parse_args(test_args)
        self.assertEqual(args.mode, "run_sim")
        self.assertEqual(args.run_id, "test123")

    def test_run_sim_with_additional_args(self):
        """
        Test basic run-sim with additional args.
        """
        test_args = [
            "run_sim",
            "--run_id", "test123",
            "--route_method", "k_shortest_path",
            "--allocation_method", "first_fit",
            "--num_requests", "500"
        ]
        parser = build_parser()
        args = parser.parse_args(test_args)
        self.assertEqual(args.route_method, "k_shortest_path")
        self.assertEqual(args.allocation_method, "first_fit")
        self.assertEqual(args.num_requests, 500)

    def test_get_train_args_minimal(self):
        """
        Test basic get-train command with minimal args.
        """
        test_args = [
            "--config_path", "ini/run_ini/config.ini",
            "--run_id", "train123",
            "--agent_type", "rl"
        ]
        with mock.patch("sys.argv", ["prog"] + test_args):
            args = get_train_args()
            self.assertEqual(args.config_path, "ini/run_ini/config.ini")
            self.assertEqual(args.run_id, "train123")
            self.assertEqual(args.agent_type, "rl")

    def test_get_gui_args_minimal(self):
        """
        Test basic get-gui command with minimal args.
        """
        test_args = [
            "--config_path", "ini/run_ini/config.ini",
            "--run_id", "gui123"
        ]
        with mock.patch("sys.argv", ["prog"] + test_args):
            args = get_gui_args()
            self.assertEqual(args.config_path, "ini/run_ini/config.ini")
            self.assertEqual(args.run_id, "gui123")


if __name__ == '__main__':
    unittest.main()
