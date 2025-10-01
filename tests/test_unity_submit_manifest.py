"""
Unit tests for fusion.unity.submit_manifest module.

Tests the functionality for submitting job manifests to SLURM cluster,
including argument parsing, environment setup, and command building.
"""
import csv
import pathlib
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from tests.test_unity_fixtures import COMMON_RESOURCES
from fusion.unity.submit_manifest import (
    parse_cli,
    read_first_row,
    build_environment_variables,
    RESOURCE_KEYS,
)
from fusion.unity.errors import ManifestNotFoundError


class TestSubmitManifest(unittest.TestCase):
    """Test cases for submit_manifest module functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_manifest_data = [
            {
                "run_id": "00001",
                "path_algorithm": "ppo",
                "erlang_start": "100",
                "network": "test_network",
                **COMMON_RESOURCES
            },
            {
                "run_id": "00002",
                "path_algorithm": "dqn",
                "erlang_start": "200",
                "network": "test_network",
                "partition": "gpu",
                "time": "12:00:00",
                "mem": "16G",
                "cpus": "4",
                "gpus": "1",
                "nodes": "1"
            }
        ]

    def create_test_manifest(self) -> pathlib.Path:
        """Create a temporary manifest file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            fieldnames = self.test_manifest_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.test_manifest_data)
            return pathlib.Path(f.name)

    @patch('sys.argv', ['submit_manifest.py', 'test_exp', 'run_rl_sim.sh'])
    def test_parse_cli_basic(self):
        """Test parse_cli with basic arguments."""
        args = parse_cli()

        self.assertEqual(args.exp, 'test_exp')
        self.assertEqual(args.script, 'run_rl_sim.sh')
        self.assertIsNone(args.rows)

    @patch('sys.argv', ['submit_manifest.py', 'test_exp', 'run_rl_sim.sh', '--rows', '10'])
    def test_parse_cli_with_rows(self):
        """Test parse_cli with rows argument."""
        args = parse_cli()

        self.assertEqual(args.exp, 'test_exp')
        self.assertEqual(args.script, 'run_rl_sim.sh')
        self.assertEqual(args.rows, 10)

    def test_read_first_row(self):
        """Test read_first_row reads manifest correctly."""
        manifest_path = self.create_test_manifest()

        try:
            first_row, total_rows = read_first_row(manifest_path)

            self.assertEqual(total_rows, 2)
            self.assertEqual(first_row["run_id"], "00001")
            self.assertEqual(first_row["path_algorithm"], "ppo")
            self.assertEqual(first_row["network"], "test_network")
        finally:
            manifest_path.unlink()

    def test_read_first_row_empty_manifest(self):
        """Test read_first_row with empty manifest file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            # Write just the header
            f.write("run_id,algorithm\n")
            manifest_path = pathlib.Path(f.name)

        try:
            with self.assertRaises(ManifestNotFoundError):
                read_first_row(manifest_path)
        finally:
            manifest_path.unlink()

    def test_build_env_basic(self):
        """Test build_env creates correct environment variables."""
        first_row = self.test_manifest_data[0]
        job_dir = pathlib.Path("experiments/test_exp")
        exp = "test_exp"

        env = build_environment_variables(first_row, 2, job_dir, exp)

        # Check mandatory metadata
        self.assertEqual(env["MANIFEST"], "unity/experiments/test_exp/manifest.csv")
        self.assertEqual(env["N_JOBS"], "1")  # 2 rows - 1 (0-indexed)
        self.assertEqual(env["JOB_DIR"], "experiments/test_exp")
        self.assertEqual(env["NETWORK"], "test_network")
        self.assertIn("JOB_NAME", env)

        # Check resource propagation (should be uppercase)
        self.assertEqual(env["PARTITION"], "gpu")
        self.assertEqual(env["TIME"], "24:00:00")
        self.assertEqual(env["MEM"], "32G")
        self.assertEqual(env["CPUS"], "8")
        self.assertEqual(env["GPUS"], "1")
        self.assertEqual(env["NODES"], "1")

    def test_build_env_job_name_format(self):
        """Test build_env creates properly formatted job name."""
        first_row = self.test_manifest_data[0]
        job_dir = pathlib.Path("experiments/0315/142530")
        exp = "0315/142530"

        env = build_environment_variables(first_row, 5, job_dir, exp)

        expected_job_name = "ppo_100_0315_142530"
        self.assertEqual(env["JOB_NAME"], expected_job_name)

    def test_build_env_missing_network(self):
        """Test build_env handles missing network field."""
        first_row = self.test_manifest_data[0].copy()
        del first_row["network"]
        job_dir = pathlib.Path("experiments/test_exp")
        exp = "test_exp"

        env = build_environment_variables(first_row, 2, job_dir, exp)

        self.assertEqual(env["NETWORK"], "")

    def test_build_env_missing_resources(self):
        """Test build_env handles missing resource fields."""
        first_row = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
            "network": "test_network"
        }
        job_dir = pathlib.Path("experiments/test_exp")
        exp = "test_exp"

        env = build_environment_variables(first_row, 1, job_dir, exp)

        # Missing resource keys should not be in environment
        for key in RESOURCE_KEYS:
            self.assertNotIn(key.upper(), env)

    def test_build_env_empty_resource_values(self):
        """Test build_env skips empty resource values."""
        first_row = self.test_manifest_data[0].copy()
        first_row["partition"] = ""
        first_row["gpus"] = ""
        job_dir = pathlib.Path("experiments/test_exp")
        exp = "test_exp"

        env = build_environment_variables(first_row, 2, job_dir, exp)

        # Empty values should not be propagated
        self.assertNotIn("PARTITION", env)
        self.assertNotIn("GPUS", env)

        # Non-empty values should still be there
        self.assertEqual(env["TIME"], "24:00:00")
        self.assertEqual(env["MEM"], "32G")

    def test_resource_keys_constant(self):
        """Test RESOURCE_KEYS constant contains expected keys."""
        expected_keys = {"partition", "time", "mem", "cpus", "gpus", "nodes"}
        self.assertEqual(RESOURCE_KEYS, expected_keys)

    @patch('pathlib.Path.exists')
    @patch('fusion.unity.submit_manifest.read_first_row')
    @patch('subprocess.run')
    def test_main_integration(self, mock_subprocess, mock_read_first_row, mock_exists):
        """Test main function integration (mocked external calls)."""
        # Mock file system checks
        mock_exists.return_value = True

        # Mock manifest reading
        mock_read_first_row.return_value = (self.test_manifest_data[0], 2)

        # Mock successful subprocess execution
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Test would require more complex mocking of sys.argv and main execution
        # This is a placeholder for integration testing structure


if __name__ == '__main__':
    unittest.main()
