"""
Unit tests for fusion.unity.make_manifest module.

Tests the functionality for creating experiment manifests from specifications,
including grid expansion, parameter validation, and CSV generation.
"""

import csv
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from fusion.unity.make_manifest import (
    _cast,
    _encode,
    _expand_grid,
    _explicit,
    _fetch,
    _is_rl,
    _parse_literal,
    _resolve_spec_path,
    _str_to_bool,
    _to_list,
    _validate_resource_keys,
    _write_csv,
)
from tests.test_unity_fixtures import COMMON_RESOURCES


class TestMakeManifest(unittest.TestCase):
    """Test cases for make_manifest module functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_resources = COMMON_RESOURCES

    def test_str_to_bool(self):
        """Test _str_to_bool with various inputs."""
        # Test true values
        true_values = ["true", "yes", "1", "TRUE", "YES"]
        for value in true_values:
            with self.subTest(value=value, expected=True):
                self.assertTrue(_str_to_bool(value))

        # Test false values
        false_values = ["false", "no", "0", "FALSE", "NO", "anything_else"]
        for value in false_values:
            with self.subTest(value=value, expected=False):
                self.assertFalse(_str_to_bool(value))

    def test_parse_literal(self):
        """Test _parse_literal with valid and invalid expressions."""
        # Test valid literals
        valid_cases = [
            ("[1, 2, 3]", [1, 2, 3]),
            ('{"key": "value"}', {"key": "value"}),
            ("42", 42),
            ("3.14", 3.14),
            ('"string"', "string"),
        ]
        for input_val, expected in valid_cases:
            with self.subTest(input_val=input_val, expected=expected):
                result = _parse_literal(input_val)
                self.assertEqual(result, expected)

        # Test invalid literals
        invalid_inputs = ["invalid_literal", "func()", "1 + 2"]
        for input_val in invalid_inputs:
            with self.subTest(input_val=input_val, valid=False):
                result = _parse_literal(input_val)
                self.assertEqual(result, input_val)

    def test_cast(self):
        """Test _cast function with various types."""
        # Test boolean type
        with patch("fusion.unity.make_manifest._PARAM_TYPES", {"enable_feature": bool}):
            test_cases = [
                ("true", True),
                ("false", False),
                ("1", True),
                ("0", False),
                (True, True),
                (False, False),
            ]
            for input_val, expected in test_cases:
                with self.subTest(
                    param="enable_feature", value=input_val, expected=expected
                ):
                    result = _cast("enable_feature", input_val)
                    self.assertEqual(result, expected)

        # Test list type
        with patch("fusion.unity.make_manifest._PARAM_TYPES", {"algorithms": list}):
            result = _cast("algorithms", '["ppo", "dqn"]')
            self.assertEqual(result, ["ppo", "dqn"])

        # Test unknown type
        result = _cast("unknown_param", "some_value")
        self.assertEqual(result, "some_value")

    def test_encode(self):
        """Test _encode function with various data types."""
        # Test boolean values
        self.assertEqual(_encode(True), "true")
        self.assertEqual(_encode(False), "false")

        # Test list and dict values
        self.assertEqual(_encode([1, 2, 3]), "[1,2,3]")
        self.assertEqual(_encode({"key": "value"}), '{"key":"value"}')

        # Test float values
        test_cases = [
            (3.14159, "3.14159"),
            (1.0, "1"),
            (0.000057, "0.000057"),
            (1000.0, "1000"),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _encode(input_val)
                self.assertEqual(result, expected)

    def test_is_rl(self):
        """Test _is_rl function identifies RL algorithms correctly."""
        rl_algorithms = [
            "ppo",
            "qr_dqn",
            "a2c",
            "dqn",
            "epsilon_greedy_bandit",
            "ucb_bandit",
            "q_learning",
        ]
        non_rl_algorithms = ["shortest_path", "k_shortest_path", "dijkstra"]

        for alg in rl_algorithms:
            with self.subTest(alg=alg):
                self.assertEqual(_is_rl(alg), "yes")

        for alg in non_rl_algorithms:
            with self.subTest(alg=alg):
                self.assertEqual(_is_rl(alg), "no")

    def test_validate_resource_keys(self):
        """Test _validate_resource_keys with valid and invalid resource keys."""
        # Test valid keys
        try:
            _validate_resource_keys(self.test_resources)
        except SystemExit:
            self.fail("_validate_resource_keys raised SystemExit with valid keys")

        # Test invalid keys
        invalid_resources = {"invalid_key": "value", "partition": "gpu"}
        with self.assertRaises(SystemExit):
            _validate_resource_keys(invalid_resources)

    def test_to_list(self):
        """Test _to_list with various inputs and contexts."""
        # Test single value
        result = _to_list("single_value", ctx="test")
        self.assertEqual(result, ["single_value"])

        # Test already a list
        input_list = ["item1", "item2", "item3"]
        result = _to_list(input_list, ctx="grid")
        self.assertEqual(result, input_list)

        # Test common context with multiple values (should raise)
        with self.assertRaises(SystemExit):
            _to_list(["value1", "value2"], ctx="common")

    def test_fetch(self):
        """Test _fetch retrieves values from grid and common sections."""
        grid = {"path_algorithm": ["ppo", "dqn"]}
        common = {"erlang_start": 100}

        # Test fetch from grid
        result = _fetch(grid, common, "path_algorithm")
        self.assertEqual(result, ["ppo", "dqn"])

        # Test fetch from common
        result = _fetch(grid, common, "erlang_start")
        self.assertEqual(result, [100])

        # Test missing key raises
        with self.assertRaises(SystemExit):
            _fetch({}, {}, "missing_key")

    def test_expand_grid_basic(self):
        """Test _expand_grid with basic grid configuration."""
        grid = {
            "common": {"network": "test_network", "num_requests": 1000},
            "path_algorithm": ["ppo", "dqn"],
            "erlang_start": [100, 200],
            "k_paths": [5],
            "obs_space": ["basic"],
        }

        rows, final_rid = _expand_grid(grid, 0)

        # Should generate 2 algorithms × 2 traffic levels × 1 k_paths × 1 obs_space = 4 rows
        self.assertEqual(len(rows), 4)
        self.assertEqual(final_rid, 4)

        # Check first row structure
        first_row = rows[0]
        self.assertIn("run_id", first_row)
        self.assertIn("path_algorithm", first_row)
        self.assertIn("erlang_start", first_row)
        self.assertIn("erlang_stop", first_row)
        self.assertIn("k_paths", first_row)
        self.assertIn("obs_space", first_row)
        self.assertIn("is_rl", first_row)
        self.assertIn("network", first_row)
        self.assertIn("num_requests", first_row)

    def test_explicit_jobs(self):
        """Test _explicit function with explicit job definitions."""
        jobs = [
            {
                "algorithm": "ppo",
                "traffic": 100,
                "erlang_start": 100,  # Add this for the default calculation
                "k_paths": 5,
                "network": "test_net",
            },
            {
                "algorithm": "dqn",
                "traffic": 200,
                "erlang_start": 200,  # Add this for the default calculation
                "k_paths": 10,
                "network": "test_net",
            },
        ]

        with patch(
            "fusion.unity.make_manifest._validate_keys"
        ):  # Skip validation for test
            rows = _explicit(jobs)

        self.assertEqual(len(rows), 2)

        # Check first row
        first_row = rows[0]
        self.assertEqual(first_row["path_algorithm"], "ppo")
        self.assertEqual(first_row["erlang_start"], 100)
        self.assertEqual(first_row["k_paths"], 5)
        self.assertEqual(first_row["is_rl"], "yes")

    def test_write_csv(self):
        """Test _write_csv creates correct CSV output."""
        rows = [
            {"run_id": "00001", "algorithm": "ppo", "traffic": 100},
            {
                "run_id": "00002",
                "algorithm": "dqn",
                "traffic": 200,
                "extra_param": "value",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            csv_path = pathlib.Path(f.name)

        try:
            _write_csv(csv_path, rows)

            # Read back and verify
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                read_rows = list(reader)

            self.assertEqual(len(read_rows), 2)
            self.assertEqual(read_rows[0]["run_id"], "00001")
            self.assertEqual(read_rows[0]["algorithm"], "ppo")
            self.assertEqual(read_rows[1]["extra_param"], "value")
            # Missing fields should be empty
            self.assertEqual(read_rows[0]["extra_param"], "")

        finally:
            csv_path.unlink()

    def test_resolve_spec_path_absolute(self):
        """Test _resolve_spec_path with absolute path."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            spec_path = pathlib.Path(f.name)

        try:
            result = _resolve_spec_path(str(spec_path))
            self.assertEqual(result, spec_path)
        finally:
            spec_path.unlink()

    def test_resolve_spec_path_not_found(self):
        """Test _resolve_spec_path with non-existent file."""
        with self.assertRaises(SystemExit):
            _resolve_spec_path("nonexistent_file.json")


if __name__ == "__main__":
    unittest.main()
