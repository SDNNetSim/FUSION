"""
Unit tests for fusion.unity.fetch_results module.

Tests the functionality for fetching results from remote cluster storage,
including path conversion, file syncing, and manifest parsing.
"""

import json
import pathlib
import tempfile
import unittest

from fusion.unity.fetch_results import (
    convert_output_path_to_input_path,
    extract_path_algorithm_from_input_directory,
    extract_topology_from_output_path,
    get_last_path_segments,
    iterate_runs_index_file,
)


class TestFetchResults(unittest.TestCase):
    """Test cases for fetch_results module functions."""

    def test_convert_output_path_to_input_path(self):
        """Test conversion from output path to input path."""
        output_path = pathlib.PurePosixPath("/data/output/topology1/experiment/s1")
        expected = pathlib.PurePosixPath("/data/input/topology1/experiment")

        result = convert_output_path_to_input_path(output_path)
        self.assertEqual(result, expected)

    def test_convert_output_path_to_input_path_no_output_dir(self):
        """Test convert_output_path_to_input_path raises ValueError when 'output' not in path."""
        invalid_path = pathlib.PurePosixPath("/data/results/topology1/experiment/s1")

        with self.assertRaises(ValueError):
            convert_output_path_to_input_path(invalid_path)

    def test_get_last_path_segments(self):
        """Test extraction of last n parts from path."""
        path = pathlib.PurePosixPath("/data/output/topology1/experiment/s1")

        result = get_last_path_segments(path, 3)
        expected = pathlib.PurePosixPath("topology1/experiment/s1")
        self.assertEqual(result, expected)

    def test_get_last_path_segments_more_than_available(self):
        """Test get_last_path_segments when n exceeds path length."""
        path = pathlib.PurePosixPath("/data/output")

        result = get_last_path_segments(path, 5)
        self.assertEqual(result, path)

    def test_extract_topology_from_output_path(self):
        """Test extraction of topology name from output path."""
        output_path = pathlib.PurePosixPath("/data/output/topology1/experiment/s1")

        result = extract_topology_from_output_path(output_path)
        self.assertEqual(result, "topology1")

    def test_extract_topology_from_output_path_no_output_dir(self):
        """Test extract_topology_from_output_path raises ValueError when 'output' not in path."""
        invalid_path = pathlib.PurePosixPath("/data/results/topology1/experiment/s1")

        with self.assertRaises(ValueError):
            extract_topology_from_output_path(invalid_path)

    def test_extract_path_algorithm_from_input_directory_success(self):
        """Test successful reading of path algorithm from JSON file."""
        test_data = {"path_algorithm": "shortest_path", "other_param": "value"}

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = pathlib.Path(temp_dir)
            test_file = input_dir / "sim_input_s1.json"

            with test_file.open("w", encoding="utf-8") as f:
                json.dump(test_data, f)

            result = extract_path_algorithm_from_input_directory(input_dir)
            self.assertEqual(result, "shortest_path")

    def test_extract_path_algorithm_from_input_directory_no_files(self):
        """Test extract_path_algorithm_from_input_directory returns None when no matching files found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = pathlib.Path(temp_dir)

            result = extract_path_algorithm_from_input_directory(input_dir)
            self.assertIsNone(result)

    def test_extract_path_algorithm_from_input_directory_missing_key(self):
        """Test extract_path_algorithm_from_input_directory returns None when path_algorithm key missing."""
        test_data = {"other_param": "value"}

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = pathlib.Path(temp_dir)
            test_file = input_dir / "sim_input_s1.json"

            with test_file.open("w", encoding="utf-8") as f:
                json.dump(test_data, f)

            result = extract_path_algorithm_from_input_directory(input_dir)
            self.assertIsNone(result)

    def test_extract_path_algorithm_from_input_directory_invalid_json(self):
        """Test extract_path_algorithm_from_input_directory handles invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = pathlib.Path(temp_dir)
            test_file = input_dir / "sim_input_s1.json"

            with test_file.open("w", encoding="utf-8") as f:
                f.write("invalid json content")

            result = extract_path_algorithm_from_input_directory(input_dir)
            self.assertIsNone(result)

    def test_iterate_runs_index_file(self):
        """Test iteration over index file entries."""
        test_data = [
            '{"path": "/data/output/topology1/experiment1/s1"}',
            '{"path": "/data/output/topology1/experiment2/s1"}',
            '{"path": "/data/output/topology2/experiment1/s1"}',
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            for line in test_data:
                f.write(line + "\n")
            f.flush()

            index_file = pathlib.Path(f.name)

        try:
            paths = list(iterate_runs_index_file(index_file))
            expected_paths = [
                pathlib.PurePosixPath("/data/output/topology1/experiment1/s1"),
                pathlib.PurePosixPath("/data/output/topology1/experiment2/s1"),
                pathlib.PurePosixPath("/data/output/topology2/experiment1/s1"),
            ]

            self.assertEqual(paths, expected_paths)
        finally:
            index_file.unlink()

    def test_iterate_runs_index_file_empty_lines(self):
        """Test iterate_runs_index_file skips empty lines."""
        test_data = [
            '{"path": "/data/output/topology1/experiment1/s1"}',
            "",
            '{"path": "/data/output/topology1/experiment2/s1"}',
            "   ",
            '{"path": "/data/output/topology2/experiment1/s1"}',
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            for line in test_data:
                f.write(line + "\n")
            f.flush()

            index_file = pathlib.Path(f.name)

        try:
            paths = list(iterate_runs_index_file(index_file))
            expected_paths = [
                pathlib.PurePosixPath("/data/output/topology1/experiment1/s1"),
                pathlib.PurePosixPath("/data/output/topology1/experiment2/s1"),
                pathlib.PurePosixPath("/data/output/topology2/experiment1/s1"),
            ]

            self.assertEqual(paths, expected_paths)
        finally:
            index_file.unlink()


if __name__ == "__main__":
    unittest.main()
