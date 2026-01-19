"""Unit tests for fusion.unity.submit_manifest module."""

import csv
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fusion.unity.constants import RESOURCE_KEYS
from fusion.unity.errors import ManifestNotFoundError
from fusion.unity.submit_manifest import (
    build_environment_variables,
    parse_cli,
    read_first_row,
)


class TestParseCli:
    """Tests for parse_cli function."""

    @patch("sys.argv", ["submit_manifest.py", "test_exp", "run_rl_sim.sh"])
    def test_parse_cli_with_basic_arguments_returns_namespace(self) -> None:
        """Test parse_cli parses basic required arguments correctly."""
        # Arrange & Act
        args = parse_cli()

        # Assert
        assert args.exp == "test_exp"
        assert args.script == "run_rl_sim.sh"
        assert args.rows is None

    @patch(
        "sys.argv",
        ["submit_manifest.py", "test_exp", "run_rl_sim.sh", "--rows", "10"],
    )
    def test_parse_cli_with_rows_argument_returns_namespace_with_rows(self) -> None:
        """Test parse_cli parses optional rows argument correctly."""
        # Arrange & Act
        args = parse_cli()

        # Assert
        assert args.exp == "test_exp"
        assert args.script == "run_rl_sim.sh"
        assert args.rows == 10


class TestReadFirstRow:
    """Tests for read_first_row function."""

    def test_read_first_row_with_valid_manifest_returns_data_and_count(self, tmp_path: Path) -> None:
        """Test read_first_row returns first row and total count."""
        # Arrange
        manifest_data = [
            {
                "run_id": "00001",
                "path_algorithm": "ppo",
                "erlang_start": "100",
                "network": "test_network",
                "partition": "gpu",
                "time": "24:00:00",
                "mem": "32G",
                "cpus": "8",
                "gpus": "1",
                "nodes": "1",
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
                "nodes": "1",
            },
        ]
        manifest_path = tmp_path / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = manifest_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_data)

        # Act
        first_row, total_rows = read_first_row(manifest_path)

        # Assert
        assert total_rows == 2
        assert first_row["run_id"] == "00001"
        assert first_row["path_algorithm"] == "ppo"
        assert first_row["network"] == "test_network"

    def test_read_first_row_with_empty_manifest_raises_error(self, tmp_path: Path) -> None:
        """Test read_first_row raises ManifestNotFoundError for empty manifest."""
        # Arrange
        manifest_path = tmp_path / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8") as f:
            f.write("run_id,algorithm\n")  # Header only

        # Act & Assert
        with pytest.raises(ManifestNotFoundError) as exc_info:
            read_first_row(manifest_path)

        assert "empty" in str(exc_info.value).lower()

    def test_read_first_row_with_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test read_first_row raises ManifestNotFoundError for missing file."""
        # Arrange
        manifest_path = tmp_path / "nonexistent_manifest.csv"

        # Act & Assert
        with pytest.raises(ManifestNotFoundError) as exc_info:
            read_first_row(manifest_path)

        assert "Cannot read manifest file" in str(exc_info.value)

    def test_read_first_row_with_single_row_returns_correct_count(self, tmp_path: Path) -> None:
        """Test read_first_row returns correct count for single-row manifest."""
        # Arrange
        manifest_data = [
            {
                "run_id": "00001",
                "path_algorithm": "ppo",
                "erlang_start": "100",
            }
        ]
        manifest_path = tmp_path / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = manifest_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_data)

        # Act
        first_row, total_rows = read_first_row(manifest_path)

        # Assert
        assert total_rows == 1
        assert first_row["run_id"] == "00001"


class TestBuildEnvironmentVariables:
    """Tests for build_environment_variables function."""

    def test_build_env_with_complete_data_returns_all_variables(self) -> None:
        """Test build_environment_variables creates all expected variables."""
        # Arrange
        first_row = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
            "network": "test_network",
            "partition": "gpu",
            "time": "24:00:00",
            "mem": "32G",
            "cpus": "8",
            "gpus": "1",
            "nodes": "1",
        }
        job_dir = Path("experiments/test_exp")
        exp = "test_exp"

        # Act
        env = build_environment_variables(first_row, 2, job_dir, exp)

        # Assert
        assert env["MANIFEST"] == "unity/experiments/test_exp/manifest.csv"
        assert env["N_JOBS"] == "1"  # 2 rows - 1 (0-indexed)
        assert env["JOB_DIR"] == "experiments/test_exp"
        assert env["NETWORK"] == "test_network"
        assert "JOB_NAME" in env
        assert env["PARTITION"] == "gpu"
        assert env["TIME"] == "24:00:00"
        assert env["MEM"] == "32G"
        assert env["CPUS"] == "8"
        assert env["GPUS"] == "1"
        assert env["NODES"] == "1"

    def test_build_env_with_date_in_experiment_name_extracts_date(self) -> None:
        """Test build_environment_variables extracts date from experiment name."""
        # Arrange
        first_row = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
            "network": "test_network",
        }
        job_dir = Path("experiments/0315/142530")
        exp = "0315_142530"

        # Act
        env = build_environment_variables(first_row, 5, job_dir, exp)

        # Assert
        assert env["DATE"] == "0315"

    def test_build_env_creates_properly_formatted_job_name(self) -> None:
        """Test build_environment_variables creates correct job name format."""
        # Arrange
        first_row = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
            "network": "test_network",
        }
        job_dir = Path("experiments/0315/142530")
        exp = "0315/142530"

        # Act
        env = build_environment_variables(first_row, 5, job_dir, exp)

        # Assert
        expected_job_name = "ppo_100_0315_142530"
        assert env["JOB_NAME"] == expected_job_name

    def test_build_env_with_missing_network_uses_empty_string(self) -> None:
        """Test build_environment_variables handles missing network field."""
        # Arrange
        first_row: dict[str, Any] = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
        }
        job_dir = Path("experiments/test_exp")
        exp = "test_exp"

        # Act
        env = build_environment_variables(first_row, 2, job_dir, exp)

        # Assert
        assert env["NETWORK"] == ""

    def test_build_env_with_missing_resources_excludes_them(self) -> None:
        """Test build_environment_variables excludes missing resource fields."""
        # Arrange
        first_row = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
            "network": "test_network",
        }
        job_dir = Path("experiments/test_exp")
        exp = "test_exp"

        # Act
        env = build_environment_variables(first_row, 1, job_dir, exp)

        # Assert
        for key in RESOURCE_KEYS:
            assert key.upper() not in env

    def test_build_env_with_empty_resource_values_skips_them(self) -> None:
        """Test build_environment_variables skips empty resource values."""
        # Arrange
        first_row = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
            "network": "test_network",
            "partition": "",
            "gpus": "",
            "time": "24:00:00",
            "mem": "32G",
        }
        job_dir = Path("experiments/test_exp")
        exp = "test_exp"

        # Act
        env = build_environment_variables(first_row, 2, job_dir, exp)

        # Assert
        assert "PARTITION" not in env
        assert "GPUS" not in env
        assert env["TIME"] == "24:00:00"
        assert env["MEM"] == "32G"

    def test_build_env_with_multiple_jobs_calculates_correct_n_jobs(self) -> None:
        """Test build_environment_variables calculates N_JOBS correctly."""
        # Arrange
        first_row = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
            "network": "test_network",
        }
        job_dir = Path("experiments/test_exp")
        exp = "test_exp"

        # Act
        env = build_environment_variables(first_row, 100, job_dir, exp)

        # Assert
        assert env["N_JOBS"] == "99"  # 100 rows - 1 (0-indexed)

    def test_build_env_resource_keys_are_uppercase(self) -> None:
        """Test build_environment_variables converts resource keys to uppercase."""
        # Arrange
        first_row = {
            "run_id": "00001",
            "path_algorithm": "ppo",
            "erlang_start": "100",
            "network": "test_network",
            "partition": "gpu",
            "time": "24:00:00",
            "mem": "32G",
            "cpus": "8",
            "gpus": "1",
            "nodes": "1",
        }
        job_dir = Path("experiments/test_exp")
        exp = "test_exp"

        # Act
        env = build_environment_variables(first_row, 2, job_dir, exp)

        # Assert
        assert "PARTITION" in env
        assert "TIME" in env
        assert "MEM" in env
        assert "CPUS" in env
        assert "GPUS" in env
        assert "NODES" in env


class TestResourceKeysConstant:
    """Tests for RESOURCE_KEYS constant."""

    def test_resource_keys_contains_expected_values(self) -> None:
        """Test RESOURCE_KEYS constant has all expected SLURM resource keys."""
        # Arrange
        expected_keys = {"partition", "time", "mem", "cpus", "gpus", "nodes"}

        # Act & Assert
        assert RESOURCE_KEYS == expected_keys
