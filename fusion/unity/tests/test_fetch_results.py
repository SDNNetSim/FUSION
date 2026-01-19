"""Unit tests for fusion.unity.fetch_results module."""

import json
import subprocess
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, patch

import pytest

from fusion.unity.errors import RemotePathError
from fusion.unity.fetch_results import (
    _execute_command_with_delay,
    convert_output_to_input_path,
    extract_path_algorithm_from_input,
    extract_topology_from_path,
    get_last_path_segments,
    iterate_runs_index_file,
    synchronize_remote_directory,
    synchronize_remote_file,
    synchronize_simulation_logs,
)


class TestConvertOutputToInputPath:
    """Tests for convert_output_to_input_path function."""

    def test_convert_output_to_input_path_with_valid_path_returns_input_path(
        self,
    ) -> None:
        """Test conversion from output path to input path with valid path."""
        # Arrange
        output_path = PurePosixPath("/data/output/topology1/experiment/s1")
        expected = PurePosixPath("/data/input/topology1/experiment")

        # Act
        result = convert_output_to_input_path(output_path)

        # Assert
        assert result == expected

    def test_convert_output_to_input_path_with_nested_dirs_returns_correct_path(
        self,
    ) -> None:
        """Test conversion handles multiple nested directories correctly."""
        # Arrange
        output_path = PurePosixPath("/data/cluster/output/topology2/exp/subexp/run/s3")
        expected = PurePosixPath("/data/cluster/input/topology2/exp/subexp/run")

        # Act
        result = convert_output_to_input_path(output_path)

        # Assert
        assert result == expected

    def test_convert_output_to_input_path_without_output_dir_raises_error(
        self,
    ) -> None:
        """Test conversion raises RemotePathError when output not in path."""
        # Arrange
        invalid_path = PurePosixPath("/data/results/topology1/experiment/s1")

        # Act & Assert
        with pytest.raises(RemotePathError) as exc_info:
            convert_output_to_input_path(invalid_path)

        assert "output" in str(exc_info.value)

    def test_convert_output_to_input_path_with_output_at_end_removes_seed_folder(
        self,
    ) -> None:
        """Test conversion removes seed folder from end of path."""
        # Arrange
        output_path = PurePosixPath("/output/topology/s10")
        expected = PurePosixPath("/input/topology")

        # Act
        result = convert_output_to_input_path(output_path)

        # Assert
        assert result == expected


class TestGetLastPathSegments:
    """Tests for get_last_path_segments function."""

    def test_get_last_path_segments_with_valid_count_returns_segments(self) -> None:
        """Test extraction of last n segments from path."""
        # Arrange
        path = PurePosixPath("/data/output/topology1/experiment/s1")
        segment_count = 3
        expected = PurePosixPath("topology1/experiment/s1")

        # Act
        result = get_last_path_segments(path, segment_count)

        # Assert
        assert result == expected

    def test_get_last_path_segments_with_count_exceeding_length_returns_full_path(
        self,
    ) -> None:
        """Test extraction when count exceeds path length returns full path."""
        # Arrange
        path = PurePosixPath("/data/output")

        # Act
        result = get_last_path_segments(path, 10)

        # Assert
        assert result == path

    def test_get_last_path_segments_with_zero_count_returns_empty_path(self) -> None:
        """Test extraction with zero count returns empty path."""
        # Arrange
        path = PurePosixPath("/data/output/topology")

        # Act
        result = get_last_path_segments(path, 0)

        # Assert
        assert result == PurePosixPath()

    def test_get_last_path_segments_with_negative_count_returns_empty_path(
        self,
    ) -> None:
        """Test extraction with negative count returns empty path."""
        # Arrange
        path = PurePosixPath("/data/output/topology")

        # Act
        result = get_last_path_segments(path, -1)

        # Assert
        assert result == PurePosixPath()

    def test_get_last_path_segments_with_single_segment_returns_correct_result(
        self,
    ) -> None:
        """Test extraction of single segment from path."""
        # Arrange
        path = PurePosixPath("/data/output/topology/experiment/s1")

        # Act
        result = get_last_path_segments(path, 1)

        # Assert
        assert result == PurePosixPath("s1")


class TestExtractTopologyFromPath:
    """Tests for extract_topology_from_path function."""

    def test_extract_topology_from_path_with_valid_path_returns_topology(
        self,
    ) -> None:
        """Test extraction of topology name from valid output path."""
        # Arrange
        output_path = PurePosixPath("/data/output/topology1/experiment/s1")

        # Act
        result = extract_topology_from_path(output_path)

        # Assert
        assert result == "topology1"

    def test_extract_topology_from_path_without_output_dir_raises_error(self) -> None:
        """Test extraction raises RemotePathError when output not in path."""
        # Arrange
        invalid_path = PurePosixPath("/data/results/topology1/experiment/s1")

        # Act & Assert
        with pytest.raises(RemotePathError) as exc_info:
            extract_topology_from_path(invalid_path)

        assert "output" in str(exc_info.value)

    def test_extract_topology_from_path_with_output_at_end_raises_error(self) -> None:
        """Test extraction raises error when no topology after output."""
        # Arrange
        invalid_path = PurePosixPath("/data/output")

        # Act & Assert
        with pytest.raises(RemotePathError) as exc_info:
            extract_topology_from_path(invalid_path)

        assert "No topology directory found" in str(exc_info.value)

    def test_extract_topology_from_path_with_different_topology_returns_correct_value(
        self,
    ) -> None:
        """Test extraction returns correct topology for different names."""
        # Arrange
        output_path = PurePosixPath("/cluster/output/nsfnet/run/s2")

        # Act
        result = extract_topology_from_path(output_path)

        # Assert
        assert result == "nsfnet"


class TestExecuteCommandWithDelay:
    """Tests for _execute_command_with_delay function."""

    @patch("fusion.unity.fetch_results.sleep")
    @patch("fusion.unity.fetch_results.subprocess.run")
    def test_execute_command_with_delay_in_normal_mode_executes_command(self, mock_run: Mock, mock_sleep: Mock) -> None:
        """Test command execution in normal mode calls subprocess."""
        # Arrange
        command = ["rsync", "-avP", "source", "dest"]
        is_dry_run = False

        # Act
        _execute_command_with_delay(command, is_dry_run)

        # Assert
        mock_sleep.assert_called_once()
        mock_run.assert_called_once_with(command, check=True)

    @patch("fusion.unity.fetch_results.sleep")
    @patch("fusion.unity.fetch_results.subprocess.run")
    def test_execute_command_with_delay_in_dry_run_mode_skips_execution(self, mock_run: Mock, mock_sleep: Mock) -> None:
        """Test command execution in dry run mode skips subprocess call."""
        # Arrange
        command = ["rsync", "-avP", "source", "dest"]
        is_dry_run = True

        # Act
        _execute_command_with_delay(command, is_dry_run)

        # Assert
        mock_sleep.assert_called_once()
        mock_run.assert_not_called()

    @patch("fusion.unity.fetch_results.sleep")
    @patch("fusion.unity.fetch_results.subprocess.run")
    def test_execute_command_with_delay_with_failing_command_raises_error(self, mock_run: Mock, mock_sleep: Mock) -> None:
        """Test command execution raises error when subprocess fails."""
        # Arrange
        command = ["rsync", "-avP", "source", "dest"]
        mock_run.side_effect = subprocess.CalledProcessError(1, command)

        # Act & Assert
        with pytest.raises(subprocess.CalledProcessError):
            _execute_command_with_delay(command, False)


class TestSynchronizeRemoteDirectory:
    """Tests for synchronize_remote_directory function."""

    @patch("fusion.unity.fetch_results._execute_command_with_delay")
    @patch("fusion.unity.fetch_results.Path.mkdir")
    def test_synchronize_remote_directory_with_valid_paths_creates_local_directory(self, mock_mkdir: Mock, mock_execute: Mock) -> None:
        """Test directory synchronization creates local target directory."""
        # Arrange
        remote_root = "user@cluster:/work/"
        absolute_path = PurePosixPath("/work/data/output/topology1/exp1/s1")
        dest_root = Path("/local/data")
        is_dry_run = False

        # Act
        synchronize_remote_directory(remote_root, absolute_path, dest_root, is_dry_run)

        # Assert
        mock_mkdir.assert_called_once()
        mock_execute.assert_called_once()

    @patch("fusion.unity.fetch_results._execute_command_with_delay")
    @patch("fusion.unity.fetch_results.Path.mkdir")
    def test_synchronize_remote_directory_with_dry_run_calls_execute_correctly(self, mock_mkdir: Mock, mock_execute: Mock) -> None:
        """Test directory synchronization in dry run mode."""
        # Arrange
        remote_root = "user@cluster:/work/"
        absolute_path = PurePosixPath("/work/data/output/topology1/exp1/s1")
        dest_root = Path("/local/data")
        is_dry_run = True

        # Act
        synchronize_remote_directory(remote_root, absolute_path, dest_root, is_dry_run)

        # Assert
        call_args = mock_execute.call_args
        assert call_args[0][1] is True  # is_dry_run parameter


class TestSynchronizeRemoteFile:
    """Tests for synchronize_remote_file function."""

    @patch("fusion.unity.fetch_results._execute_command_with_delay")
    @patch("fusion.unity.fetch_results.Path.mkdir")
    def test_synchronize_remote_file_with_valid_paths_creates_parent_directory(self, mock_mkdir: Mock, mock_execute: Mock) -> None:
        """Test file synchronization creates parent directory."""
        # Arrange
        remote_root = "user@cluster:/work/"
        remote_file = PurePosixPath("/work/data/file.json")
        local_file = Path("/local/data/file.json")
        is_dry_run = False

        # Act
        synchronize_remote_file(remote_root, remote_file, local_file, is_dry_run)

        # Assert
        mock_mkdir.assert_called_once()
        mock_execute.assert_called_once()


class TestSynchronizeSimulationLogs:
    """Tests for synchronize_simulation_logs function."""

    @patch("fusion.unity.fetch_results._execute_command_with_delay")
    @patch("fusion.unity.fetch_results.Path.mkdir")
    def test_synchronize_simulation_logs_with_valid_params_creates_directory(self, mock_mkdir: Mock, mock_execute: Mock) -> None:
        """Test log synchronization creates local directory structure."""
        # Arrange
        remote_logs_root = "user@cluster:/logs/"
        algorithm = "ppo"
        topology = "nsfnet"
        date_timestamp = PurePosixPath("2024/01/15")
        dest_root = Path("/local/logs")
        is_dry_run = False

        # Act
        synchronize_simulation_logs(
            remote_logs_root,
            algorithm,
            topology,
            date_timestamp,
            dest_root,
            is_dry_run,
        )

        # Assert
        mock_mkdir.assert_called_once()
        mock_execute.assert_called_once()


class TestExtractPathAlgorithmFromInput:
    """Tests for extract_path_algorithm_from_input function."""

    def test_extract_path_algorithm_from_input_with_valid_file_returns_algorithm(self, tmp_path: Path) -> None:
        """Test successful extraction of path algorithm from JSON file."""
        # Arrange
        test_data = {"path_algorithm": "shortest_path", "other_param": "value"}
        input_dir = tmp_path
        test_file = input_dir / "sim_input_s1.json"
        test_file.write_text(json.dumps(test_data), encoding="utf-8")

        # Act
        result = extract_path_algorithm_from_input(input_dir)

        # Assert
        assert result == "shortest_path"

    def test_extract_path_algorithm_from_input_with_no_files_returns_none(self, tmp_path: Path) -> None:
        """Test extraction returns None when no matching files found."""
        # Arrange
        input_dir = tmp_path

        # Act
        result = extract_path_algorithm_from_input(input_dir)

        # Assert
        assert result is None

    def test_extract_path_algorithm_from_input_with_missing_key_returns_none(self, tmp_path: Path) -> None:
        """Test extraction returns None when path_algorithm key missing."""
        # Arrange
        test_data = {"other_param": "value"}
        input_dir = tmp_path
        test_file = input_dir / "sim_input_s1.json"
        test_file.write_text(json.dumps(test_data), encoding="utf-8")

        # Act
        result = extract_path_algorithm_from_input(input_dir)

        # Assert
        assert result is None

    def test_extract_path_algorithm_from_input_with_invalid_json_returns_none(self, tmp_path: Path) -> None:
        """Test extraction handles invalid JSON gracefully."""
        # Arrange
        input_dir = tmp_path
        test_file = input_dir / "sim_input_s1.json"
        test_file.write_text("invalid json content", encoding="utf-8")

        # Act
        result = extract_path_algorithm_from_input(input_dir)

        # Assert
        assert result is None

    def test_extract_path_algorithm_from_input_with_multiple_files_returns_first_valid(self, tmp_path: Path) -> None:
        """Test extraction returns first valid algorithm when multiple files exist."""
        # Arrange
        input_dir = tmp_path
        test_file1 = input_dir / "sim_input_s1.json"
        test_file2 = input_dir / "sim_input_s2.json"
        test_file1.write_text(json.dumps({"path_algorithm": "ppo"}), encoding="utf-8")
        test_file2.write_text(json.dumps({"path_algorithm": "dqn"}), encoding="utf-8")

        # Act
        result = extract_path_algorithm_from_input(input_dir)

        # Assert
        assert result in ["ppo", "dqn"]  # Either is valid


class TestIterateRunsIndexFile:
    """Tests for iterate_runs_index_file function."""

    def test_iterate_runs_index_file_with_valid_entries_yields_paths(self, tmp_path: Path) -> None:
        """Test iteration over index file with valid entries."""
        # Arrange
        test_data = [
            '{"path": "/data/output/topology1/experiment1/s1"}',
            '{"path": "/data/output/topology1/experiment2/s1"}',
            '{"path": "/data/output/topology2/experiment1/s1"}',
        ]
        index_file = tmp_path / "runs_index.json"
        index_file.write_text("\n".join(test_data), encoding="utf-8")
        expected_paths = [
            PurePosixPath("/data/output/topology1/experiment1/s1"),
            PurePosixPath("/data/output/topology1/experiment2/s1"),
            PurePosixPath("/data/output/topology2/experiment1/s1"),
        ]

        # Act
        result = list(iterate_runs_index_file(index_file))

        # Assert
        assert result == expected_paths

    def test_iterate_runs_index_file_with_empty_lines_skips_them(self, tmp_path: Path) -> None:
        """Test iteration skips empty lines in index file."""
        # Arrange
        test_data = [
            '{"path": "/data/output/topology1/experiment1/s1"}',
            "",
            '{"path": "/data/output/topology1/experiment2/s1"}',
            "   ",
            '{"path": "/data/output/topology2/experiment1/s1"}',
        ]
        index_file = tmp_path / "runs_index.json"
        index_file.write_text("\n".join(test_data), encoding="utf-8")
        expected_paths = [
            PurePosixPath("/data/output/topology1/experiment1/s1"),
            PurePosixPath("/data/output/topology1/experiment2/s1"),
            PurePosixPath("/data/output/topology2/experiment1/s1"),
        ]

        # Act
        result = list(iterate_runs_index_file(index_file))

        # Assert
        assert result == expected_paths

    def test_iterate_runs_index_file_with_invalid_json_continues_iteration(self, tmp_path: Path) -> None:
        """Test iteration continues when encountering invalid JSON."""
        # Arrange
        test_data = [
            '{"path": "/data/output/topology1/experiment1/s1"}',
            "invalid json",
            '{"path": "/data/output/topology2/experiment1/s1"}',
        ]
        index_file = tmp_path / "runs_index.json"
        index_file.write_text("\n".join(test_data), encoding="utf-8")
        expected_paths = [
            PurePosixPath("/data/output/topology1/experiment1/s1"),
            PurePosixPath("/data/output/topology2/experiment1/s1"),
        ]

        # Act
        result = list(iterate_runs_index_file(index_file))

        # Assert
        assert result == expected_paths

    def test_iterate_runs_index_file_with_missing_path_key_continues_iteration(self, tmp_path: Path) -> None:
        """Test iteration continues when path key is missing."""
        # Arrange
        test_data = [
            '{"path": "/data/output/topology1/experiment1/s1"}',
            '{"other_key": "value"}',
            '{"path": "/data/output/topology2/experiment1/s1"}',
        ]
        index_file = tmp_path / "runs_index.json"
        index_file.write_text("\n".join(test_data), encoding="utf-8")
        expected_paths = [
            PurePosixPath("/data/output/topology1/experiment1/s1"),
            PurePosixPath("/data/output/topology2/experiment1/s1"),
        ]

        # Act
        result = list(iterate_runs_index_file(index_file))

        # Assert
        assert result == expected_paths
