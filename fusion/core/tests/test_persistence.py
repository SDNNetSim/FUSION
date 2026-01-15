"""
Unit tests for fusion.core.persistence module.

Tests the StatsPersistence class functionality including statistics
saving, loading, and file format handling.
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import mock_open, patch

import pytest

from ..persistence import StatsPersistence


class TestStatsPersistenceInitialization:
    """Tests for StatsPersistence initialization."""

    def test_init_with_valid_params_creates_instance(self) -> None:
        """Test successful initialization with valid parameters."""
        # Arrange
        engine_props = {"file_type": "json", "erlang": 100.0, "thread_num": "s1"}
        sim_info = "test_simulation"

        # Act
        persistence = StatsPersistence(engine_props, sim_info)

        # Assert
        assert persistence.engine_props == engine_props
        assert persistence.sim_info == sim_info

    def test_init_with_empty_props_creates_instance(self) -> None:
        """Test initialization with minimal engine properties."""
        # Arrange
        engine_props: dict[str, Any] = {}
        sim_info = "minimal_test"

        # Act
        persistence = StatsPersistence(engine_props, sim_info)

        # Assert
        assert persistence.engine_props == {}
        assert persistence.sim_info == sim_info


class TestStatsPersistenceSaveStats:
    """Tests for statistics saving functionality."""

    @pytest.fixture
    def sample_persistence(self) -> Any:
        """Provide sample StatsPersistence for tests."""
        engine_props = {
            "file_type": "json",
            "erlang": 100.0,
            "thread_num": "s1",
            "save_start_end_slots": True,
        }
        return StatsPersistence(engine_props, "test_sim")

    @pytest.fixture
    def sample_stats_data(self) -> Any:
        """Provide sample statistics data for tests."""
        stats_dict = {"simulation_time": 1000.0, "total_requests": 500}

        # Use SimpleNamespace so vars() works properly for _prepare_iteration_stats
        stats_props = SimpleNamespace(
            link_usage_dict={"link_1": 0.5, "link_2": 0.8},
            simulation_blocking_list=[0.1, 0.15, 0.12],
            simulation_bitrate_blocking_list=[0.05, 0.08, 0.06],
            modulations_used_dict={"QPSK": 100, "16QAM": 50},
            bandwidth_blocking_dict={"50GHz": 0.1, "100GHz": 0.2},
            number_of_transponders=10,
            request_id=500,
            start_slot_list=[1, 5, 10],
            end_slot_list=[4, 9, 14],
            transponders_list=[2, 3, 1],
            hops_list=[3, 4, 2],
            lengths_list=[100.5, 200.8, 150.2],
            route_times_list=[0.1, 0.15, 0.12],
            crosstalk_list=[0.5, None, 0.8],
            snr_list=[],
        )

        blocking_stats = {
            "block_mean": 0.12,
            "block_variance": 0.001,
            "block_ci": 0.02,
            "block_ci_percent": 0.05,
            "bit_rate_block_mean": 0.06,
            "bit_rate_block_variance": 0.0005,
            "bit_rate_block_ci": 0.01,
            "bit_rate_block_ci_percent": 0.03,
            "iteration": 5,
        }

        return stats_dict, stats_props, blocking_stats

    @patch("fusion.core.persistence.create_directory")
    @patch("fusion.core.persistence.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_stats_creates_json_file_successfully(
        self,
        mock_json_dump: Any,
        mock_file_open: Any,
        mock_exists: Any,
        mock_create_dir: Any,
        sample_persistence: Any,
        sample_stats_data: Any,
    ) -> None:
        """Test successful JSON file creation and statistics saving."""
        # Arrange
        stats_dict, stats_props, blocking_stats = sample_stats_data
        mock_exists.return_value = False

        # Act
        sample_persistence.save_stats(stats_dict, stats_props, blocking_stats)

        # Assert
        mock_create_dir.assert_called_once()
        mock_file_open.assert_called_once()
        mock_json_dump.assert_called_once()

        # Verify json.dump was called with expected structure
        call_args = mock_json_dump.call_args[0]
        saved_data = call_args[0]
        assert "blocking_mean" in saved_data
        assert "link_usage" in saved_data
        assert "iter_stats" in saved_data

    @patch("fusion.core.persistence.os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"existing": "data", "iter_stats": {"0": {"old": "data"}}}',
    )
    @patch("json.load")
    @patch("json.dump")
    def test_save_stats_preserves_existing_iter_stats(
        self,
        mock_json_dump: Any,
        mock_json_load: Any,
        mock_file_open: Any,
        mock_exists: Any,
        sample_persistence: Any,
        sample_stats_data: Any,
    ) -> None:
        """Test that existing iteration statistics are preserved."""
        # Arrange
        stats_dict, stats_props, blocking_stats = sample_stats_data
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "existing": "data",
            "iter_stats": {"0": {"old": "data"}},
        }

        # Act
        sample_persistence.save_stats(stats_dict, stats_props, blocking_stats)

        # Assert
        call_args = mock_json_dump.call_args[0]
        saved_data = call_args[0]
        assert saved_data["iter_stats"]["0"]["old"] == "data"
        assert 5 in saved_data["iter_stats"]  # New iteration data added

    def test_save_stats_with_csv_file_type_raises_not_implemented(
        self, sample_persistence: Any, sample_stats_data: Any
    ) -> None:
        """Test that CSV file type raises NotImplementedError."""
        # Arrange
        sample_persistence.engine_props["file_type"] = "csv"
        stats_dict, stats_props, blocking_stats = sample_stats_data

        # Act & Assert
        with pytest.raises(NotImplementedError, match="CSV output not yet implemented"):
            sample_persistence.save_stats(stats_dict, stats_props, blocking_stats)

    def test_save_stats_with_invalid_file_type_raises_not_implemented(
        self, sample_persistence: Any, sample_stats_data: Any
    ) -> None:
        """Test invalid file type raises NotImplementedError."""
        # Arrange
        sample_persistence.engine_props["file_type"] = "xml"
        stats_dict, stats_props, blocking_stats = sample_stats_data

        # Act & Assert
        with pytest.raises(NotImplementedError, match="Invalid file type: xml"):
            sample_persistence.save_stats(stats_dict, stats_props, blocking_stats)

    @patch("fusion.core.persistence.create_directory")
    @patch("fusion.core.persistence.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_stats_handles_missing_start_end_slots_when_disabled(
        self,
        mock_json_dump: Any,
        mock_file_open: Any,
        mock_exists: Any,
        mock_create_dir: Any,
        sample_persistence: Any,
        sample_stats_data: Any,
    ) -> None:
        """Test start/end slot handling when save_start_end_slots is False."""
        # Arrange
        sample_persistence.engine_props["save_start_end_slots"] = False
        stats_dict, stats_props, blocking_stats = sample_stats_data
        mock_exists.return_value = False

        # Act
        sample_persistence.save_stats(stats_dict, stats_props, blocking_stats)

        # Assert
        call_args = mock_json_dump.call_args[0]
        saved_data = call_args[0]
        iter_data = saved_data["iter_stats"][5]
        assert iter_data["start_slot_list"] == []
        assert iter_data["end_slot_list"] == []


class TestStatsPersistenceIterationStats:
    """Tests for iteration statistics preparation."""

    @pytest.fixture
    def persistence_for_iter_stats(self) -> Any:
        """Provide persistence instance for iteration stats testing."""
        engine_props = {"save_start_end_slots": True}
        return StatsPersistence(engine_props, "iter_test")

    def test_prepare_iteration_stats_calculates_list_statistics(
        self, persistence_for_iter_stats: Any
    ) -> None:
        """Test calculation of mean, min, max for list statistics."""
        # Arrange - use SimpleNamespace so vars() works properly
        stats_props = SimpleNamespace(
            transponders_list=[2, 4, 6],
            hops_list=[1, 3, 5],
            lengths_list=[100.0, 200.0, 300.0],
            route_times_list=[0.1, 0.2, 0.3],
            crosstalk_list=[0.5, 0.8, 1.0],
            snr_list=[],
            simulation_blocking_list=[],
            simulation_bitrate_blocking_list=[],
            modulations_used_dict={},
            bandwidth_blocking_dict={},
            number_of_transponders=0,
            request_id=0,
            start_slot_list=[],
            end_slot_list=[],
        )

        # Act
        result = persistence_for_iter_stats._prepare_iteration_stats(stats_props, 1)

        # Assert
        assert result["trans_mean"] == 4.0
        assert result["trans_min"] == 2.0
        assert result["trans_max"] == 6.0
        assert result["hops_mean"] == 3.0
        assert result["lengths_mean"] == 200.0
        assert result["route_times_mean"] == 0.2
        assert result["xt_mean"] == 0.77  # rounded to 2 decimal places

    def test_prepare_iteration_stats_handles_empty_lists(
        self, persistence_for_iter_stats: Any
    ) -> None:
        """Test handling of empty statistical lists."""
        # Arrange - use SimpleNamespace so vars() works properly
        stats_props = SimpleNamespace(
            transponders_list=[],
            hops_list=[],
            lengths_list=[],
            route_times_list=[],
            crosstalk_list=[],
            snr_list=[],
            simulation_blocking_list=[],
            simulation_bitrate_blocking_list=[],
            modulations_used_dict={},
            bandwidth_blocking_dict={},
            number_of_transponders=0,
            request_id=0,
            start_slot_list=[],
            end_slot_list=[],
        )

        # Act
        result = persistence_for_iter_stats._prepare_iteration_stats(stats_props, 1)

        # Assert
        assert result["trans_mean"] is None
        assert result["trans_min"] is None
        assert result["trans_max"] is None
        assert result["hops_mean"] is None

    def test_prepare_iteration_stats_handles_crosstalk_with_none_values(
        self, persistence_for_iter_stats: Any
    ) -> None:
        """Test crosstalk handling with None values converted to 0."""
        # Arrange - use SimpleNamespace so vars() works properly
        stats_props = SimpleNamespace(
            transponders_list=[],
            hops_list=[],
            lengths_list=[],
            route_times_list=[],
            crosstalk_list=[0.5, None, 0.8, None],
            snr_list=[],
            simulation_blocking_list=[],
            simulation_bitrate_blocking_list=[],
            modulations_used_dict={},
            bandwidth_blocking_dict={},
            number_of_transponders=0,
            request_id=0,
            start_slot_list=[],
            end_slot_list=[],
        )

        # Act
        result = persistence_for_iter_stats._prepare_iteration_stats(stats_props, 1)

        # Assert
        # Mean of [0.5, 0, 0.8, 0] = 0.325
        assert result["xt_mean"] == 0.33  # rounded to 2 decimal places

    def test_prepare_iteration_stats_maps_property_names_correctly(
        self, persistence_for_iter_stats: Any
    ) -> None:
        """Test correct mapping of property names to output keys."""
        # Arrange - use SimpleNamespace so vars() works properly
        stats_props = SimpleNamespace(
            simulation_blocking_list=[0.1, 0.2],
            simulation_bitrate_blocking_list=[0.05, 0.1],
            modulations_used_dict={"QPSK": 10},
            bandwidth_blocking_dict={"50GHz": 0.1},
            number_of_transponders=5,
            request_id=100,
            start_slot_list=[1, 5],
            end_slot_list=[4, 9],
            transponders_list=[],
            hops_list=[],
            lengths_list=[],
            route_times_list=[],
            crosstalk_list=[],
            snr_list=[],
        )

        # Act
        result = persistence_for_iter_stats._prepare_iteration_stats(stats_props, 1)

        # Assert
        assert result["sim_block_list"] == [0.1, 0.2]
        assert result["sim_br_block_list"] == [0.05, 0.1]
        assert result["mods_used_dict"] == {"QPSK": 10}
        assert result["block_bw_dict"] == {"50GHz": 0.1}
        assert result["num_trans"] == 5
        assert result["req_id"] == 100


class TestStatsPersistenceFileOperations:
    """Tests for file operation methods."""

    @pytest.fixture
    def file_persistence(self) -> Any:
        """Provide persistence instance for file operation testing."""
        engine_props = {"erlang": 150.0, "thread_num": "s2"}
        return StatsPersistence(engine_props, "file_test_sim")

    @patch("fusion.core.persistence.PROJECT_ROOT", "/test/root")
    def test_get_save_path_constructs_correct_path(self, file_persistence: Any) -> None:
        """Test save path construction with all parameters."""
        # Act
        path = file_persistence._get_save_path("custom_data")

        # Assert
        expected = "/test/root/custom_data/output/file_test_sim/s2/150.0_erlang.json"
        assert path == expected

    @patch("fusion.core.persistence.PROJECT_ROOT", "/test/root")
    def test_get_save_path_uses_default_base_path(self, file_persistence: Any) -> None:
        """Test save path construction with default base path."""
        # Act
        path = file_persistence._get_save_path()

        # Assert
        expected = "/test/root/data/output/file_test_sim/s2/150.0_erlang.json"
        assert path == expected

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"test": "data", "value": 42}',
    )
    @patch("json.load")
    def test_load_stats_reads_json_file_successfully(
        self, mock_json_load: Any, mock_file_open: Any
    ) -> None:
        """Test successful loading of JSON statistics file."""
        # Arrange
        persistence = StatsPersistence({}, "load_test")
        test_data = {"test": "data", "value": 42}
        mock_json_load.return_value = test_data
        file_path = "/path/to/stats.json"

        # Act
        result = persistence.load_stats(file_path)

        # Assert
        mock_file_open.assert_called_once_with(file_path, encoding="utf-8")
        mock_json_load.assert_called_once()
        assert result == test_data

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='["not", "a", "dict"]',
    )
    @patch("json.load")
    def test_load_stats_raises_error_for_non_dict_json(
        self, mock_json_load: Any, mock_file_open: Any
    ) -> None:
        """Test error handling when JSON file contains non-dictionary data."""
        # Arrange
        persistence = StatsPersistence({}, "error_test")
        mock_json_load.return_value = ["not", "a", "dict"]
        file_path = "/path/to/invalid.json"

        # Act & Assert
        with pytest.raises(
            ValueError, match="Expected dictionary in JSON file, got <class 'list'>"
        ):
            persistence.load_stats(file_path)

    def test_load_stats_raises_error_for_unsupported_format(self) -> None:
        """Test error handling for unsupported file formats."""
        # Arrange
        persistence = StatsPersistence({}, "format_test")
        file_path = "/path/to/stats.csv"

        # Act & Assert
        with pytest.raises(
            NotImplementedError, match="Loading from .*/stats.csv not supported"
        ):
            persistence.load_stats(file_path)


class TestStatsPersistenceEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_save_stats_handles_corrupted_existing_file(self) -> None:
        """Test handling of corrupted existing JSON file."""
        # Arrange
        persistence = StatsPersistence(
            {"file_type": "json", "erlang": 100.0, "thread_num": "s1"}, "corrupt_test"
        )
        stats_dict: dict[str, float | None] = {}
        # Use SimpleNamespace so vars() works properly
        stats_props = SimpleNamespace(
            link_usage_dict={},
            simulation_blocking_list=[],
            simulation_bitrate_blocking_list=[],
            modulations_used_dict={},
            bandwidth_blocking_dict={},
            number_of_transponders=0,
            request_id=0,
            start_slot_list=[],
            end_slot_list=[],
            transponders_list=[],
            hops_list=[],
            lengths_list=[],
            route_times_list=[],
            crosstalk_list=[],
            snr_list=[],
        )
        blocking_stats: dict[str, float | None] = {"iteration": 0.0}

        # Act & Assert - should not raise exception
        with (
            patch("fusion.core.persistence.os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="invalid json")),
            patch("json.load", side_effect=json.JSONDecodeError("msg", "doc", 0)),
            patch("json.dump"),
            patch("fusion.core.persistence.create_directory"),
        ):
            persistence.save_stats(stats_dict, stats_props, blocking_stats)

    @patch("fusion.core.persistence.os.path.exists")
    def test_save_stats_handles_file_access_error(self, mock_exists: Any) -> None:
        """Test handling of file access errors during existing file read."""
        # Arrange
        persistence = StatsPersistence(
            {"file_type": "json", "erlang": 100.0, "thread_num": "s1"},
            "access_error_test",
        )
        stats_dict: dict[str, float | None] = {}
        # Use SimpleNamespace so vars() works properly
        stats_props = SimpleNamespace(
            link_usage_dict={},
            simulation_blocking_list=[],
            simulation_bitrate_blocking_list=[],
            modulations_used_dict={},
            bandwidth_blocking_dict={},
            number_of_transponders=0,
            request_id=0,
            start_slot_list=[],
            end_slot_list=[],
            transponders_list=[],
            hops_list=[],
            lengths_list=[],
            route_times_list=[],
            crosstalk_list=[],
            snr_list=[],
        )
        blocking_stats: dict[str, float | None] = {"iteration": 0.0}

        mock_exists.return_value = True

        # Create a proper mock for the second call to open
        mock_file = mock_open()

        # Act & Assert - should handle gracefully and start fresh
        with (
            patch(
                "builtins.open",
                side_effect=[OSError("Permission denied"), mock_file.return_value],
            ),
            patch("json.dump"),
            patch("fusion.core.persistence.create_directory"),
        ):
            # Should not raise exception despite initial file access error
            persistence.save_stats(stats_dict, stats_props, blocking_stats)
