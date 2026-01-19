"""Unit tests for fusion.sim.utils.simulation module."""

import os
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from ..simulation import (
    get_erlang_values,
    get_start_time,
    log_message,
    run_simulation_for_erlangs,
    save_study_results,
)


class TestLogging:
    """Tests for logging utilities."""

    def test_log_message_with_queue_puts_message(self) -> None:
        """Test logging message to queue."""
        # Arrange
        message = "Test message"
        log_queue = MagicMock()

        # Act
        log_message(message, log_queue)

        # Assert
        log_queue.put.assert_called_once_with(message)

    @patch("fusion.sim.utils.simulation.logger")
    def test_log_message_without_queue_uses_logger(self, mock_logger: MagicMock) -> None:
        """Test logging message to logger when queue is None.

        :param mock_logger: Mock logger
        :type mock_logger: MagicMock
        """
        # Arrange
        message = "Test message"
        log_queue = None

        # Act
        log_message(message, log_queue)

        # Assert
        mock_logger.info.assert_called_once_with(message)

    @patch("fusion.sim.utils.simulation.logger")
    def test_log_message_with_false_queue_uses_logger(self, mock_logger: MagicMock) -> None:
        """Test logging with falsy queue value uses logger.

        :param mock_logger: Mock logger
        :type mock_logger: MagicMock
        """
        # Arrange
        message = "Another test"
        log_queue = 0  # Falsy value

        # Act
        log_message(message, log_queue)

        # Assert
        mock_logger.info.assert_called_once_with(message)


class TestSimulationStartTime:
    """Tests for simulation start time generation."""

    @patch("fusion.sim.utils.simulation.Path.exists")
    @patch("fusion.sim.utils.simulation.time.sleep")
    @patch("fusion.sim.utils.simulation.datetime")
    def test_get_start_time_generates_unique_timestamp(
        self,
        mock_datetime: MagicMock,
        mock_sleep: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        """Test generation of unique start time.

        :param mock_datetime: Mock datetime
        :type mock_datetime: MagicMock
        :param mock_sleep: Mock sleep
        :type mock_sleep: MagicMock
        :param mock_exists: Mock Path.exists
        :type mock_exists: MagicMock
        """
        # Arrange
        mock_datetime.now.return_value.strftime.return_value = "0101_12_00_00_123456"
        mock_exists.return_value = False
        sim_dict = {"s1": {"network": "test_network", "date": None, "sim_start": None}}

        # Act
        result = get_start_time(sim_dict)

        # Assert
        assert result["s1"]["date"] == "0101"
        assert result["s1"]["sim_start"] == "12_00_00_123456"
        mock_sleep.assert_called_once_with(0.1)

    @patch("fusion.sim.utils.simulation.Path.exists")
    @patch("fusion.sim.utils.simulation.time.sleep")
    @patch("fusion.sim.utils.simulation.datetime")
    @patch("fusion.sim.utils.simulation.logger")
    def test_get_start_time_retries_on_duplicate(
        self,
        mock_logger: MagicMock,
        mock_datetime: MagicMock,
        mock_sleep: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        """Test that duplicate timestamps trigger retry.

        :param mock_logger: Mock logger
        :type mock_logger: MagicMock
        :param mock_datetime: Mock datetime
        :type mock_datetime: MagicMock
        :param mock_sleep: Mock sleep
        :type mock_sleep: MagicMock
        :param mock_exists: Mock Path.exists
        :type mock_exists: MagicMock
        """
        # Arrange
        mock_datetime.now.return_value.strftime.side_effect = [
            "0101_12_00_00_111111",
            "0101_12_00_00_222222",
        ]
        mock_exists.side_effect = [True, False]
        sim_dict = {"s1": {"network": "test_network", "date": None, "sim_start": None}}

        # Act
        result = get_start_time(sim_dict)

        # Assert
        assert result["s1"]["date"] == "0101"
        assert result["s1"]["sim_start"] == "12_00_00_222222"
        mock_logger.warning.assert_called_once()
        assert mock_sleep.call_count == 2

    @patch("fusion.sim.utils.simulation.Path.exists")
    @patch("fusion.sim.utils.simulation.time.sleep")
    @patch("fusion.sim.utils.simulation.datetime")
    def test_get_start_time_creates_correct_path(
        self,
        mock_datetime: MagicMock,
        mock_sleep: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        """Test that correct path is checked for existence.

        :param mock_datetime: Mock datetime
        :type mock_datetime: MagicMock
        :param mock_sleep: Mock sleep
        :type mock_sleep: MagicMock
        :param mock_exists: Mock Path.exists
        :type mock_exists: MagicMock
        """
        # Arrange
        mock_datetime.now.return_value.strftime.return_value = "1225_10_30_45_678901"
        mock_exists.return_value = False
        sim_dict = {"s1": {"network": "my_network", "date": None, "sim_start": None}}

        # Act
        get_start_time(sim_dict)

        # Assert
        mock_exists.assert_called_once()


class TestErlangValues:
    """Tests for Erlang value generation."""

    def test_get_erlang_values_with_valid_range_returns_list(self) -> None:
        """Test generating Erlang values from valid range."""
        # Arrange
        sim_dict = {"erlang_start": 10, "erlang_stop": 50, "erlang_step": 10}
        expected = [10, 20, 30, 40]

        # Act
        result = get_erlang_values(sim_dict)

        # Assert
        assert result == expected

    def test_get_erlang_values_with_step_one_returns_all_values(self) -> None:
        """Test generating Erlang values with step size 1."""
        # Arrange
        sim_dict = {"erlang_start": 1, "erlang_stop": 5, "erlang_step": 1}
        expected = [1, 2, 3, 4]

        # Act
        result = get_erlang_values(sim_dict)

        # Assert
        assert result == expected

    def test_get_erlang_values_with_large_step_returns_sparse_list(self) -> None:
        """Test generating Erlang values with large step size."""
        # Arrange
        sim_dict = {"erlang_start": 0, "erlang_stop": 100, "erlang_step": 25}
        expected = [0, 25, 50, 75]

        # Act
        result = get_erlang_values(sim_dict)

        # Assert
        assert result == expected

    def test_get_erlang_values_with_string_numbers_converts_correctly(self) -> None:
        """Test that string numbers are converted to integers."""
        # Arrange
        sim_dict = {"erlang_start": "5", "erlang_stop": "15", "erlang_step": "5"}
        expected = [5, 10]

        # Act
        result = get_erlang_values(sim_dict)

        # Assert
        assert result == expected

    def test_get_erlang_values_with_equal_start_stop_returns_empty(self) -> None:
        """Test that equal start and stop returns empty list."""
        # Arrange
        sim_dict = {"erlang_start": 10, "erlang_stop": 10, "erlang_step": 5}
        expected: list = []

        # Act
        result = get_erlang_values(sim_dict)

        # Assert
        assert result == expected


class TestSimulationExecution:
    """Tests for simulation execution functions."""

    def test_run_simulation_for_erlangs_executes_for_each_value(self) -> None:
        """Test running simulation for multiple Erlang values."""
        # Arrange
        mock_env = MagicMock()
        mock_env.engine_obj.engine_props = {}
        mock_env.sim_dict = {"cores_per_link": 7, "holding_time": 2}
        erlang_list = [10.0, 20.0, 30.0]
        mock_run_func = MagicMock(side_effect=[100, 200, 300])
        callback_list: list[MagicMock] = []
        trial = None

        # Act
        result = run_simulation_for_erlangs(
            mock_env,
            erlang_list,
            mock_env.sim_dict,
            mock_run_func,
            callback_list,
            trial,
        )

        # Assert
        assert result == 200.0  # Mean of [100, 200, 300]
        assert mock_run_func.call_count == 3

    def test_run_simulation_for_erlangs_sets_arrival_rate_correctly(self) -> None:
        """Test that arrival rate is calculated correctly."""
        # Arrange
        mock_env = MagicMock()
        mock_env.engine_obj.engine_props = {}
        mock_env.sim_dict = {"cores_per_link": 10, "holding_time": 5}
        erlang_list = [20.0]
        mock_run_func = MagicMock(return_value=100)

        # Act
        run_simulation_for_erlangs(mock_env, erlang_list, mock_env.sim_dict, mock_run_func, [], None)

        # Assert
        assert mock_env.engine_obj.engine_props["erlang"] == 20.0
        assert mock_env.engine_obj.engine_props["arrival_rate"] == 40.0  # (10*20)/5

    def test_run_simulation_for_erlangs_with_empty_list_returns_nan(self) -> None:
        """Test running simulation with empty Erlang list."""
        # Arrange
        mock_env = MagicMock()
        mock_env.engine_obj.engine_props = {}
        mock_env.sim_dict = {"cores_per_link": 7, "holding_time": 2}
        erlang_list: list = []
        mock_run_func = MagicMock()

        # Act
        result = run_simulation_for_erlangs(mock_env, erlang_list, mock_env.sim_dict, mock_run_func, [], None)

        # Assert
        assert np.isnan(result)
        mock_run_func.assert_not_called()


class TestStudyResultsSaving:
    """Tests for saving study results."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("pickle.dump")
    def test_save_study_results_creates_directory_and_files(
        self,
        mock_pickle_dump: MagicMock,
        mock_makedirs: MagicMock,
        mock_file: Any,
    ) -> None:
        """Test that study results are saved correctly.

        :param mock_pickle_dump: Mock pickle.dump
        :type mock_pickle_dump: MagicMock
        :param mock_makedirs: Mock os.makedirs
        :type mock_makedirs: MagicMock
        :param mock_file: Mock file operations
        :type mock_file: mock_open
        """
        # Arrange
        mock_env = MagicMock()
        mock_env.engine_obj.engine_props = {
            "network": "test_network",
            "date": "2025-01-01",
            "sim_start": "12:00:00",
            "path_algorithm": "q_learning",
        }
        study = MagicMock()
        study_name = "test_study.pkl"
        best_params: dict[str, Any] = {"param1": 0.5, "param2": 10}
        best_reward = 100.0
        best_sim_start = 0

        # Act
        save_study_results(study, mock_env, study_name, best_params, best_reward, best_sim_start)

        # Assert
        expected_dir = os.path.join("logs", "q_learning", "test_network", "2025-01-01", "12:00:00")
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
        mock_pickle_dump.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("pickle.dump")
    def test_save_study_results_writes_hyperparameters_file(
        self,
        mock_pickle_dump: MagicMock,
        mock_makedirs: MagicMock,
        mock_file: Any,
    ) -> None:
        """Test that hyperparameters file is written correctly.

        :param mock_pickle_dump: Mock pickle.dump
        :type mock_pickle_dump: MagicMock
        :param mock_makedirs: Mock os.makedirs
        :type mock_makedirs: MagicMock
        :param mock_file: Mock file operations
        :type mock_file: mock_open
        """
        # Arrange
        mock_env = MagicMock()
        mock_env.engine_obj.engine_props = {
            "network": "test_network",
            "date": "2025-01-01",
            "sim_start": "12:00:00",
            "path_algorithm": "q_learning",
        }
        study = MagicMock()
        study_name = "test_study.pkl"
        best_params = {"learning_rate": 0.001, "batch_size": 32}
        best_reward = 150.5
        best_sim_start = 123456

        # Act
        save_study_results(study, mock_env, study_name, best_params, best_reward, best_sim_start)

        # Assert
        expected_hyperparams_path = os.path.join(
            "logs",
            "q_learning",
            "test_network",
            "2025-01-01",
            "12:00:00",
            "best_hyperparams.txt",
        )
        mock_file.assert_any_call(expected_hyperparams_path, "w", encoding="utf-8")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("pickle.dump")
    def test_save_study_results_with_different_algorithm_creates_correct_path(
        self,
        mock_pickle_dump: MagicMock,
        mock_makedirs: MagicMock,
        mock_file: Any,
    ) -> None:
        """Test path creation with different algorithm.

        :param mock_pickle_dump: Mock pickle.dump
        :type mock_pickle_dump: MagicMock
        :param mock_makedirs: Mock os.makedirs
        :type mock_makedirs: MagicMock
        :param mock_file: Mock file operations
        :type mock_file: mock_open
        """
        # Arrange
        mock_env = MagicMock()
        mock_env.engine_obj.engine_props = {
            "network": "my_network",
            "date": "2025-02-15",
            "sim_start": "08:30:00",
            "path_algorithm": "dijkstra",
        }
        study = MagicMock()
        study_name = "study.pkl"
        best_params: dict[str, Any] = {}
        best_reward = 0.0
        best_sim_start = 0

        # Act
        save_study_results(study, mock_env, study_name, best_params, best_reward, best_sim_start)

        # Assert
        expected_dir = os.path.join("logs", "dijkstra", "my_network", "2025-02-15", "08:30:00")
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
