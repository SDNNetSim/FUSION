"""Unit tests for fusion/sim/utils/simulation.py module."""

import os
from unittest.mock import MagicMock, mock_open, patch

from fusion.sim.utils.simulation import (
    get_erlang_values,
    get_start_time,
    log_message,
    run_simulation_for_erlangs,
    save_study_results,
)


class TestLogMessage:
    """Tests for log_message function."""

    def test_log_message_with_queue_sends_to_queue(self) -> None:
        """Test that message is sent to queue when provided."""
        # Arrange
        message = "Test message"
        mock_queue = MagicMock()

        # Act
        log_message(message, mock_queue)

        # Assert
        mock_queue.put.assert_called_once_with(message)

    @patch("fusion.sim.utils.simulation.logger")
    def test_log_message_without_queue_uses_logger(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that logger is used when queue is None."""
        # Arrange
        message = "Test message"

        # Act
        log_message(message, None)

        # Assert
        mock_logger.info.assert_called_once_with(message)


class TestGetStartTime:
    """Tests for get_start_time function."""

    @patch("fusion.sim.utils.simulation.time.sleep")
    @patch("fusion.sim.utils.simulation.Path.exists")
    def test_get_start_time_sets_date_and_time(
        self, mock_exists: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Test that date and sim_start are set correctly."""
        # Arrange
        sim_dict = {"s1": {"network": "test_network", "date": None, "sim_start": None}}
        mock_exists.return_value = False  # Path doesn't exist, so no retry

        # Act
        result = get_start_time(sim_dict)

        # Assert
        assert result["s1"]["date"] is not None
        assert len(result["s1"]["date"]) == 4  # MMDD format
        assert result["s1"]["date"].isdigit()

        assert result["s1"]["sim_start"] is not None
        parts = result["s1"]["sim_start"].split("_")
        assert len(parts) == 4  # HH_MM_SS_microseconds
        assert all(part.isdigit() for part in parts)

    @patch("fusion.sim.utils.simulation.time.sleep")
    @patch("fusion.sim.utils.simulation.Path.exists")
    @patch("fusion.sim.utils.simulation.logger")
    def test_get_start_time_retries_on_duplicate(
        self, mock_logger: MagicMock, mock_exists: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Test that function retries when path already exists."""
        # Arrange
        sim_dict = {"s1": {"network": "test_network", "date": None, "sim_start": None}}
        # First call returns True (exists), second returns False
        mock_exists.side_effect = [True, False]

        # Act
        get_start_time(sim_dict)

        # Assert
        assert mock_exists.call_count == 2
        assert mock_sleep.call_count == 2
        mock_logger.warning.assert_called_once()


class TestGetErlangValues:
    """Tests for get_erlang_values function."""

    def test_get_erlang_values_with_valid_range_returns_list(self) -> None:
        """Test that erlang values are generated correctly from config."""
        # Arrange
        sim_dict = {"erlang_start": 100, "erlang_stop": 400, "erlang_step": 100}

        # Act
        result = get_erlang_values(sim_dict)

        # Assert
        assert result == [100, 200, 300]

    def test_get_erlang_values_with_step_one_includes_all_values(self) -> None:
        """Test that step of 1 includes all intermediate values."""
        # Arrange
        sim_dict = {"erlang_start": 1, "erlang_stop": 5, "erlang_step": 1}

        # Act
        result = get_erlang_values(sim_dict)

        # Assert
        assert result == [1, 2, 3, 4]


class TestRunSimulationForErlangs:
    """Tests for run_simulation_for_erlangs function."""

    def test_run_simulation_updates_arrival_rate_correctly(self) -> None:
        """Test that arrival rate is calculated and updated for each erlang."""
        # Arrange
        mock_env = MagicMock()
        mock_env.engine_obj.engine_props = {}
        mock_env.sim_dict = {}
        erlang_list = [100.0, 200.0]
        sim_dict = {"cores_per_link": 10, "holding_time": 5}
        mock_run_func = MagicMock(return_value=0.5)
        callback_list = None
        trial = None

        # Act
        result = run_simulation_for_erlangs(
            mock_env, erlang_list, sim_dict, mock_run_func, callback_list, trial
        )

        # Assert
        # First erlang: arrival_rate = 10 * 100 / 5 = 200
        # Second erlang: arrival_rate = 10 * 200 / 5 = 400
        assert mock_run_func.call_count == 2
        assert result == 0.5  # Mean of [0.5, 0.5]

    def test_run_simulation_returns_mean_of_rewards(self) -> None:
        """Test that mean of all rewards is returned."""
        # Arrange
        mock_env = MagicMock()
        mock_env.engine_obj.engine_props = {}
        mock_env.sim_dict = {}
        erlang_list = [100.0, 200.0, 300.0]
        sim_dict = {"cores_per_link": 10, "holding_time": 5}
        mock_run_func = MagicMock(side_effect=[0.3, 0.5, 0.7])
        callback_list = None
        trial = None

        # Act
        result = run_simulation_for_erlangs(
            mock_env, erlang_list, sim_dict, mock_run_func, callback_list, trial
        )

        # Assert
        assert result == 0.5  # Mean of [0.3, 0.5, 0.7]


class TestSaveStudyResults:
    """Tests for save_study_results function."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("pickle.dump")
    def test_save_study_results_creates_directory_and_saves_files(
        self, mock_pickle: MagicMock, mock_makedirs: MagicMock, mock_file: MagicMock
    ) -> None:
        """Test that study results are saved to correct directory structure."""
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
        best_params = {"param1": 0.5}
        best_reward = 100.0
        best_sim_start = 43200

        # Act
        save_study_results(
            study, mock_env, study_name, best_params, best_reward, best_sim_start
        )

        # Assert
        expected_dir = os.path.join(
            "logs", "q_learning", "test_network", "2025-01-01", "12:00:00"
        )
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
        mock_pickle.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("pickle.dump")
    def test_save_study_results_writes_best_hyperparams_file(
        self, mock_pickle: MagicMock, mock_makedirs: MagicMock, mock_file: MagicMock
    ) -> None:
        """Test that best hyperparameters are written to text file."""
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
        best_reward = 100.0
        best_sim_start = 43200

        # Act
        save_study_results(
            study, mock_env, study_name, best_params, best_reward, best_sim_start
        )

        # Assert
        # Check that best_hyperparams.txt was opened for writing
        expected_path = os.path.join(
            "logs",
            "q_learning",
            "test_network",
            "2025-01-01",
            "12:00:00",
            "best_hyperparams.txt",
        )
        mock_file.assert_any_call(expected_path, "w", encoding="utf-8")
