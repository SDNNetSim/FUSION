"""Unit tests for fusion.modules.rl.algorithms.persistence module."""

from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from fusion.modules.rl.algorithms.persistence import (
    BanditModelPersistence,
    QLearningModelPersistence,
)
from fusion.modules.rl.errors import AlgorithmNotFoundError


class TestBanditModelPersistence:
    """Tests for BanditModelPersistence class."""

    def test_load_model_reads_json_file(self) -> None:
        """Test that load_model correctly reads and parses JSON file."""
        # Arrange
        mock_data = '{"state1": [1.0, 2.0], "state2": [3.0, 4.0]}'
        expected = {"state1": [1.0, 2.0], "state2": [3.0, 4.0]}

        with patch("builtins.open", mock_open(read_data=mock_data)):
            # Act
            result = BanditModelPersistence.load_model("test_model.json")

            # Assert
            assert result == expected

    def test_load_model_constructs_correct_path(self) -> None:
        """Test that load_model constructs path with logs directory."""
        # Arrange
        mock_data = "{}"

        with patch("builtins.open", mock_open(read_data=mock_data)) as m:
            # Act
            BanditModelPersistence.load_model("subdir/model.json")

            # Assert
            m.assert_called_once()
            call_args = m.call_args[0][0]
            assert str(call_args) == str(Path("logs") / "subdir/model.json")

    def test_save_model_returns_early_if_none(self) -> None:
        """Test that save_model returns early when state_values_dict is None."""
        # Act & Assert - should not raise any errors
        BanditModelPersistence.save_model(
            state_values_dict=None,
            erlang=30.0,
            cores_per_link=4,
            save_dir="test_dir",
            is_path=True,
            trial=0,
        )

    def test_save_model_converts_tuples_to_strings(self) -> None:
        """Test that save_model converts tuple keys to strings."""
        # Arrange
        state_values = {(0, 1): np.array([1.0, 2.0]), (1, 2): np.array([3.0, 4.0])}

        with patch("builtins.open", mock_open()), patch("json.dump") as mock_dump:
            # Act
            BanditModelPersistence.save_model(
                state_values_dict=state_values,
                erlang=30.0,
                cores_per_link=4,
                save_dir="test_dir",
                is_path=True,
                trial=0,
            )

            # Assert
            mock_dump.assert_called_once()
            saved_data = mock_dump.call_args[0][0]
            assert "(0, 1)" in saved_data
            assert "(1, 2)" in saved_data
            assert saved_data["(0, 1)"] == [1.0, 2.0]

    def test_save_model_creates_correct_filename_for_path(self) -> None:
        """Test that save_model creates correct filename for path agent."""
        # Arrange
        state_values = {(0, 1): np.array([1.0])}

        with patch("builtins.open", mock_open()) as m, patch("json.dump"):
            # Act
            BanditModelPersistence.save_model(
                state_values_dict=state_values,
                erlang=60.0,
                cores_per_link=4,
                save_dir="test_dir",
                is_path=True,
                trial=2,
            )

            # Assert
            call_path = m.call_args[0][0]
            assert "state_vals_e60.0_routes_c4_t3.json" in str(call_path)

    def test_save_model_raises_error_for_core_agent(self) -> None:
        """Test that save_model raises error for core agent (not implemented)."""
        # Arrange
        state_values = {(0, 1): np.array([1.0])}

        # Act & Assert
        with pytest.raises(AlgorithmNotFoundError, match="Core agent bandit model"):
            BanditModelPersistence.save_model(
                state_values_dict=state_values,
                erlang=30.0,
                cores_per_link=4,
                save_dir="test_dir",
                is_path=False,
                trial=0,
            )


class TestQLearningModelPersistence:
    """Tests for QLearningModelPersistence class."""

    def test_save_model_saves_numpy_array(self) -> None:
        """Test that save_model saves rewards as numpy array."""
        # Arrange
        q_dict = {"state1": [1.0, 2.0]}
        rewards_avg = np.array([10.0, 20.0, 30.0])

        with (
            patch("numpy.save") as mock_npsave,
            patch("builtins.open", mock_open()),
            patch("json.dump"),
        ):
            # Act
            QLearningModelPersistence.save_model(
                q_dict=q_dict,
                rewards_avg=rewards_avg,
                erlang=30.0,
                cores_per_link=4,
                base_str="routes",
                trial=0,
                iteration=5,
                save_dir="test_dir",
            )

            # Assert
            mock_npsave.assert_called_once()
            call_args = mock_npsave.call_args
            np.testing.assert_array_equal(call_args[0][1], rewards_avg)

    def test_save_model_creates_correct_filenames(self) -> None:
        """Test that save_model creates correct filenames for both files."""
        # Arrange
        q_dict = {"state1": [1.0]}
        rewards_avg = np.array([10.0])

        with (
            patch("numpy.save") as mock_npsave,
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump"),
        ):
            # Act
            QLearningModelPersistence.save_model(
                q_dict=q_dict,
                rewards_avg=rewards_avg,
                erlang=60.0,
                cores_per_link=4,
                base_str="routes",
                trial=1,
                iteration=10,
                save_dir="test_dir",
            )

            # Assert
            npy_path = mock_npsave.call_args[0][0]
            json_path = mock_file.call_args[0][0]

            assert "rewards_e60.0_routes_c4_t2_iter_10.npy" in str(npy_path)
            assert "state_vals_e60.0_routes_c4_t2.json" in str(json_path)

    def test_save_model_dumps_q_dict_as_json(self) -> None:
        """Test that save_model dumps Q-dict as JSON."""
        # Arrange
        q_dict = {"(0, 1)": [1.0, 2.0, 3.0]}
        rewards_avg = np.array([10.0])

        with (
            patch("numpy.save"),
            patch("builtins.open", mock_open()),
            patch("json.dump") as mock_dump,
        ):
            # Act
            QLearningModelPersistence.save_model(
                q_dict=q_dict,
                rewards_avg=rewards_avg,
                erlang=30.0,
                cores_per_link=4,
                base_str="routes",
                trial=0,
                iteration=1,
                save_dir="test_dir",
            )

            # Assert
            mock_dump.assert_called_once()
            saved_data = mock_dump.call_args[0][0]
            assert saved_data == q_dict

    def test_save_model_raises_error_for_cores(self) -> None:
        """Test that save_model raises error for core models (not implemented)."""
        # Arrange
        q_dict = {"state1": [1.0]}
        rewards_avg = np.array([10.0])

        # Act & Assert
        with pytest.raises(AlgorithmNotFoundError, match="Core Q-learning model"):
            QLearningModelPersistence.save_model(
                q_dict=q_dict,
                rewards_avg=rewards_avg,
                erlang=30.0,
                cores_per_link=4,
                base_str="cores",
                trial=0,
                iteration=1,
                save_dir="test_dir",
            )

    def test_save_model_uses_correct_save_directory(self) -> None:
        """Test that save_model uses provided save_dir correctly."""
        # Arrange
        q_dict = {"state1": [1.0]}
        rewards_avg = np.array([10.0])
        save_dir = "custom/save/dir"

        with (
            patch("numpy.save") as mock_npsave,
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump"),
        ):
            # Act
            QLearningModelPersistence.save_model(
                q_dict=q_dict,
                rewards_avg=rewards_avg,
                erlang=30.0,
                cores_per_link=4,
                base_str="routes",
                trial=0,
                iteration=1,
                save_dir=save_dir,
            )

            # Assert
            npy_path = mock_npsave.call_args[0][0]
            json_path = mock_file.call_args[0][0]

            assert str(npy_path).startswith(save_dir)
            assert str(json_path).startswith(save_dir)
