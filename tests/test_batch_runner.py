"""
Tests for batch_runner orchestrator.
"""

# pylint: disable=protected-access,attribute-defined-outside-init

from unittest.mock import MagicMock, Mock, patch

import pytest

from fusion.sim.batch_runner import BatchRunner, run_batch_simulation


class TestBatchRunner:
    """Test cases for BatchRunner class."""

    def setup_method(self):
        """Set up test configuration."""
        self.test_config = {
            "s1": {
                "network": "NSFNet",
                "erlang": "300",
                "holding_time": 1.0,
                "sim_thread_erlangs": 10,
                "num_requests": 1000,
                "save_files": False,
            }
        }

    def test_initialization(self):
        """Test BatchRunner initialization."""
        runner = BatchRunner(self.test_config)
        assert runner.config == self.test_config
        assert runner.sim_start is not None
        assert runner.results == []

    def test_get_single_erlang_value(self):
        """Test parsing single Erlang value."""
        runner = BatchRunner(self.test_config)
        erlangs = runner._get_erlang_values({"erlangs": "300"})
        assert erlangs == [300.0]

    def test_get_multiple_erlang_values(self):
        """Test parsing multiple Erlang values."""
        runner = BatchRunner(self.test_config)

        # Comma-separated values
        erlangs = runner._get_erlang_values({"erlangs": "300, 600, 900"})
        assert erlangs == [300.0, 600.0, 900.0]

        # Range with step
        erlangs = runner._get_erlang_values({"erlangs": "300-900:300"})
        assert erlangs == [300.0, 600.0, 900.0]

        # Range with default step
        erlangs = runner._get_erlang_values({"erlangs": "100-300"})
        assert erlangs == [100.0, 200.0, 300.0]

    def test_validate_bandwidth_config_valid(self):
        """Test bandwidth validation with valid config."""
        runner = BatchRunner(self.test_config)
        sim_params = {
            "request_distribution": {"25": 0.5, "50": 0.5},
            "mod_per_bw": {"25": ["QPSK"], "50": ["16-QAM"]},
        }

        # Should not raise any exception
        runner._validate_bandwidth_config(sim_params)

    def test_validate_bandwidth_config_missing(self):
        """Test bandwidth validation with missing bandwidth."""
        runner = BatchRunner(self.test_config)
        sim_params = {
            "request_distribution": {"25": 0.5, "100": 0.5},
            "mod_per_bw": {"25": ["QPSK"]},
        }

        with pytest.raises(ValueError, match="Bandwidth mismatch"):
            runner._validate_bandwidth_config(sim_params)

    @patch("fusion.sim.batch_runner.create_input")
    def test_prepare_simulation(self, mock_create):
        """Test simulation preparation."""
        runner = BatchRunner(self.test_config)

        mock_create.return_value = {"test": "data", "mod_per_bw": {}}
        result = runner.prepare_simulation({"save_files": False})

        mock_create.assert_called_once()
        assert result == {"test": "data", "mod_per_bw": {}}

    @patch("fusion.sim.batch_runner.SimulationEngine")
    @patch("fusion.sim.batch_runner.log_message")
    def test_run_single_erlang(self, _mock_log, mock_engine_class):
        """Test running single Erlang simulation."""
        # Mock engine
        mock_engine = Mock()
        mock_engine.run.return_value = {"blocking": 0.01}
        mock_engine.stats_obj = Mock()
        mock_engine.stats_obj.__dict__ = {
            "blocking_probability": 0.01,
            "total_requests": 1000,
        }
        mock_engine_class.return_value = mock_engine

        runner = BatchRunner(self.test_config)
        runner.progress_dict = {"total": 100, "done": 0}

        result = runner.run_single_erlang(300.0, {"holding_time": 1.0}, 0, 1)

        assert result["erlang"] == 300.0
        assert result["results"] == {"blocking": 0.01}
        assert "elapsed_time" in result
        assert result["stats"]["blocking_probability"] == 0.01

    @patch("fusion.sim.batch_runner.multiprocessing.Pool")
    @patch("fusion.sim.batch_runner.log_message")
    def test_run_parallel_batch(self, _mock_log, mock_pool_class):
        """Test parallel batch execution."""
        # Mock pool
        mock_pool = MagicMock()
        mock_pool.starmap.return_value = [
            {"erlang": 300.0, "results": {}, "elapsed_time": 1.0, "stats": {}},
            {"erlang": 600.0, "results": {}, "elapsed_time": 1.5, "stats": {}},
        ]
        mock_pool_class.return_value.__enter__.return_value = mock_pool

        runner = BatchRunner(self.test_config)
        results = runner.run_parallel_batch([300.0, 600.0], {}, num_processes=2)

        assert len(results) == 2
        assert results[0]["erlang"] == 300.0
        assert results[1]["erlang"] == 600.0

    @patch("fusion.sim.batch_runner.BatchRunner.run_single_erlang")
    def test_run_sequential_batch(self, mock_run_single):
        """Test sequential batch execution."""
        mock_run_single.side_effect = [
            {"erlang": 300.0, "results": {}},
            {"erlang": 600.0, "results": {}},
        ]

        runner = BatchRunner(self.test_config)
        results = runner.run_sequential_batch([300.0, 600.0], {})

        assert len(results) == 2
        assert mock_run_single.call_count == 2

    @patch("fusion.sim.batch_runner.BatchRunner.prepare_simulation")
    @patch("fusion.sim.batch_runner.BatchRunner.run_sequential_batch")
    def test_run_complete(self, mock_seq_batch, mock_prepare):
        """Test complete run method."""
        mock_prepare.return_value = {"erlangs": "300,600", "sim_thread_erlangs": 10}
        mock_seq_batch.return_value = [
            {
                "erlang": 300.0,
                "results": {},
                "elapsed_time": 1.0,
                "stats": {"blocking_probability": 0.01},
            }
        ]

        runner = BatchRunner(self.test_config)
        results = runner.run(parallel=False)

        assert len(results) == 1
        assert runner.results == results

    @patch("fusion.sim.batch_runner.BatchRunner")
    def test_run_batch_simulation_function(self, mock_runner_class):
        """Test convenience function."""
        mock_runner = Mock()
        mock_runner.run.return_value = [{"test": "result"}]
        mock_runner_class.return_value = mock_runner

        result = run_batch_simulation({"test": "config"}, parallel=True)

        mock_runner_class.assert_called_once_with({"test": "config"})
        mock_runner.run.assert_called_once_with(parallel=True, num_processes=None)
        assert result == [{"test": "result"}]
