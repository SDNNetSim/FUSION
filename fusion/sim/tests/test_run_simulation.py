"""Unit tests for fusion/sim/run_simulation.py module."""

from unittest.mock import MagicMock, patch

from fusion.sim.run_simulation import run_simulation, run_simulation_pipeline


class TestRunSimulation:
    """Tests for run_simulation function."""

    @patch("fusion.sim.run_simulation.run_batch_simulation")
    def test_run_simulation_calls_batch_runner(self, mock_batch_runner: MagicMock) -> None:
        """Test that run_simulation delegates to run_batch_simulation."""
        # Arrange
        config_dict = {"s1": {"network": "test_net"}}
        mock_batch_runner.return_value = [{"erlang": 100, "results": {}}]

        # Act
        result = run_simulation(config_dict)

        # Assert
        mock_batch_runner.assert_called_once_with(config_dict, parallel=False)
        assert result == {"erlang": 100, "results": {}}

    @patch("fusion.sim.run_simulation.run_batch_simulation")
    def test_run_simulation_returns_none_when_no_results(self, mock_batch_runner: MagicMock) -> None:
        """Test that run_simulation returns None when results list is empty."""
        # Arrange
        config_dict = {"s1": {"network": "test_net"}}
        mock_batch_runner.return_value = []

        # Act
        result = run_simulation(config_dict)

        # Assert
        assert result is None

    @patch("fusion.sim.run_simulation.run_batch_simulation")
    def test_run_simulation_returns_first_result_for_compatibility(self, mock_batch_runner: MagicMock) -> None:
        """Test that run_simulation returns first result from list."""
        # Arrange
        config_dict = {"s1": {"network": "test_net"}}
        mock_batch_runner.return_value = [
            {"erlang": 100, "results": {}},
            {"erlang": 200, "results": {}},
        ]

        # Act
        result = run_simulation(config_dict)

        # Assert
        assert result == {"erlang": 100, "results": {}}


class TestRunSimulationPipeline:
    """Tests for run_simulation_pipeline function."""

    @patch("fusion.sim.run_simulation.run_batch_simulation")
    @patch("fusion.cli.config_setup.load_and_validate_config")
    def test_run_simulation_pipeline_loads_config_and_runs(self, mock_load_config: MagicMock, mock_batch_runner: MagicMock) -> None:
        """Test that pipeline loads config and executes simulation."""
        # Arrange
        mock_args = MagicMock()
        mock_args.parallel = False
        mock_args.num_processes = None
        mock_load_config.return_value = {"s1": {"network": "test_net"}}
        mock_batch_runner.return_value = [{"erlang": 100}]

        # Act
        result = run_simulation_pipeline(mock_args)

        # Assert
        mock_load_config.assert_called_once_with(mock_args)
        mock_batch_runner.assert_called_once_with({"s1": {"network": "test_net"}}, parallel=False, num_processes=None)
        assert result == [{"erlang": 100}]

    @patch("fusion.sim.run_simulation.run_batch_simulation")
    @patch("fusion.cli.config_setup.load_and_validate_config")
    def test_run_simulation_pipeline_handles_parallel_execution(self, mock_load_config: MagicMock, mock_batch_runner: MagicMock) -> None:
        """Test that pipeline passes parallel execution parameters."""
        # Arrange
        mock_args = MagicMock()
        mock_args.parallel = True
        mock_args.num_processes = 4
        mock_load_config.return_value = {"s1": {"network": "test_net"}}
        mock_batch_runner.return_value = []

        # Act
        run_simulation_pipeline(mock_args)

        # Assert
        mock_batch_runner.assert_called_once_with({"s1": {"network": "test_net"}}, parallel=True, num_processes=4)

    @patch("fusion.sim.run_simulation.run_batch_simulation")
    @patch("fusion.cli.config_setup.load_and_validate_config")
    def test_run_simulation_pipeline_defaults_to_sequential_when_no_parallel_attr(
        self, mock_load_config: MagicMock, mock_batch_runner: MagicMock
    ) -> None:
        """Test that pipeline defaults to sequential execution without parallel attr."""
        # Arrange
        mock_args = MagicMock(spec=[])  # No parallel or num_processes attributes
        mock_load_config.return_value = {"s1": {"network": "test_net"}}
        mock_batch_runner.return_value = []

        # Act
        run_simulation_pipeline(mock_args)

        # Assert
        mock_batch_runner.assert_called_once_with({"s1": {"network": "test_net"}}, parallel=False, num_processes=None)

    @patch("fusion.sim.run_simulation.run_batch_simulation")
    @patch("fusion.cli.config_setup.load_and_validate_config")
    def test_run_simulation_pipeline_ignores_stop_flag(self, mock_load_config: MagicMock, mock_batch_runner: MagicMock) -> None:
        """Test that stop_flag parameter is accepted but ignored."""
        # Arrange
        mock_args = MagicMock()
        mock_args.parallel = False
        mock_args.num_processes = None
        mock_load_config.return_value = {"s1": {"network": "test_net"}}
        mock_batch_runner.return_value = []
        mock_stop_flag = MagicMock()

        # Act
        result = run_simulation_pipeline(mock_args, stop_flag=mock_stop_flag)

        # Assert
        # stop_flag is accepted but not used in the new implementation
        assert result == []
