"""Unit tests for fusion/sim/network_simulator.py module."""

from unittest.mock import MagicMock, patch

import pytest

from fusion.sim.network_simulator import (
    NetworkSimulator,
    _validate_bandwidth_consistency,
    run,
)


@pytest.fixture
def sample_engine_props() -> dict:
    """Provide sample engine properties for tests."""
    return {
        "network": "test_net",
        "cores_per_link": 10,
        "holding_time": 5,
        "erlang_start": 100,
        "erlang_stop": 300,
        "erlang_step": 100,
        "max_iters": 1000,
        "mod_per_bw": {"100": "QPSK", "200": "16-QAM"},
        "request_distribution": {"100": 0.5, "200": 0.5},
        "thread_num": "s1",
        "date": "0101",
        "sim_start": "12_00_00_000000",
        "mod_assumption": "test_assumption",
        "mod_assumption_path": "data/mod_assumptions.json",
        "const_link_weight": 1,
        "is_only_core_node": True,
    }


class TestValidateBandwidthConsistency:
    """Tests for _validate_bandwidth_consistency function."""

    def test_validate_bandwidth_with_matching_config_passes(self) -> None:
        """Test that matching bandwidth configuration passes validation."""
        # Arrange
        engine_props = {
            "request_distribution": {"100": 0.5, "200": 0.5},
            "mod_per_bw": {"100": "QPSK", "200": "16-QAM"},
        }

        # Act & Assert (should not raise)
        _validate_bandwidth_consistency(engine_props)

    def test_validate_bandwidth_with_missing_bandwidth_raises_error(self) -> None:
        """Test that missing bandwidth in mod_per_bw raises ValueError."""
        # Arrange
        engine_props = {
            "request_distribution": {"100": 0.5, "200": 0.5, "400": 0.3},
            "mod_per_bw": {"100": "QPSK", "200": "16-QAM"},
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Bandwidth configuration mismatch"):
            _validate_bandwidth_consistency(engine_props)

    def test_validate_bandwidth_with_empty_mod_per_bw_raises_error(self) -> None:
        """Test that empty mod_per_bw raises ValueError."""
        # Arrange
        engine_props = {
            "request_distribution": {"100": 0.5},
            "mod_per_bw": {},
        }

        # Act & Assert
        with pytest.raises(ValueError, match="mod_per_bw is empty"):
            _validate_bandwidth_consistency(engine_props)

    def test_validate_bandwidth_without_request_distribution_passes(self) -> None:
        """Test that missing request_distribution passes validation."""
        # Arrange
        engine_props = {"mod_per_bw": {"100": "QPSK"}}

        # Act & Assert (should not raise)
        _validate_bandwidth_consistency(engine_props)


class TestNetworkSimulator:
    """Tests for NetworkSimulator class."""

    def test_network_simulator_initialization(self) -> None:
        """Test that NetworkSimulator initializes with empty properties."""
        # Act
        simulator = NetworkSimulator()

        # Assert
        assert simulator.properties == {}

    @patch("fusion.sim.network_simulator.SimulationEngine")
    @patch("fusion.sim.network_simulator.create_input")
    @patch("fusion.sim.network_simulator.save_input")
    def test_run_generic_sim_single_erlang_creates_engine(
        self,
        _mock_save_input: MagicMock,
        mock_create_input: MagicMock,
        mock_engine_class: MagicMock,
        sample_engine_props: dict,
    ) -> None:
        """Test that _run_generic_sim creates and runs simulation engine."""
        # Arrange
        simulator = NetworkSimulator()
        simulator.properties = sample_engine_props
        mock_create_input.return_value = {
            "mod_per_bw": {},
            "topology_info": {},
            "thread_num": "s1",
        }
        mock_engine = MagicMock()
        mock_engine.run.return_value = 100
        mock_engine_class.return_value = mock_engine
        progress_dict: dict[str, int] = {}

        # Act
        result = simulator._run_generic_sim(
            erlang=100.0,
            first_erlang=True,
            erlang_index=0,
            progress_dict=progress_dict,
            done_offset=0,
        )

        # Assert
        mock_engine_class.assert_called_once()
        mock_engine.run.assert_called_once()
        assert result == 100

    @patch("fusion.sim.network_simulator.Manager")
    @patch("fusion.sim.network_simulator.SimulationEngine")
    @patch("fusion.sim.network_simulator.create_input")
    @patch("fusion.sim.network_simulator.save_input")
    def test_run_generic_sim_multiple_erlangs_sequential(
        self,
        _mock_save_input: MagicMock,
        mock_create_input: MagicMock,
        mock_engine_class: MagicMock,
        mock_manager: MagicMock,
        sample_engine_props: dict,
    ) -> None:
        """Test that run_generic_sim runs multiple erlangs sequentially."""
        # Arrange
        simulator = NetworkSimulator()
        simulator.properties = sample_engine_props
        mock_create_input.return_value = {
            "mod_per_bw": {},
            "topology_info": {},
            "thread_num": "s1",
        }
        mock_engine = MagicMock()
        mock_engine.run.side_effect = [100, 200, 300]
        mock_engine_class.return_value = mock_engine

        mock_manager_inst = MagicMock()
        mock_manager_inst.dict.return_value = {}
        mock_manager.return_value = mock_manager_inst

        # Act
        simulator.run_generic_sim()

        # Assert
        # Should run for 3 erlangs (100-300 step 100 inclusive = [100, 200, 300])
        assert mock_engine.run.call_count == 3

    @patch("fusion.sim.network_simulator.run")
    def test_run_sim_sets_properties_and_calls_run_generic(
        self, mock_run: MagicMock, sample_engine_props: dict
    ) -> None:
        """Test that run_sim sets up properties and delegates to run_generic_sim."""
        # Arrange
        simulator = NetworkSimulator()
        kwargs = {
            "thread_params": sample_engine_props,
            "thread_num": "s1",
            "sim_start": "0101_12_00_00_000000",
        }

        # Act
        with patch.object(simulator, "run_generic_sim") as mock_run_generic:
            simulator.run_sim(**kwargs)

            # Assert
            assert simulator.properties["date"] == "0101"
            assert simulator.properties["sim_start"] == "12_00_00_000000"
            assert simulator.properties["thread_num"] == "s1"
            mock_run_generic.assert_called_once()


class TestRunFunction:
    """Tests for run function."""

    @patch("fusion.sim.network_simulator.Process")
    @patch("fusion.sim.network_simulator.logger")
    def test_run_creates_process_for_each_simulation(
        self, mock_logger: MagicMock, mock_process: MagicMock
    ) -> None:
        """Test that run creates a process for each simulation config."""
        # Arrange
        sims_dict = {
            "s1": {
                "network": "test_net",
                "sim_start": "12_00_00_000000",
                "date": "0101",
            },
            "s2": {
                "network": "test_net",
                "sim_start": "12_00_00_000000",
                "date": "0101",
            },
        }
        stop_flag = MagicMock()
        mock_proc = MagicMock()
        mock_process.return_value = mock_proc

        # Act
        run(sims_dict, stop_flag)

        # Assert
        assert mock_process.call_count == 2
        assert mock_proc.start.call_count == 2
        assert mock_proc.join.call_count == 2

    @patch("fusion.sim.network_simulator.Process")
    @patch("fusion.sim.network_simulator.datetime")
    def test_run_generates_sim_start_when_missing(
        self, mock_datetime: MagicMock, mock_process: MagicMock
    ) -> None:
        """Test that run generates sim_start when not provided."""
        # Arrange
        sims_dict = {"s1": {"network": "test_net"}}
        stop_flag = MagicMock()
        mock_datetime.now.return_value.strftime.return_value = "0101_12_00_00_000000"
        mock_proc = MagicMock()
        mock_process.return_value = mock_proc

        # Act
        run(sims_dict, stop_flag)

        # Assert
        mock_datetime.now.assert_called()

    @patch("fusion.sim.network_simulator.Process")
    def test_run_uses_existing_sim_start_when_provided(
        self, mock_process: MagicMock
    ) -> None:
        """Test that run uses existing sim_start and date when provided."""
        # Arrange
        sims_dict = {
            "s1": {
                "network": "test_net",
                "sim_start": "12_00_00_000000",
                "date": "0101",
            }
        }
        stop_flag = MagicMock()
        mock_proc = MagicMock()
        mock_process.return_value = mock_proc

        # Act
        run(sims_dict, stop_flag)

        # Assert
        # Check that NetworkSimulator.run_sim is called with correct sim_start
        call_kwargs = mock_process.call_args[1]["kwargs"]
        assert call_kwargs["sim_start"] == "0101_12_00_00_000000"
