"""
Unit tests for fusion.core.simulation module.

Tests the SimulationEngine class functionality including initialization,
topology creation, request handling, and simulation execution.
"""

import signal
from typing import Any
from unittest.mock import Mock, patch

import networkx as nx
import pytest

from ..simulation import SimulationEngine


class TestSimulationEngineInitialization:
    """Tests for SimulationEngine initialization."""

    @pytest.fixture
    def basic_engine_props(self) -> dict[str, Any]:
        """Provide basic engine properties for testing."""
        return {
            "network": "test_network",
            "date": "2024-01-01",
            "sim_start": "test_sim",
            "output_train_data": False,
            "stop_flag": None,
        }

    def test_init_creates_simulation_engine_with_required_components(
        self, basic_engine_props: dict[str, Any]
    ) -> None:
        """Test successful initialization with all required components."""
        # Act
        engine = SimulationEngine(basic_engine_props)

        # Assert
        assert engine.engine_props == basic_engine_props
        assert engine.network_spectrum_dict == {}
        assert engine.reqs_dict is None
        assert engine.reqs_status_dict == {}
        assert engine.iteration == 0
        assert isinstance(engine.topology, nx.Graph)
        assert engine.sim_info == "test_network/2024-01-01/test_sim"
        assert engine.sdn_obj is not None
        assert engine.stats_obj is not None
        assert engine.reporter is not None
        assert engine.persistence is not None
        assert engine.ml_metrics is None  # output_train_data is False

    def test_init_with_ml_metrics_enabled_creates_ml_collector(
        self, basic_engine_props: dict[str, Any]
    ) -> None:
        """Test initialization with ML metrics collection enabled."""
        # Arrange
        basic_engine_props["output_train_data"] = True

        # Act
        engine = SimulationEngine(basic_engine_props)

        # Assert
        assert engine.ml_metrics is not None

    def test_init_with_stop_flag_sets_flag_correctly(
        self, basic_engine_props: dict[str, Any]
    ) -> None:
        """Test initialization with stop flag."""
        # Arrange
        stop_flag = Mock()
        basic_engine_props["stop_flag"] = stop_flag

        # Act
        engine = SimulationEngine(basic_engine_props)

        # Assert
        assert engine.stop_flag == stop_flag


class TestSimulationEngineTopologyCreation:
    """Tests for topology creation functionality."""

    @pytest.fixture
    def topology_engine_props(self) -> dict[str, Any]:
        """Provide engine properties with topology information."""
        return {
            "network": "test",
            "date": "2024-01-01",
            "sim_start": "topo_test",
            "output_train_data": False,
            "topology_info": {
                "nodes": {
                    "A": {"type": "core"},
                    "B": {"type": "edge"},
                    "C": {"type": "core"},
                },
                "links": {
                    "1": {
                        "source": "A",
                        "destination": "B",
                        "length": 100.0,
                        "fiber": {"num_cores": 4},
                    },
                    "2": {
                        "source": "B",
                        "destination": "C",
                        "length": 150.0,
                        "fiber": {"num_cores": 4},
                    },
                },
            },
            "c_band": 80,
            "l_band": 0,
            "s_band": 0,
            "o_band": 0,
            "e_band": 0,
        }

    def test_create_topology_adds_nodes_from_topology_info(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test that create_topology adds all nodes from topology_info."""
        # Arrange
        engine = SimulationEngine(topology_engine_props)

        # Act
        engine.create_topology()

        # Assert
        assert list(engine.topology.nodes()) == ["A", "B", "C"]

    def test_create_topology_creates_bidirectional_links(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test that create_topology creates bidirectional network links."""
        # Arrange
        engine = SimulationEngine(topology_engine_props)

        # Act
        engine.create_topology()

        # Assert
        assert engine.topology.has_edge("A", "B")
        assert engine.topology.has_edge("B", "C")
        assert ("A", "B") in engine.network_spectrum_dict
        assert ("B", "A") in engine.network_spectrum_dict
        assert ("B", "C") in engine.network_spectrum_dict
        assert ("C", "B") in engine.network_spectrum_dict

    def test_create_topology_initializes_spectrum_matrices(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test spectrum matrices initialization for each link."""
        # Arrange
        engine = SimulationEngine(topology_engine_props)

        # Act
        engine.create_topology()

        # Assert
        link_data = engine.network_spectrum_dict[("A", "B")]
        assert "cores_matrix" in link_data
        assert "c" in link_data["cores_matrix"]
        # 4 cores, 80 slots
        assert link_data["cores_matrix"]["c"].shape == (4, 80)
        assert link_data["link_num"] == 1
        assert link_data["usage_count"] == 0
        assert link_data["throughput"] == 0

    def test_create_topology_handles_multiple_bands(self) -> None:
        """Test topology creation with multiple enabled bands."""
        # Arrange
        engine_props = {
            "network": "test",
            "date": "2024-01-01",
            "sim_start": "multi_band",
            "output_train_data": False,
            "topology_info": {
                "nodes": {"A": {}, "B": {}},
                "links": {
                    "1": {
                        "source": "A",
                        "destination": "B",
                        "length": 100.0,
                        "fiber": {"num_cores": 2},
                    }
                },
            },
            "c_band": 80,
            "l_band": 60,
            "s_band": 0,
            "o_band": 0,
            "e_band": 0,
        }
        engine = SimulationEngine(engine_props)

        # Act
        engine.create_topology()

        # Assert
        link_data = engine.network_spectrum_dict[("A", "B")]
        assert "c" in link_data["cores_matrix"]
        assert "l" in link_data["cores_matrix"]
        assert "s" not in link_data["cores_matrix"]
        assert link_data["cores_matrix"]["c"].shape == (2, 80)
        assert link_data["cores_matrix"]["l"].shape == (2, 60)

    def test_create_topology_updates_engine_props_and_components(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test that create_topology updates engine properties and components."""
        # Arrange
        engine = SimulationEngine(topology_engine_props)

        # Act
        engine.create_topology()

        # Assert
        assert "topology" in engine.engine_props
        assert engine.engine_props["topology"] == engine.topology
        assert "band_list" in engine.engine_props
        assert engine.engine_props["band_list"] == ["c"]
        assert engine.stats_obj.topology == engine.topology
        assert engine.sdn_obj.sdn_props.topology == engine.topology


class TestSimulationEngineRequestHandling:
    """Tests for request handling functionality."""

    @pytest.fixture
    def engine_with_requests(self) -> SimulationEngine:
        """Provide engine with sample requests for testing."""
        engine_props = {
            "network": "test",
            "date": "2024-01-01",
            "sim_start": "req_test",
            "output_train_data": False,
            "save_snapshots": False,
        }
        engine = SimulationEngine(engine_props)
        engine.reqs_dict = {
            1.0: {
                "req_id": 1,
                "source": "A",
                "destination": "B",
                "arrive": 1.0,
                "depart": 5.0,
                "request_type": "arrival",
                "bandwidth": "100GHz",
                "mod_formats": {"QPSK": {"max_length": [200]}},
            },
            5.0: {
                "req_id": 1,
                "source": "A",
                "destination": "B",
                "arrive": 1.0,
                "depart": 5.0,
                "request_type": "release",
                "bandwidth": "100GHz",
                "mod_formats": {"QPSK": {"max_length": [200]}},
            },
        }
        engine.reqs_status_dict = {
            1: {
                "mod_format": ["QPSK"],
                "path": ["A", "B"],
                "is_sliced": False,
                "was_routed": True,
                "core_list": [0],
                "band": ["c"],
                "start_slot_list": [10],
                "end_slot_list": [20],
                "bandwidth_list": ["100GHz"],
                "snr_cost": [0.5],
            }
        }
        return engine

    def test_handle_arrival_updates_sdn_controller_with_request_data(
        self, engine_with_requests: SimulationEngine
    ) -> None:
        """Test that handle_arrival updates SDN controller with request parameters."""
        # Arrange
        current_time = 1.0
        engine_with_requests.sdn_obj = Mock()
        engine_with_requests.sdn_obj.sdn_props = Mock()
        engine_with_requests.sdn_obj.sdn_props.network_spectrum_dict = {}
        engine_with_requests.sdn_obj.sdn_props.was_routed = True
        engine_with_requests.sdn_obj.sdn_props.number_of_transponders = 2
        engine_with_requests.stats_obj = Mock()
        engine_with_requests.stats_obj.iter_update = Mock()

        # Act
        engine_with_requests.handle_arrival(current_time)

        # Assert
        assert engine_with_requests.sdn_obj.sdn_props.update_params.called
        assert engine_with_requests.sdn_obj.handle_event.called

    def test_handle_arrival_with_forced_parameters_passes_to_sdn(
        self, engine_with_requests: SimulationEngine
    ) -> None:
        """Test handle_arrival with forced routing parameters."""
        # Arrange
        current_time = 1.0
        force_route_matrix = [["A", "B"]]
        force_core = 1
        force_slicing = True
        forced_index = 5
        force_mod_format = "16QAM"

        engine_with_requests.sdn_obj = Mock()
        engine_with_requests.sdn_obj.sdn_props = Mock()
        engine_with_requests.sdn_obj.sdn_props.network_spectrum_dict = {}
        engine_with_requests.sdn_obj.sdn_props.was_routed = False
        engine_with_requests.stats_obj = Mock()
        engine_with_requests.stats_obj.iter_update = Mock()

        # Act
        engine_with_requests.handle_arrival(
            current_time=current_time,
            force_route_matrix=force_route_matrix,
            force_core=force_core,
            force_slicing=force_slicing,
            forced_index=forced_index,
            force_mod_format=force_mod_format,
        )

        # Assert
        call_args = engine_with_requests.sdn_obj.handle_event.call_args
        assert call_args[1]["force_route_matrix"] == force_route_matrix
        assert call_args[1]["force_core"] == force_core
        assert call_args[1]["force_slicing"] == force_slicing
        assert call_args[1]["forced_index"] == forced_index
        assert call_args[1]["force_mod_format"] == force_mod_format

    def test_handle_release_calls_sdn_controller_for_teardown(
        self, engine_with_requests: SimulationEngine
    ) -> None:
        """Test that handle_release calls SDN controller for resource teardown."""
        # Arrange
        current_time = 5.0
        engine_with_requests.sdn_obj = Mock()
        engine_with_requests.sdn_obj.sdn_props = Mock()
        engine_with_requests.sdn_obj.sdn_props.network_spectrum_dict = {}

        # Act
        engine_with_requests.handle_release(current_time)

        # Assert
        assert engine_with_requests.sdn_obj.handle_event.called
        call_args = engine_with_requests.sdn_obj.handle_event.call_args
        assert call_args[1]["request_type"] == "release"

    def test_handle_release_with_missing_request_id_handles_gracefully(
        self, engine_with_requests: SimulationEngine
    ) -> None:
        """Test handle_release when request ID not in status dict."""
        # Arrange
        current_time = 5.0
        assert engine_with_requests.reqs_dict is not None
        engine_with_requests.reqs_dict[5.0]["req_id"] = 999  # Non-existent ID
        engine_with_requests.sdn_obj.sdn_props = Mock()

        # Act & Assert - should not raise exception
        engine_with_requests.handle_release(current_time)

    def test_handle_request_processes_arrival_type_correctly(
        self, engine_with_requests: SimulationEngine
    ) -> None:
        """Test handle_request processes arrival request type."""
        # Arrange
        current_time = 1.0
        request_number = 1
        engine_with_requests.engine_props["save_snapshots"] = False
        engine_with_requests.engine_props["output_train_data"] = False

        with patch.object(engine_with_requests, "handle_arrival") as mock_arrival:
            # Act
            engine_with_requests.handle_request(current_time, request_number)

            # Assert
            mock_arrival.assert_called_once_with(current_time=current_time)

    def test_handle_request_processes_release_type_correctly(
        self, engine_with_requests: SimulationEngine
    ) -> None:
        """Test handle_request processes release request type."""
        # Arrange
        current_time = 5.0
        request_number = 1

        # Act & Assert
        with patch.object(engine_with_requests, "handle_release") as mock_release:
            engine_with_requests.handle_request(current_time, request_number)
            mock_release.assert_called_once_with(current_time=current_time)

    def test_handle_request_with_invalid_type_raises_error(
        self, engine_with_requests: SimulationEngine
    ) -> None:
        """Test handle_request raises error for invalid request type."""
        # Arrange
        assert engine_with_requests.reqs_dict is not None
        engine_with_requests.reqs_dict[1.0]["request_type"] = "invalid"
        current_time = 1.0
        request_number = 1

        # Act & Assert
        with pytest.raises(NotImplementedError, match="Request type unrecognized"):
            engine_with_requests.handle_request(current_time, request_number)


class TestSimulationEngineIterationManagement:
    """Tests for simulation iteration management."""

    @pytest.fixture
    def iteration_engine(self) -> SimulationEngine:
        """Provide engine for iteration testing."""
        engine_props = {
            "network": "test",
            "date": "2024-01-01",
            "sim_start": "iter_test",
            "output_train_data": False,
            "max_iters": 10,
            "print_step": 1,
            "save_step": 5,
            "erlang": 100.0,
            "thread_num": "s1",
            "deploy_model": False,
            "seeds": None,
        }
        engine = SimulationEngine(engine_props)
        engine.stats_obj = Mock()
        engine.stats_obj.stats_props = Mock()
        engine.stats_obj.stats_props.simulation_blocking_list = [0.1, 0.15]
        # Explicitly create Mock objects for methods
        init_mock = Mock()
        calc_block_mock = Mock()
        finalize_mock = Mock()
        calc_conf_mock = Mock(return_value=False)
        engine.stats_obj.init_iter_stats = init_mock
        engine.stats_obj.calculate_blocking_statistics = calc_block_mock
        engine.stats_obj.finalize_iteration_statistics = finalize_mock
        engine.stats_obj.calculate_confidence_interval = calc_conf_mock
        return engine

    def test_init_iter_initializes_iteration_state(
        self, iteration_engine: SimulationEngine
    ) -> None:
        """Test init_iter properly initializes iteration state."""
        # Arrange
        iteration = 5
        seed = 42
        iteration_engine.network_spectrum_dict = {
            ("A", "B"): {"usage_count": 10, "throughput": 1000}
        }

        with patch.object(iteration_engine, "generate_requests"):
            # Act
            iteration_engine.init_iter(iteration, seed)

            # Assert
            assert iteration_engine.iteration == 5
            assert iteration_engine.stats_obj.iteration == 5
            iteration_engine.stats_obj.init_iter_stats.assert_called()  # type: ignore
            # Check network spectrum reset
            link_data = iteration_engine.network_spectrum_dict[("A", "B")]
            assert link_data["usage_count"] == 0
            assert link_data["throughput"] == 0

    def test_init_iter_with_trial_updates_thread_num(
        self, iteration_engine: SimulationEngine
    ) -> None:
        """Test init_iter updates thread_num when trial is provided."""
        # Arrange
        iteration = 0
        trial = 2

        with patch.object(iteration_engine, "generate_requests"):
            # Act
            iteration_engine.init_iter(iteration, trial=trial)

            # Assert
            assert iteration_engine.engine_props["thread_num"] == "s3"  # trial + 1

    @patch("fusion.core.simulation.load_model")
    def test_init_iter_loads_ml_model_when_deploy_model_enabled(
        self, mock_load_model: Mock, iteration_engine: SimulationEngine
    ) -> None:
        """Test init_iter loads ML model when deploy_model is True."""
        # Arrange
        iteration_engine.engine_props["deploy_model"] = True
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        with patch.object(iteration_engine, "generate_requests"):
            # Act
            iteration_engine.init_iter(0, print_flag=True)

            # Assert
            mock_load_model.assert_called_once()
            assert iteration_engine.ml_model == mock_model

    def test_init_iter_uses_seed_from_list_when_provided(
        self, iteration_engine: SimulationEngine
    ) -> None:
        """Test init_iter uses seed from seeds list when available."""
        # Arrange
        iteration_engine.engine_props["seeds"] = [10, 20, 30]
        iteration = 1

        with patch.object(iteration_engine, "generate_requests") as mock_gen:
            # Act
            iteration_engine.init_iter(iteration)

            # Assert
            mock_gen.assert_called_once_with(20)  # seeds[1]

    def test_init_iter_uses_iteration_plus_one_as_default_seed(
        self, iteration_engine: SimulationEngine
    ) -> None:
        """Test init_iter uses iteration+1 as default seed."""
        # Arrange
        iteration = 3

        with patch.object(iteration_engine, "generate_requests") as mock_gen:
            # Act
            iteration_engine.init_iter(iteration)

            # Assert
            mock_gen.assert_called_once_with(4)  # iteration + 1

    @patch("fusion.core.simulation.seed_request_generation")
    @patch("fusion.core.simulation.seed_rl_components")
    def test_init_iter_uses_separate_seeding_with_request_seeds_and_rl_seed(
        self,
        mock_seed_rl: Mock,
        mock_seed_request: Mock,
        iteration_engine: SimulationEngine,
    ) -> None:
        """Test separate seeding when both request_seeds and rl_seed provided."""
        # Arrange
        iteration_engine.engine_props["request_seeds"] = [10, 20, 30]
        iteration_engine.engine_props["rl_seed"] = 99
        iteration = 1

        with patch.object(iteration_engine, "generate_requests"):
            # Act
            iteration_engine.init_iter(iteration)

            # Assert - RL seed should be constant (99)
            mock_seed_rl.assert_called_once_with(99)
            # Request seed should vary per iteration (20 for iteration 1)
            mock_seed_request.assert_called_once_with(20)

    @patch("fusion.core.simulation.seed_request_generation")
    @patch("fusion.core.simulation.seed_rl_components")
    def test_init_iter_uses_general_seed_for_rl_when_rl_seed_not_specified(
        self,
        mock_seed_rl: Mock,
        mock_seed_request: Mock,
        iteration_engine: SimulationEngine,
    ) -> None:
        """Test init_iter uses general seed for RL when rl_seed not specified."""
        # Arrange
        iteration_engine.engine_props["seed"] = 42
        iteration_engine.engine_props["request_seeds"] = [10, 20, 30]
        iteration = 1

        with patch.object(iteration_engine, "generate_requests"):
            # Act
            iteration_engine.init_iter(iteration)

            # Assert - RL should use general seed (constant across iterations)
            mock_seed_rl.assert_called_once_with(42)
            # Request seed should vary per iteration
            mock_seed_request.assert_called_once_with(20)

    @patch("fusion.core.simulation.seed_request_generation")
    @patch("fusion.core.simulation.seed_rl_components")
    def test_init_iter_varies_both_seeds_when_only_general_seed_specified(
        self,
        mock_seed_rl: Mock,
        mock_seed_request: Mock,
        iteration_engine: SimulationEngine,
    ) -> None:
        """Test both seeds vary when only general seed specified (backward compat)."""
        # Arrange
        iteration_engine.engine_props["seed"] = 42
        iteration = 3

        with patch.object(iteration_engine, "generate_requests"):
            # Act
            iteration_engine.init_iter(iteration)

            # Assert - Both should use iteration+1 when no specific seeding
            # RL uses general seed (constant)
            mock_seed_rl.assert_called_once_with(42)
            # Request uses iteration+1 (varies)
            mock_seed_request.assert_called_once_with(4)

    @patch("fusion.core.simulation.seed_request_generation")
    @patch("fusion.core.simulation.seed_rl_components")
    def test_init_iter_uses_same_seed_for_both_when_no_config_specified(
        self,
        mock_seed_rl: Mock,
        mock_seed_request: Mock,
        iteration_engine: SimulationEngine,
    ) -> None:
        """Test init_iter uses same seed (iteration+1) for both when no config."""
        # Arrange
        iteration = 2

        with patch.object(iteration_engine, "generate_requests"):
            # Act
            iteration_engine.init_iter(iteration)

            # Assert - Both should use iteration+1 as default
            mock_seed_rl.assert_called_once_with(3)  # iteration + 1
            mock_seed_request.assert_called_once_with(3)  # iteration + 1

    def test_end_iter_calculates_statistics_and_saves_on_save_step(
        self, iteration_engine: SimulationEngine
    ) -> None:
        """Test end_iter calculates statistics and saves on appropriate steps."""
        # Arrange
        # save_step is 5, so should save on iteration 4 (4+1=5)
        iteration = 4
        iteration_engine.engine_props["is_training"] = False
        iteration_engine.stats_obj.calculate_confidence_interval.return_value = False  # type: ignore[attr-defined]

        with patch.object(iteration_engine, "_save_all_stats") as mock_save:
            # Act
            result = iteration_engine.end_iter(iteration)

            # Assert
            iteration_engine.stats_obj.calculate_blocking_statistics.assert_called()  # type: ignore
            iteration_engine.stats_obj.finalize_iteration_statistics.assert_called()  # type: ignore
            mock_save.assert_called_once_with("data")
            assert result is False

    def test_end_iter_skips_confidence_interval_during_training(
        self, iteration_engine: SimulationEngine
    ) -> None:
        """Test end_iter skips confidence interval calculation during training."""
        # Arrange
        iteration = 1
        iteration_engine.engine_props["is_training"] = True

        # Act
        result = iteration_engine.end_iter(iteration, print_flag=False)

        # Assert
        iteration_engine.stats_obj.calculate_confidence_interval.assert_not_called()  # type: ignore[attr-defined]
        assert result is False


class TestSimulationEngineFullExecution:
    """Tests for complete simulation execution."""

    @pytest.fixture
    def execution_engine(self) -> SimulationEngine:
        """Provide engine for full execution testing."""
        engine_props = {
            "network": "test",
            "date": "2024-01-01",
            "sim_start": "exec_test",
            "output_train_data": False,
            "max_iters": 3,
            "print_step": 1,
            "save_step": 3,
            "erlang": 100.0,
            "topology_info": {
                "nodes": {"A": {}, "B": {}},
                "links": {
                    "1": {
                        "source": "A",
                        "destination": "B",
                        "length": 100.0,
                        "fiber": {"num_cores": 1},
                    }
                },
            },
            "c_band": 80,
        }
        return SimulationEngine(engine_props)

    def test_run_executes_complete_simulation_workflow(
        self, execution_engine: SimulationEngine
    ) -> None:
        """Test run method executes complete simulation workflow."""
        # Arrange
        execution_engine.reqs_dict = {1.0: {"request_type": "arrival"}}

        with (
            patch.object(execution_engine, "init_iter") as mock_init,
            patch.object(execution_engine, "end_iter", return_value=False) as mock_end,
            patch.object(execution_engine, "handle_request"),
        ):
            # Act
            result = execution_engine.run(seed=42)

            # Assert
            assert mock_init.call_count == 3  # max_iters
            assert mock_end.call_count == 3
            assert result == 3  # Number of completed iterations

    def test_run_stops_early_when_confidence_interval_reached(
        self, execution_engine: SimulationEngine
    ) -> None:
        """Test run stops early when confidence interval is reached."""
        # Arrange
        execution_engine.reqs_dict = {1.0: {"request_type": "arrival"}}

        with (
            patch.object(execution_engine, "init_iter"),
            patch.object(
                execution_engine, "end_iter", side_effect=[False, True, False]
            ),
            patch.object(execution_engine, "handle_request"),
        ):
            # Act
            result = execution_engine.run()

            # Assert
            assert result == 2  # Stopped after 2 iterations

    def test_run_stops_when_stop_flag_is_set(
        self, execution_engine: SimulationEngine
    ) -> None:
        """Test run stops when stop flag is set."""
        # Arrange
        stop_flag = Mock()
        stop_flag.is_set.side_effect = [False, True]  # Stop after first iteration
        execution_engine.engine_props["stop_flag"] = stop_flag
        execution_engine.stop_flag = stop_flag
        execution_engine.reqs_dict = {1.0: {"request_type": "arrival"}}

        with (
            patch.object(execution_engine, "init_iter"),
            patch.object(execution_engine, "end_iter", return_value=False),
            patch.object(execution_engine, "handle_request"),
        ):
            # Act
            result = execution_engine.run()

            # Assert
            assert result == 1  # Stopped after 1 iteration


class TestSimulationEngineSignalHandling:
    """Tests for signal handling functionality."""

    @pytest.fixture
    def signal_engine(self) -> SimulationEngine:
        """Provide engine for signal handling tests."""
        engine_props = {
            "network": "test",
            "date": "2024-01-01",
            "sim_start": "signal_test",
            "output_train_data": False,
        }
        return SimulationEngine(engine_props)

    def test_signal_save_handler_calls_save_all_stats(
        self, signal_engine: SimulationEngine
    ) -> None:
        """Test signal handler calls save_all_stats."""
        # Arrange
        signum = signal.SIGINT
        frame = Mock()

        with patch.object(signal_engine, "_save_all_stats") as mock_save:
            # Act
            signal_engine._signal_save_handler(signum, frame)

            # Assert
            mock_save.assert_called_once()


class TestSimulationEngineEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_update_arrival_params_with_no_requests_dict_handles_gracefully(
        self,
    ) -> None:
        """Test update_arrival_params handles missing requests dict."""
        # Arrange
        engine = SimulationEngine(
            {
                "network": "test",
                "date": "2024-01-01",
                "sim_start": "edge_test",
                "output_train_data": False,
            }
        )
        current_time = 1.0

        # Act & Assert - should not raise exception
        engine.update_arrival_params(current_time)

    def test_update_arrival_params_with_missing_time_handles_gracefully(self) -> None:
        """Test update_arrival_params handles missing time in requests dict."""
        # Arrange
        engine = SimulationEngine(
            {
                "network": "test",
                "date": "2024-01-01",
                "sim_start": "edge_test",
                "output_train_data": False,
            }
        )
        engine.reqs_dict = {2.0: {"req_id": 1}}
        current_time = 1.0  # Different time

        # Act & Assert - should not raise exception
        engine.update_arrival_params(current_time)

    def test_generate_requests_calls_get_requests_with_correct_params(self) -> None:
        """Test generate_requests calls get_requests function properly."""
        # Arrange
        engine = SimulationEngine(
            {
                "network": "test",
                "date": "2024-01-01",
                "sim_start": "gen_test",
                "output_train_data": False,
            }
        )
        seed = 123

        with patch("fusion.core.simulation.get_requests") as mock_get_requests:
            mock_get_requests.return_value = {
                1.0: {"req_id": 1},
                2.0: {"req_id": 2},
            }

            # Act
            engine.generate_requests(seed)

            # Assert
            mock_get_requests.assert_called_once_with(
                seed=seed, engine_props=engine.engine_props
            )
            assert engine.reqs_dict == {
                1.0: {"req_id": 1},
                2.0: {"req_id": 2},
            }

    def test_save_all_stats_saves_ml_data_when_ml_metrics_available(self) -> None:
        """Test _save_all_stats saves ML data when ML metrics collector exists."""
        # Arrange
        engine = SimulationEngine(
            {
                "network": "test",
                "date": "2024-01-01",
                "sim_start": "ml_test",
                "output_train_data": True,
                "max_iters": 10,
            }
        )
        engine.ml_metrics = Mock()
        engine.stats_obj = Mock()
        engine.stats_obj.iteration = 5
        engine.stats_obj.get_blocking_statistics.return_value = {}
        engine.stats_obj.stats_props = Mock()

        with patch.object(engine.persistence, "save_stats"):
            # Act
            engine._save_all_stats("test_data")

            # Assert
            engine.ml_metrics.save_train_data.assert_called_once_with(
                iteration=5, max_iterations=10, base_file_path="test_data"
            )
