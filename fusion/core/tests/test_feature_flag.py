"""
Unit tests for v5 feature flag and orchestrator wiring (P3.3).

Tests the use_orchestrator feature flag resolution, configuration validation,
and dual-path operation in SimulationEngine.
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ..simulation import (
    DEFAULT_USE_ORCHESTRATOR,
    ENV_VAR_USE_ORCHESTRATOR,
    SimulationEngine,
    _resolve_use_orchestrator,
    _validate_orchestrator_config,
)


class TestFeatureFlagResolution:
    """Tests for use_orchestrator feature flag resolution (P3.3.b)."""

    def test_resolve_default_when_no_config(self) -> None:
        """Test default value is returned when no config provided."""
        # Arrange
        engine_props: dict[str, Any] = {}
        # Ensure env var is not set
        if ENV_VAR_USE_ORCHESTRATOR in os.environ:
            del os.environ[ENV_VAR_USE_ORCHESTRATOR]

        # Act
        result = _resolve_use_orchestrator(engine_props)

        # Assert
        assert result == DEFAULT_USE_ORCHESTRATOR
        assert result is False

    def test_resolve_from_engine_props_true(self) -> None:
        """Test resolution from engine_props when set to True."""
        # Arrange
        engine_props = {"use_orchestrator": True}
        if ENV_VAR_USE_ORCHESTRATOR in os.environ:
            del os.environ[ENV_VAR_USE_ORCHESTRATOR]

        # Act
        result = _resolve_use_orchestrator(engine_props)

        # Assert
        assert result is True

    def test_resolve_from_engine_props_false(self) -> None:
        """Test resolution from engine_props when set to False."""
        # Arrange
        engine_props = {"use_orchestrator": False}
        if ENV_VAR_USE_ORCHESTRATOR in os.environ:
            del os.environ[ENV_VAR_USE_ORCHESTRATOR]

        # Act
        result = _resolve_use_orchestrator(engine_props)

        # Assert
        assert result is False

    def test_env_var_takes_priority_over_engine_props(self) -> None:
        """Test environment variable has higher priority than engine_props."""
        # Arrange
        engine_props = {"use_orchestrator": False}
        os.environ[ENV_VAR_USE_ORCHESTRATOR] = "true"

        try:
            # Act
            result = _resolve_use_orchestrator(engine_props)

            # Assert
            assert result is True
        finally:
            del os.environ[ENV_VAR_USE_ORCHESTRATOR]

    def test_env_var_true_values(self) -> None:
        """Test various truthy values for environment variable."""
        true_values = ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"]

        for value in true_values:
            os.environ[ENV_VAR_USE_ORCHESTRATOR] = value
            try:
                result = _resolve_use_orchestrator({})
                assert result is True, f"Expected True for env var value '{value}'"
            finally:
                del os.environ[ENV_VAR_USE_ORCHESTRATOR]

    def test_env_var_false_values(self) -> None:
        """Test various falsy values for environment variable."""
        false_values = ["false", "FALSE", "False", "0", "no", "NO", "off", "OFF", ""]

        for value in false_values:
            os.environ[ENV_VAR_USE_ORCHESTRATOR] = value
            try:
                result = _resolve_use_orchestrator({})
                assert result is False, f"Expected False for env var value '{value}'"
            finally:
                del os.environ[ENV_VAR_USE_ORCHESTRATOR]


class TestConfigValidation:
    """Tests for orchestrator configuration validation (P3.3.e)."""

    def test_validate_passes_for_standard_config(self) -> None:
        """Test validation passes for standard configuration."""
        # Arrange
        engine_props = {
            "route_method": "k_shortest_path",
            "max_segments": 1,
        }

        # Act & Assert - should not raise
        _validate_orchestrator_config(engine_props)

    def test_validate_passes_for_protection_only(self) -> None:
        """Test validation passes for protection-only config."""
        # Arrange
        engine_props = {
            "route_method": "1plus1_protection",
            "max_segments": 1,
        }

        # Act & Assert - should not raise
        _validate_orchestrator_config(engine_props)

    def test_validate_passes_for_slicing_only(self) -> None:
        """Test validation passes for slicing-only config."""
        # Arrange
        engine_props = {
            "route_method": "k_shortest_path",
            "max_segments": 3,
        }

        # Act & Assert - should not raise
        _validate_orchestrator_config(engine_props)

    def test_validate_fails_for_protection_plus_slicing(self) -> None:
        """Test validation fails when protection and slicing combined."""
        # Arrange
        engine_props = {
            "route_method": "1plus1_protection",
            "max_segments": 3,
        }

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            _validate_orchestrator_config(engine_props)

        assert "protection" in str(exc_info.value).lower()
        assert "slicing" in str(exc_info.value).lower()

    def test_validate_fails_for_node_disjoint_without_protection(self) -> None:
        """Test validation fails for node_disjoint without protection."""
        # Arrange
        engine_props = {
            "route_method": "k_shortest_path",
            "node_disjoint_protection": True,
        }

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            _validate_orchestrator_config(engine_props)

        assert "node_disjoint" in str(exc_info.value).lower()

    def test_validate_warns_for_slicing_without_grooming(self) -> None:
        """Test validation warns (but passes) for slicing without grooming."""
        # Arrange
        engine_props = {
            "route_method": "k_shortest_path",
            "max_segments": 3,
            "is_grooming_enabled": False,
        }

        # Act & Assert - should not raise (just warns)
        _validate_orchestrator_config(engine_props)

    def test_validate_warns_for_snr_recheck_without_snr(self) -> None:
        """Test validation warns (but passes) for snr_recheck without snr."""
        # Arrange
        engine_props = {
            "route_method": "k_shortest_path",
            "snr_recheck": True,
            "snr_type": None,
        }

        # Act & Assert - should not raise (just warns)
        _validate_orchestrator_config(engine_props)

    def test_validate_passes_for_full_valid_config(self) -> None:
        """Test validation passes for full valid config with all features."""
        # Arrange
        engine_props = {
            "route_method": "k_shortest_path",
            "max_segments": 3,
            "is_grooming_enabled": True,
            "snr_type": "snr_e2e",
            "snr_recheck": True,
        }

        # Act & Assert - should not raise
        _validate_orchestrator_config(engine_props)


class TestSimulationEngineFeatureFlag:
    """Tests for SimulationEngine dual-path initialization."""

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

    def test_engine_init_defaults_to_legacy_mode(
        self, basic_engine_props: dict[str, Any]
    ) -> None:
        """Test SimulationEngine defaults to legacy mode."""
        # Arrange
        if ENV_VAR_USE_ORCHESTRATOR in os.environ:
            del os.environ[ENV_VAR_USE_ORCHESTRATOR]

        # Act
        engine = SimulationEngine(basic_engine_props)

        # Assert
        assert engine.use_orchestrator is False
        assert engine._orchestrator is None
        assert engine._sim_config is None
        assert engine._network_state is None

    def test_engine_init_with_orchestrator_flag_in_props(
        self, basic_engine_props: dict[str, Any]
    ) -> None:
        """Test SimulationEngine with use_orchestrator in engine_props."""
        # Arrange
        basic_engine_props["use_orchestrator"] = True

        # Act
        engine = SimulationEngine(basic_engine_props)

        # Assert
        assert engine.use_orchestrator is True

    def test_engine_init_with_orchestrator_env_var(
        self, basic_engine_props: dict[str, Any]
    ) -> None:
        """Test SimulationEngine with FUSION_USE_ORCHESTRATOR env var."""
        # Arrange
        os.environ[ENV_VAR_USE_ORCHESTRATOR] = "true"

        try:
            # Act
            engine = SimulationEngine(basic_engine_props)

            # Assert
            assert engine.use_orchestrator is True
        finally:
            del os.environ[ENV_VAR_USE_ORCHESTRATOR]

    def test_engine_init_validates_config_in_orchestrator_mode(
        self, basic_engine_props: dict[str, Any]
    ) -> None:
        """Test SimulationEngine validates config when orchestrator mode enabled."""
        # Arrange
        basic_engine_props["use_orchestrator"] = True
        basic_engine_props["route_method"] = "1plus1_protection"
        basic_engine_props["max_segments"] = 3

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            SimulationEngine(basic_engine_props)

        assert "protection" in str(exc_info.value).lower()


class TestSimulationEngineDualPath:
    """Tests for SimulationEngine dual-path operation."""

    @pytest.fixture
    def topology_engine_props(self) -> dict[str, Any]:
        """Provide engine properties with topology information."""
        return {
            "network": "test",
            "date": "2024-01-01",
            "sim_start": "dual_path_test",
            "output_train_data": False,
            "use_orchestrator": True,
            "topology_info": {
                "nodes": {"A": {}, "B": {}, "C": {}},
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
            "guard_slots": 1,
            "num_requests": 100,
            "arrival_rate": 1.0,
            "holding_time": 1.0,
            "route_method": "k_shortest_path",
            "k_paths": 3,
            "allocation_method": "first_fit",
            "cores_per_link": 4,
        }

    def test_create_topology_initializes_orchestrator_path(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test create_topology initializes v5 components when orchestrator enabled."""
        # Arrange
        engine = SimulationEngine(topology_engine_props)

        # Act
        engine.create_topology()

        # Assert
        assert engine._sim_config is not None
        assert engine._network_state is not None
        assert engine._orchestrator is not None

    def test_create_topology_legacy_mode_skips_orchestrator_init(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test create_topology skips v5 init when in legacy mode."""
        # Arrange
        topology_engine_props["use_orchestrator"] = False
        engine = SimulationEngine(topology_engine_props)

        # Act
        engine.create_topology()

        # Assert
        assert engine._orchestrator is None

    def test_handle_arrival_uses_orchestrator_when_enabled(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test handle_arrival delegates to orchestrator when enabled."""
        # Arrange
        engine = SimulationEngine(topology_engine_props)
        engine.create_topology()
        engine.reqs_dict = {
            (1, 1.0): {
                "req_id": 1,
                "source": "A",
                "destination": "C",
                "arrive": 1.0,
                "depart": 5.0,
                "request_type": "arrival",
                "bandwidth": 100,
                "mod_formats": {"QPSK": {"max_length": [200]}},
            }
        }

        # Mock orchestrator
        mock_result = Mock()
        mock_result.success = True
        mock_result.lightpaths_created = (1,)
        mock_result.lightpaths_groomed = ()
        mock_result.all_lightpath_ids = (1,)
        mock_result.is_groomed = False
        mock_result.is_partially_groomed = False
        mock_result.is_sliced = False
        mock_result.modulations = ("QPSK",)
        mock_result.cores = (0,)
        mock_result.bands = ("c",)
        mock_result.start_slots = (0,)
        mock_result.end_slots = (10,)
        mock_result.bandwidth_allocations = (100,)
        mock_result.lightpath_bandwidths = (100,)
        mock_result.block_reason = None

        engine._orchestrator = Mock()
        engine._orchestrator.handle_arrival.return_value = mock_result

        # Mock network_state entirely (get_lightpath is read-only on real NetworkState)
        mock_lightpath = Mock()
        mock_lightpath.path = ["A", "B", "C"]
        mock_network_state = Mock()
        mock_network_state.get_lightpath.return_value = mock_lightpath
        mock_network_state.network_spectrum_dict = engine.network_spectrum_dict
        engine._network_state = mock_network_state

        # Act
        engine.handle_arrival(current_time=(1, 1.0))

        # Assert
        engine._orchestrator.handle_arrival.assert_called_once()

    def test_handle_arrival_uses_legacy_when_disabled(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test handle_arrival uses SDNController when orchestrator disabled."""
        # Arrange
        topology_engine_props["use_orchestrator"] = False
        engine = SimulationEngine(topology_engine_props)
        engine.create_topology()
        engine.reqs_dict = {
            1.0: {
                "req_id": 1,
                "source": "A",
                "destination": "C",
                "arrive": 1.0,
                "depart": 5.0,
                "request_type": "arrival",
                "bandwidth": 100,
                "mod_formats": {"QPSK": {"max_length": [200]}},
            }
        }

        # Mock SDN controller
        engine.sdn_obj = Mock()
        engine.sdn_obj.sdn_props = Mock()
        engine.sdn_obj.sdn_props.network_spectrum_dict = {}
        engine.sdn_obj.sdn_props.was_routed = True
        engine.sdn_obj.sdn_props.number_of_transponders = 2
        engine.stats_obj = Mock()

        # Act
        engine.handle_arrival(current_time=1.0)

        # Assert
        engine.sdn_obj.handle_event.assert_called_once()

    def test_reset_clears_v5_state_in_orchestrator_mode(
        self, topology_engine_props: dict[str, Any]
    ) -> None:
        """Test reset clears v5 request cache when in orchestrator mode."""
        # Arrange
        engine = SimulationEngine(topology_engine_props)
        engine.create_topology()
        engine._v5_requests = {(1, 1.0): Mock()}

        # Act
        engine.reset()

        # Assert
        assert engine._v5_requests == {}
        assert engine._network_state is not None


class TestDualPathStatsUpdate:
    """Tests for _update_stats_from_result functionality."""

    @pytest.fixture
    def engine_with_topology(self) -> SimulationEngine:
        """Provide engine with initialized topology."""
        engine_props = {
            "network": "test",
            "date": "2024-01-01",
            "sim_start": "stats_test",
            "output_train_data": False,
            "use_orchestrator": True,
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
            "l_band": 0,
            "s_band": 0,
            "o_band": 0,
            "e_band": 0,
            "guard_slots": 1,
            "num_requests": 100,
            "arrival_rate": 1.0,
            "holding_time": 1.0,
            "route_method": "k_shortest_path",
            "k_paths": 3,
            "allocation_method": "first_fit",
            "cores_per_link": 2,
        }
        engine = SimulationEngine(engine_props)
        engine.create_topology()
        return engine

    def test_stats_updated_on_successful_allocation(
        self, engine_with_topology: SimulationEngine
    ) -> None:
        """Test stats are updated correctly on successful allocation."""
        # Arrange
        engine_with_topology.reqs_dict = {
            (1, 1.0): {
                "req_id": 1,
                "source": "A",
                "destination": "B",
                "bandwidth": 100,
            }
        }

        mock_request = Mock()
        mock_request.request_id = 1
        mock_request.bandwidth_gbps = 100

        mock_result = Mock()
        mock_result.success = True
        mock_result.all_lightpath_ids = (1,)
        mock_result.lightpaths_created = (1,)
        mock_result.lightpaths_groomed = ()
        mock_result.is_groomed = False
        mock_result.is_partially_groomed = False
        mock_result.is_sliced = False
        mock_result.modulations = ("QPSK",)
        mock_result.cores = (0,)
        mock_result.bands = ("c",)
        mock_result.start_slots = (0,)
        mock_result.end_slots = (10,)
        mock_result.bandwidth_allocations = (100,)
        mock_result.lightpath_bandwidths = (100,)
        mock_result.block_reason = None

        initial_bit_rate = engine_with_topology.stats_obj.bit_rate_request
        initial_blocked = engine_with_topology.stats_obj.blocked_requests

        # Act
        engine_with_topology._update_stats_from_result(
            (1, 1.0), mock_request, mock_result
        )

        # Assert - success means bit_rate_request updated but blocked_requests unchanged
        assert engine_with_topology.stats_obj.bit_rate_request == initial_bit_rate + 100
        assert engine_with_topology.stats_obj.blocked_requests == initial_blocked
        assert 1 in engine_with_topology.reqs_status_dict

    def test_stats_updated_on_blocked_allocation(
        self, engine_with_topology: SimulationEngine
    ) -> None:
        """Test stats are updated correctly on blocked allocation."""
        # Arrange
        engine_with_topology.reqs_dict = {
            (1, 1.0): {
                "req_id": 1,
                "source": "A",
                "destination": "B",
                "bandwidth": 100,
            }
        }

        mock_request = Mock()
        mock_request.request_id = 1
        mock_request.bandwidth_gbps = 100

        mock_block_reason = Mock()
        mock_block_reason.to_legacy_string.return_value = "congestion"

        mock_result = Mock()
        mock_result.success = False
        mock_result.block_reason = mock_block_reason
        mock_result.all_lightpath_ids = ()

        initial_blocked = engine_with_topology.stats_obj.blocked_requests
        initial_bit_rate_blocked = engine_with_topology.stats_obj.bit_rate_blocked

        # Act
        engine_with_topology._update_stats_from_result(
            (1, 1.0), mock_request, mock_result
        )

        # Assert
        assert engine_with_topology.stats_obj.blocked_requests == initial_blocked + 1
        assert engine_with_topology.stats_obj.bit_rate_blocked == initial_bit_rate_blocked + 100
