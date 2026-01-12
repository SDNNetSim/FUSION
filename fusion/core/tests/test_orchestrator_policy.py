"""
Unit tests for SDNOrchestrator policy integration (P5.5).

Tests cover:
1. Backward compatibility: policy disabled produces same outcomes
2. Policy invocation: mock policy called once per request
3. Protection gating: when protection disabled, ProtectionPipeline not used

Phase: P5.5 - Orchestrator Integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

if TYPE_CHECKING:
    from fusion.core.orchestrator import SDNOrchestrator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock SimulationConfig."""
    config = MagicMock()
    config.grooming_enabled = False
    config.snr_enabled = False
    config.slicing_enabled = False
    config.snr_recheck = False
    config.can_partially_serve = False
    config.protection_enabled = False
    config.guard_slots = 0
    config.dynamic_lps = False
    config.fixed_grid = True
    return config


@pytest.fixture
def mock_pipelines() -> MagicMock:
    """Create mock PipelineSet with all pipelines."""
    pipelines = MagicMock()
    pipelines.routing = MagicMock()
    pipelines.spectrum = MagicMock()
    pipelines.grooming = None
    pipelines.snr = None
    pipelines.slicing = None
    return pipelines


@pytest.fixture
def mock_request() -> MagicMock:
    """Create mock Request."""
    request = MagicMock()
    request.request_id = 1
    request.source = "A"
    request.destination = "C"
    request.bandwidth_gbps = 100
    request.lightpath_ids = []
    request.protection_required = False
    return request


@pytest.fixture
def mock_network_state() -> MagicMock:
    """Create mock NetworkState."""
    network_state = MagicMock()
    return network_state


@pytest.fixture
def orchestrator(mock_config: MagicMock, mock_pipelines: MagicMock) -> SDNOrchestrator:
    """Create orchestrator with mock dependencies (no policy)."""
    from fusion.core.orchestrator import SDNOrchestrator

    return SDNOrchestrator(mock_config, mock_pipelines)


@pytest.fixture
def mock_policy() -> MagicMock:
    """Create mock ControlPolicy."""
    policy = MagicMock()
    policy.get_name.return_value = "MockPolicy"
    policy.select_action.return_value = 0  # Select first option
    policy.update.return_value = None
    return policy


@pytest.fixture
def mock_path_option() -> MagicMock:
    """Create mock PathOption."""
    option = MagicMock()
    option.path_index = 0
    option.path = ("A", "B", "C")
    option.weight_km = 100.0
    option.is_feasible = True
    option.modulation = "QPSK"
    option.slots_needed = 4
    option.congestion = 0.0
    option.available_slots = 1.0
    return option


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests verifying backward compatibility when policy is disabled."""

    def test_handle_arrival_without_policy_delegates_correctly(
        self,
        orchestrator: SDNOrchestrator,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival works without policy set."""
        # Setup route result
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.paths = (("A", "B", "C"),)
        route_result.modulations = (("QPSK",),)
        route_result.weights_km = (100.0,)
        route_result.has_protection = False
        route_result.connection_index = None
        orchestrator.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 4
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        spectrum_result.slots_needed = 4
        spectrum_result.snr_db = None
        spectrum_result.achieved_bandwidth_gbps = None
        orchestrator.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_lightpath.request_allocations = {}
        mock_network_state.create_lightpath.return_value = mock_lightpath

        # Verify no policy is set
        assert orchestrator.policy is None
        assert not orchestrator.has_policy()

        # Call handle_arrival (should work normally without policy)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        # Verify routing was called
        orchestrator.routing.find_routes.assert_called_once()
        assert result.success

    def test_handle_arrival_with_policy_delegates_when_no_policy(
        self,
        orchestrator: SDNOrchestrator,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival_with_policy delegates to handle_arrival when no policy."""
        # Setup route result
        route_result = MagicMock()
        route_result.is_empty = True
        orchestrator.routing.find_routes.return_value = route_result

        # Call with_policy when no policy set
        assert orchestrator.policy is None
        result = orchestrator.handle_arrival_with_policy(mock_request, mock_network_state)

        # Should have called routing (through handle_arrival)
        orchestrator.routing.find_routes.assert_called_once()

    def test_orchestrator_init_defaults_policy_to_none(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
    ) -> None:
        """Orchestrator defaults policy to None."""
        from fusion.core.orchestrator import SDNOrchestrator

        orch = SDNOrchestrator(mock_config, mock_pipelines)

        assert orch.policy is None
        assert orch.rl_adapter is None
        assert orch.protection_pipeline is None


# =============================================================================
# Policy Invocation Tests
# =============================================================================


class TestPolicyInvocation:
    """Tests verifying policy is invoked correctly."""

    def test_policy_called_once_per_request(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
        mock_policy: MagicMock,
    ) -> None:
        """Policy.select_action called exactly once per request."""
        from dataclasses import dataclass

        from fusion.core.orchestrator import SDNOrchestrator

        # Create orchestrator with policy
        orchestrator = SDNOrchestrator(mock_config, mock_pipelines, policy=mock_policy)

        # Setup mock adapter
        mock_adapter = MagicMock()
        mock_option = MagicMock()
        mock_option.path_index = 0
        mock_option.path = ("A", "B", "C")
        mock_option.is_feasible = True
        mock_option.modulation = None  # Use None to avoid forced_modulation path
        mock_adapter.get_path_options.return_value = [mock_option]
        orchestrator.rl_adapter = mock_adapter

        # Setup route result for the forced path - use real dataclass
        @dataclass
        class MockRouteResult:
            is_empty: bool = False
            paths: tuple = (("A", "B", "C"),)
            modulations: tuple = (("QPSK",),)
            weights_km: tuple = (100.0,)
            has_protection: bool = False
            connection_index: int | None = None

        route_result = MockRouteResult()
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 4
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        spectrum_result.slots_needed = 4
        spectrum_result.snr_db = None
        spectrum_result.achieved_bandwidth_gbps = None
        mock_pipelines.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_lightpath.request_allocations = {}
        mock_network_state.create_lightpath.return_value = mock_lightpath

        # Call handle_arrival_with_policy
        orchestrator.handle_arrival_with_policy(mock_request, mock_network_state)

        # Verify policy was called exactly once
        mock_policy.select_action.assert_called_once()

    def test_policy_receives_correct_options_count(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
        mock_policy: MagicMock,
    ) -> None:
        """Policy receives options list with correct length."""
        from dataclasses import dataclass

        from fusion.core.orchestrator import SDNOrchestrator

        # Create orchestrator with policy
        orchestrator = SDNOrchestrator(mock_config, mock_pipelines, policy=mock_policy)

        # Create 3 mock path options
        mock_options = []
        for i in range(3):
            opt = MagicMock()
            opt.path_index = i
            opt.path = ("A", f"B{i}", "C")
            opt.is_feasible = i == 0  # Only first is feasible
            opt.modulation = None  # Avoid forced_modulation code path
            mock_options.append(opt)

        # Setup mock adapter
        mock_adapter = MagicMock()
        mock_adapter.get_path_options.return_value = mock_options
        orchestrator.rl_adapter = mock_adapter

        # Setup route result - use real dataclass
        @dataclass
        class MockRouteResult:
            is_empty: bool = False
            paths: tuple = (("A", "B0", "C"),)
            modulations: tuple = (("QPSK",),)
            weights_km: tuple = (100.0,)
            has_protection: bool = False
            connection_index: int | None = None

        route_result = MockRouteResult()
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 4
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        spectrum_result.slots_needed = 4
        spectrum_result.snr_db = None
        spectrum_result.achieved_bandwidth_gbps = None
        mock_pipelines.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_lightpath.request_allocations = {}
        mock_network_state.create_lightpath.return_value = mock_lightpath

        # Call handle_arrival_with_policy
        orchestrator.handle_arrival_with_policy(mock_request, mock_network_state)

        # Verify policy received 3 options
        call_args = mock_policy.select_action.call_args
        options_arg = call_args[0][1]  # Second positional arg is options
        assert len(options_arg) == 3

    def test_policy_update_called_after_allocation(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
        mock_policy: MagicMock,
    ) -> None:
        """Policy.update called after allocation completes."""
        from dataclasses import dataclass

        from fusion.core.orchestrator import SDNOrchestrator

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines, policy=mock_policy)

        # Setup mock adapter
        mock_adapter = MagicMock()
        mock_option = MagicMock()
        mock_option.path_index = 0
        mock_option.path = ("A", "B", "C")
        mock_option.is_feasible = True
        mock_option.modulation = None  # Avoid forced_modulation code path
        mock_adapter.get_path_options.return_value = [mock_option]
        orchestrator.rl_adapter = mock_adapter

        # Setup route result - use real dataclass
        @dataclass
        class MockRouteResult:
            is_empty: bool = False
            paths: tuple = (("A", "B", "C"),)
            modulations: tuple = (("QPSK",),)
            weights_km: tuple = (100.0,)
            has_protection: bool = False
            connection_index: int | None = None

        route_result = MockRouteResult()
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 4
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        spectrum_result.slots_needed = 4
        spectrum_result.snr_db = None
        spectrum_result.achieved_bandwidth_gbps = None
        mock_pipelines.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_lightpath.request_allocations = {}
        mock_network_state.create_lightpath.return_value = mock_lightpath

        # Call
        orchestrator.handle_arrival_with_policy(mock_request, mock_network_state)

        # Verify update was called
        mock_policy.update.assert_called_once()
        # Should have been called with action=0 and positive reward (success)
        call_args = mock_policy.update.call_args
        assert call_args[0][1] == 0  # action
        assert call_args[0][2] > 0  # reward (success)

    def test_invalid_action_blocks_request(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
        mock_policy: MagicMock,
    ) -> None:
        """Invalid action from policy results in blocked request."""
        from fusion.core.orchestrator import SDNOrchestrator
        from fusion.domain.request import BlockReason

        # Policy returns invalid action
        mock_policy.select_action.return_value = -1

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines, policy=mock_policy)

        # Setup mock adapter
        mock_adapter = MagicMock()
        mock_option = MagicMock()
        mock_option.path_index = 0
        mock_option.path = ("A", "B", "C")
        mock_option.is_feasible = True
        mock_adapter.get_path_options.return_value = [mock_option]
        orchestrator.rl_adapter = mock_adapter

        # Call
        result = orchestrator.handle_arrival_with_policy(mock_request, mock_network_state)

        # Should be blocked
        assert not result.success
        assert result.block_reason == BlockReason.CONGESTION


# =============================================================================
# Protection Gating Tests
# =============================================================================


class TestProtectionGating:
    """Tests verifying protection pipeline is only used when enabled."""

    def test_protection_pipeline_not_used_when_disabled(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
        mock_policy: MagicMock,
    ) -> None:
        """ProtectionPipeline not used when protection_enabled=False."""
        from dataclasses import dataclass

        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.protection_enabled = False

        # Create mock protection pipeline
        mock_protection = MagicMock()

        orchestrator = SDNOrchestrator(
            mock_config,
            mock_pipelines,
            policy=mock_policy,
            protection_pipeline=mock_protection,
        )

        # Setup mock adapter
        mock_adapter = MagicMock()
        mock_option = MagicMock()
        mock_option.path_index = 0
        mock_option.path = ("A", "B", "C")
        mock_option.is_feasible = True
        mock_option.modulation = None  # Avoid forced_modulation code path
        mock_adapter.get_path_options.return_value = [mock_option]
        orchestrator.rl_adapter = mock_adapter

        # Setup route result - use real dataclass
        @dataclass
        class MockRouteResult:
            is_empty: bool = False
            paths: tuple = (("A", "B", "C"),)
            modulations: tuple = (("QPSK",),)
            weights_km: tuple = (100.0,)
            has_protection: bool = False
            connection_index: int | None = None

        route_result = MockRouteResult()
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 4
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        spectrum_result.slots_needed = 4
        spectrum_result.snr_db = None
        spectrum_result.achieved_bandwidth_gbps = None
        mock_pipelines.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_lightpath.request_allocations = {}
        mock_network_state.create_lightpath.return_value = mock_lightpath

        # Call
        orchestrator.handle_arrival_with_policy(mock_request, mock_network_state)

        # Protection pipeline should NOT be called
        mock_protection.find_protected_paths.assert_not_called()
        mock_protection.allocate_protected.assert_not_called()

    def test_protection_pipeline_not_used_when_request_not_protected(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
        mock_policy: MagicMock,
    ) -> None:
        """ProtectionPipeline not used when request doesn't require protection."""
        from dataclasses import dataclass

        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.protection_enabled = True
        mock_request.protection_required = False  # Request doesn't need protection

        mock_protection = MagicMock()

        orchestrator = SDNOrchestrator(
            mock_config,
            mock_pipelines,
            policy=mock_policy,
            protection_pipeline=mock_protection,
        )

        # Setup mock adapter
        mock_adapter = MagicMock()
        mock_option = MagicMock()
        mock_option.path_index = 0
        mock_option.path = ("A", "B", "C")
        mock_option.is_feasible = True
        mock_option.modulation = None  # Avoid forced_modulation code path
        mock_adapter.get_path_options.return_value = [mock_option]
        orchestrator.rl_adapter = mock_adapter

        # Setup route result - use real dataclass
        @dataclass
        class MockRouteResult:
            is_empty: bool = False
            paths: tuple = (("A", "B", "C"),)
            modulations: tuple = (("QPSK",),)
            weights_km: tuple = (100.0,)
            has_protection: bool = False
            connection_index: int | None = None

        route_result = MockRouteResult()
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 4
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        spectrum_result.slots_needed = 4
        spectrum_result.snr_db = None
        spectrum_result.achieved_bandwidth_gbps = None
        mock_pipelines.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_lightpath.request_allocations = {}
        mock_network_state.create_lightpath.return_value = mock_lightpath

        # Call
        orchestrator.handle_arrival_with_policy(mock_request, mock_network_state)

        # Protection pipeline should NOT be called
        mock_protection.find_protected_paths.assert_not_called()

    def test_protection_pipeline_used_when_all_conditions_met(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
        mock_policy: MagicMock,
    ) -> None:
        """ProtectionPipeline used when all conditions are met."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.protection_enabled = True
        mock_request.protection_required = True

        # Setup network state with topology
        mock_network_state.topology = MagicMock()

        mock_protection = MagicMock()
        mock_protection.find_protected_paths.return_value = (
            ["A", "B", "C"],
            ["A", "D", "C"],
        )
        mock_protection.verify_disjointness.return_value = True
        mock_protection.allocate_protected.return_value = MagicMock(
            success=False, failure_reason="no_common_spectrum"
        )

        # Setup route result for fallback
        route_result = MagicMock()
        route_result.is_empty = True
        route_result.has_protection = False
        mock_pipelines.routing.find_routes.return_value = route_result

        orchestrator = SDNOrchestrator(
            mock_config,
            mock_pipelines,
            policy=mock_policy,
            protection_pipeline=mock_protection,
        )

        # Setup mock adapter
        mock_adapter = MagicMock()
        orchestrator.rl_adapter = mock_adapter

        # Call
        orchestrator.handle_arrival_with_policy(mock_request, mock_network_state)

        # Protection pipeline should be called
        mock_protection.find_protected_paths.assert_called_once()


# =============================================================================
# PolicyFactory Tests
# =============================================================================


class TestPolicyFactory:
    """Tests for PolicyFactory."""

    def test_create_default_policy(self) -> None:
        """Factory creates FirstFeasiblePolicy by default."""
        from fusion.policies import FirstFeasiblePolicy, PolicyFactory

        policy = PolicyFactory.create(None)

        assert isinstance(policy, FirstFeasiblePolicy)

    def test_create_heuristic_policy(self) -> None:
        """Factory creates heuristic policies correctly."""
        from fusion.policies import (
            PolicyConfig,
            PolicyFactory,
            ShortestFeasiblePolicy,
        )

        config = PolicyConfig(policy_type="heuristic", policy_name="shortest")
        policy = PolicyFactory.create(config)

        assert isinstance(policy, ShortestFeasiblePolicy)

    def test_create_from_dict(self) -> None:
        """Factory creates policy from dict config."""
        from fusion.policies import LeastCongestedPolicy, PolicyFactory

        config_dict = {
            "policy_type": "heuristic",
            "policy_name": "least_congested",
        }
        policy = PolicyFactory.from_dict(config_dict)

        assert isinstance(policy, LeastCongestedPolicy)

    def test_get_default_policy(self) -> None:
        """get_default_policy returns FirstFeasiblePolicy."""
        from fusion.policies import FirstFeasiblePolicy, PolicyFactory

        policy = PolicyFactory.get_default_policy()

        assert isinstance(policy, FirstFeasiblePolicy)

    def test_unknown_heuristic_raises(self) -> None:
        """Unknown heuristic name raises ValueError."""
        from fusion.policies import PolicyConfig, PolicyFactory

        config = PolicyConfig(policy_type="heuristic", policy_name="unknown_policy")

        with pytest.raises(ValueError, match="Unknown heuristic policy"):
            PolicyFactory.create(config)

    def test_ml_without_model_path_raises(self) -> None:
        """ML policy without model_path raises ValueError."""
        from fusion.policies import PolicyConfig, PolicyFactory

        config = PolicyConfig(policy_type="ml", policy_name="ml", model_path=None)

        with pytest.raises(ValueError, match="model_path required"):
            PolicyFactory.create(config)


# =============================================================================
# Orchestrator Property Tests
# =============================================================================


class TestOrchestratorProperties:
    """Tests for orchestrator policy properties."""

    def test_policy_property_getter_setter(
        self,
        orchestrator: SDNOrchestrator,
        mock_policy: MagicMock,
    ) -> None:
        """Policy can be set and retrieved via property."""
        assert orchestrator.policy is None

        orchestrator.policy = mock_policy

        assert orchestrator.policy is mock_policy
        assert orchestrator.has_policy()

    def test_rl_adapter_property_getter_setter(
        self,
        orchestrator: SDNOrchestrator,
    ) -> None:
        """RL adapter can be set and retrieved via property."""
        assert orchestrator.rl_adapter is None

        mock_adapter = MagicMock()
        orchestrator.rl_adapter = mock_adapter

        assert orchestrator.rl_adapter is mock_adapter

    def test_protection_pipeline_property_getter_setter(
        self,
        orchestrator: SDNOrchestrator,
    ) -> None:
        """Protection pipeline can be set and retrieved via property."""
        assert orchestrator.protection_pipeline is None

        mock_protection = MagicMock()
        orchestrator.protection_pipeline = mock_protection

        assert orchestrator.protection_pipeline is mock_protection
