"""
Unit tests for SDNOrchestrator.

Tests use mock pipelines to verify orchestration logic without
requiring actual pipeline implementations.

Phase: P3.2 - SDN Orchestrator Creation
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, PropertyMock, patch

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
def orchestrator(mock_config: MagicMock, mock_pipelines: MagicMock) -> "SDNOrchestrator":
    """Create orchestrator with mock dependencies."""
    from fusion.core.orchestrator import SDNOrchestrator

    return SDNOrchestrator(mock_config, mock_pipelines)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSDNOrchestratorInit:
    """Tests for SDNOrchestrator initialization."""

    def test_init_stores_config(
        self, mock_config: MagicMock, mock_pipelines: MagicMock
    ) -> None:
        """Orchestrator stores config reference."""
        from fusion.core.orchestrator import SDNOrchestrator

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)

        assert orchestrator.config is mock_config

    def test_init_extracts_pipelines(
        self, mock_config: MagicMock, mock_pipelines: MagicMock
    ) -> None:
        """Orchestrator extracts pipelines from PipelineSet."""
        from fusion.core.orchestrator import SDNOrchestrator

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)

        assert orchestrator.routing is mock_pipelines.routing
        assert orchestrator.spectrum is mock_pipelines.spectrum
        assert orchestrator.grooming is mock_pipelines.grooming
        assert orchestrator.snr is mock_pipelines.snr
        assert orchestrator.slicing is mock_pipelines.slicing

    def test_init_does_not_store_network_state(
        self, mock_config: MagicMock, mock_pipelines: MagicMock
    ) -> None:
        """Orchestrator does not have network_state attribute."""
        from fusion.core.orchestrator import SDNOrchestrator

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)

        assert not hasattr(orchestrator, "_network_state")
        assert not hasattr(orchestrator, "network_state")


# =============================================================================
# handle_arrival Tests - Basic Path
# =============================================================================


class TestHandleArrivalBasicPath:
    """Tests for handle_arrival basic path (no grooming, no slicing)."""

    def test_returns_success_when_allocation_succeeds(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival returns success when spectrum available."""
        # Setup route result
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.paths = (("A", "B", "C"),)
        route_result.modulations = (("QPSK", "8QAM"),)
        route_result.weights_km = (100.0,)
        route_result.has_protection = False
        orchestrator.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 10
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        orchestrator.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_lightpath.request_allocations = {}
        mock_network_state.create_lightpath.return_value = mock_lightpath

        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is True
        orchestrator.routing.find_routes.assert_called_once()
        orchestrator.spectrum.find_spectrum.assert_called()

    def test_returns_failure_when_no_routes(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival returns failure when no routes found."""
        route_result = MagicMock()
        route_result.is_empty = True
        orchestrator.routing.find_routes.return_value = route_result

        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is False
        assert result.block_reason is not None

    def test_returns_failure_when_no_spectrum(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival returns failure when no spectrum on any path."""
        # Setup route result with paths
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.paths = (("A", "B", "C"),)
        route_result.modulations = (("QPSK",),)
        route_result.weights_km = (100.0,)
        route_result.has_protection = False
        orchestrator.routing.find_routes.return_value = route_result

        # Setup spectrum result - no free spectrum
        spectrum_result = MagicMock()
        spectrum_result.is_free = False
        orchestrator.spectrum.find_spectrum.return_value = spectrum_result

        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is False

    def test_tries_multiple_modulations(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival tries all modulations before failing."""
        # Setup route result with multiple modulations
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.paths = (("A", "B", "C"),)
        route_result.modulations = (("16QAM", "QPSK", "BPSK"),)
        route_result.weights_km = (100.0,)
        route_result.has_protection = False
        orchestrator.routing.find_routes.return_value = route_result

        # First two modulations fail, third succeeds
        spectrum_results = [
            MagicMock(is_free=False),
            MagicMock(is_free=False),
            MagicMock(
                is_free=True,
                start_slot=0,
                end_slot=10,
                core=0,
                band="c",
                modulation="BPSK",
            ),
        ]
        orchestrator.spectrum.find_spectrum.side_effect = spectrum_results

        # Setup lightpath
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_lightpath.request_allocations = {}
        mock_network_state.create_lightpath.return_value = mock_lightpath

        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is True
        assert orchestrator.spectrum.find_spectrum.call_count == 3


# =============================================================================
# handle_arrival Tests - With Grooming
# =============================================================================


class TestHandleArrivalWithGrooming:
    """Tests for handle_arrival with grooming enabled."""

    def test_returns_groomed_when_fully_groomed(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival returns groomed result when fully groomed."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.grooming_enabled = True
        mock_grooming = MagicMock()
        mock_pipelines.grooming = mock_grooming

        # Setup grooming result - fully groomed
        groom_result = MagicMock()
        groom_result.fully_groomed = True
        groom_result.lightpaths_used = (1, 2)
        mock_grooming.try_groom.return_value = groom_result

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is True
        assert result.is_groomed is True
        mock_grooming.try_groom.assert_called_once()
        # Should not call routing if fully groomed
        mock_pipelines.routing.find_routes.assert_not_called()

    def test_continues_to_routing_when_partially_groomed(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival continues routing when partially groomed."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.grooming_enabled = True
        mock_grooming = MagicMock()
        mock_pipelines.grooming = mock_grooming

        # Setup grooming result - partial
        groom_result = MagicMock()
        groom_result.fully_groomed = False
        groom_result.partially_groomed = True
        groom_result.lightpaths_used = (1,)
        groom_result.remaining_bandwidth_gbps = 50
        groom_result.forced_path = ("A", "B", "C")
        mock_grooming.try_groom.return_value = groom_result

        # Setup route result
        route_result = MagicMock()
        route_result.is_empty = True
        route_result.has_protection = False
        mock_pipelines.routing.find_routes.return_value = route_result

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        orchestrator.handle_arrival(mock_request, mock_network_state)

        # Should call routing after partial grooming
        mock_pipelines.routing.find_routes.assert_called_once()

    def test_rollback_grooming_on_failure(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival rolls back grooming when allocation fails (P3.2.f)."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.grooming_enabled = True
        mock_config.can_partially_serve = False
        mock_grooming = MagicMock()
        mock_pipelines.grooming = mock_grooming

        # Setup partial grooming
        groom_result = MagicMock()
        groom_result.fully_groomed = False
        groom_result.partially_groomed = True
        groom_result.lightpaths_used = (1, 2)
        groom_result.remaining_bandwidth_gbps = 50
        groom_result.forced_path = None
        mock_grooming.try_groom.return_value = groom_result

        # Setup route failure
        route_result = MagicMock()
        route_result.is_empty = True
        route_result.has_protection = False
        mock_pipelines.routing.find_routes.return_value = route_result

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is False
        # Should call rollback_groom
        mock_grooming.rollback_groom.assert_called_once()

    def test_partial_serve_accepts_partial_grooming(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """When can_partially_serve=True, partial grooming is accepted."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.grooming_enabled = True
        mock_config.can_partially_serve = True
        mock_grooming = MagicMock()
        mock_pipelines.grooming = mock_grooming

        # Setup partial grooming
        groom_result = MagicMock()
        groom_result.fully_groomed = False
        groom_result.partially_groomed = True
        groom_result.lightpaths_used = (1,)
        groom_result.remaining_bandwidth_gbps = 50
        groom_result.forced_path = None
        mock_grooming.try_groom.return_value = groom_result

        # Setup route failure
        route_result = MagicMock()
        route_result.is_empty = True
        route_result.has_protection = False
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup bandwidth lookup
        mock_lp = MagicMock()
        mock_lp.request_allocations = {mock_request.request_id: 50}
        mock_network_state.get_lightpath.return_value = mock_lp

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is True
        assert result.is_partially_groomed is True
        # Should NOT rollback
        mock_grooming.rollback_groom.assert_not_called()


# =============================================================================
# handle_arrival Tests - With SNR
# =============================================================================


class TestHandleArrivalWithSNR:
    """Tests for handle_arrival with SNR validation enabled."""

    def test_releases_lightpath_when_snr_fails(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival releases lightpath when SNR validation fails."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.snr_enabled = True
        mock_snr = MagicMock()
        mock_pipelines.snr = mock_snr

        # Setup route result
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.paths = (("A", "B", "C"),)
        route_result.modulations = (("QPSK",),)
        route_result.weights_km = (100.0,)
        route_result.has_protection = False
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 10
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        mock_pipelines.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_network_state.create_lightpath.return_value = mock_lightpath

        # Setup SNR failure
        snr_result = MagicMock()
        snr_result.passed = False
        mock_snr.validate.return_value = snr_result

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        # Should release lightpath after SNR failure
        mock_network_state.release_lightpath.assert_called_with(1)
        assert result.success is False

    def test_congestion_check_releases_on_recheck_failure(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival releases lightpath when congestion check fails (P3.2.h)."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.snr_enabled = True
        mock_config.snr_recheck = True
        mock_snr = MagicMock()
        mock_pipelines.snr = mock_snr

        # Setup route result
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.paths = (("A", "B", "C"),)
        route_result.modulations = (("QPSK",),)
        route_result.weights_km = (100.0,)
        route_result.has_protection = False
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup spectrum result
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 10
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        mock_pipelines.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath
        mock_lightpath = MagicMock()
        mock_lightpath.lightpath_id = 1
        mock_network_state.create_lightpath.return_value = mock_lightpath

        # SNR validation passes
        snr_result = MagicMock()
        snr_result.passed = True
        mock_snr.validate.return_value = snr_result

        # But recheck fails (existing LP would be degraded)
        recheck_result = MagicMock()
        recheck_result.all_pass = False
        recheck_result.degraded_lightpath_ids = (99,)
        mock_snr.recheck_affected.return_value = recheck_result

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        # Should release due to congestion
        mock_network_state.release_lightpath.assert_called_with(1)
        assert result.success is False


# =============================================================================
# handle_arrival Tests - With Protection (P3.2.g)
# =============================================================================


class TestHandleArrivalWithProtection:
    """Tests for handle_arrival with 1+1 protection."""

    def test_uses_protection_flow_when_enabled(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival uses protection flow when protection_enabled."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.protection_enabled = True
        mock_request.protection_required = True

        # Setup routing with protection paths
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.has_protection = True
        route_result.paths = (("A", "B", "C"),)
        route_result.backup_paths = (("A", "D", "C"),)
        route_result.modulations = (("QPSK",),)
        route_result.backup_modulations = (("QPSK",),)
        route_result.weights_km = (100.0,)
        route_result.backup_weights_km = (120.0,)
        mock_pipelines.routing.find_routes.return_value = route_result

        # Setup spectrum for both paths
        spectrum_result = MagicMock()
        spectrum_result.is_free = True
        spectrum_result.start_slot = 0
        spectrum_result.end_slot = 10
        spectrum_result.core = 0
        spectrum_result.band = "c"
        spectrum_result.modulation = "QPSK"
        mock_pipelines.spectrum.find_spectrum.return_value = spectrum_result

        # Setup lightpath creation
        working_lp = MagicMock()
        working_lp.lightpath_id = 1
        backup_lp = MagicMock()
        backup_lp.lightpath_id = 2
        mock_network_state.create_lightpath.side_effect = [working_lp, backup_lp]

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is True
        assert result.is_protected is True
        # Should create 2 lightpaths
        assert mock_network_state.create_lightpath.call_count == 2

    def test_fails_when_no_protection_paths(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival fails when no disjoint backup path available."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.protection_enabled = True
        mock_request.protection_required = True

        # Route result without backup paths
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.has_protection = False  # No backup paths
        route_result.paths = (("A", "B", "C"),)
        mock_pipelines.routing.find_routes.return_value = route_result

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        assert result.success is False

    def test_rollback_working_when_backup_spectrum_fails(
        self,
        mock_config: MagicMock,
        mock_pipelines: MagicMock,
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival rolls back working LP when backup spectrum fails."""
        from fusion.core.orchestrator import SDNOrchestrator

        mock_config.protection_enabled = True
        mock_request.protection_required = True

        # Setup routing with protection paths
        route_result = MagicMock()
        route_result.is_empty = False
        route_result.has_protection = True
        route_result.paths = (("A", "B", "C"),)
        route_result.backup_paths = (("A", "D", "C"),)
        route_result.modulations = (("QPSK",),)
        route_result.backup_modulations = (("QPSK",),)
        route_result.weights_km = (100.0,)
        route_result.backup_weights_km = (120.0,)
        mock_pipelines.routing.find_routes.return_value = route_result

        # Working spectrum succeeds, backup fails
        working_spectrum = MagicMock()
        working_spectrum.is_free = True
        working_spectrum.start_slot = 0
        working_spectrum.end_slot = 10
        working_spectrum.core = 0
        working_spectrum.band = "c"
        working_spectrum.modulation = "QPSK"

        backup_spectrum = MagicMock()
        backup_spectrum.is_free = False  # Backup fails

        mock_pipelines.spectrum.find_spectrum.side_effect = [
            working_spectrum,
            backup_spectrum,
        ]

        # Setup working lightpath
        working_lp = MagicMock()
        working_lp.lightpath_id = 1
        mock_network_state.create_lightpath.return_value = working_lp

        orchestrator = SDNOrchestrator(mock_config, mock_pipelines)
        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        # Should rollback working LP
        mock_network_state.release_lightpath.assert_called_with(1)
        assert result.success is False


# =============================================================================
# handle_release Tests
# =============================================================================


class TestHandleRelease:
    """Tests for handle_release."""

    def test_releases_all_lightpaths(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_release releases all request lightpaths."""
        mock_request.lightpath_ids = [1, 2, 3]
        mock_request.request_id = 1

        # Setup lightpaths - explicitly set protection_lp_id to None
        # to avoid MagicMock auto-creating truthy attributes
        def get_lp(lp_id: int) -> MagicMock:
            lp = MagicMock()
            lp.lightpath_id = lp_id
            lp.request_allocations = {1: 50}
            lp.remaining_bandwidth_gbps = 50
            lp.protection_lp_id = None  # No protection LP
            return lp

        mock_network_state.get_lightpath.side_effect = lambda x: get_lp(x)

        orchestrator.handle_release(mock_request, mock_network_state)

        assert mock_network_state.get_lightpath.call_count == 3
        assert mock_request.lightpath_ids == []

    def test_releases_empty_lightpaths(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_release fully releases lightpaths with no allocations."""
        mock_request.lightpath_ids = [1]
        mock_request.request_id = 1

        mock_lp = MagicMock()
        mock_lp.lightpath_id = 1
        mock_lp.request_allocations = {1: 100}  # Only this request
        mock_lp.remaining_bandwidth_gbps = 0
        mock_lp.protection_lp_id = None  # No protection LP
        mock_network_state.get_lightpath.return_value = mock_lp

        orchestrator.handle_release(mock_request, mock_network_state)

        # Should release lightpath since no more allocations
        mock_network_state.release_lightpath.assert_called_with(1)

    def test_does_not_release_shared_lightpath(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_release does not release lightpath shared with other requests."""
        mock_request.lightpath_ids = [1]
        mock_request.request_id = 1

        mock_lp = MagicMock()
        mock_lp.lightpath_id = 1
        mock_lp.request_allocations = {1: 50, 2: 50}  # Shared with request 2
        mock_lp.remaining_bandwidth_gbps = 0
        mock_lp.protection_lp_id = None  # No protection LP
        mock_network_state.get_lightpath.return_value = mock_lp

        orchestrator.handle_release(mock_request, mock_network_state)

        # Should NOT release lightpath (still has request 2)
        mock_network_state.release_lightpath.assert_not_called()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_missing_lightpath_gracefully(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_release handles missing lightpath gracefully."""
        mock_request.lightpath_ids = [1, 2]
        mock_request.request_id = 1

        # First LP exists, second doesn't
        mock_lp = MagicMock()
        mock_lp.lightpath_id = 1
        mock_lp.request_allocations = {1: 100}
        mock_lp.remaining_bandwidth_gbps = 0
        mock_lp.protection_lp_id = None  # No protection LP
        mock_network_state.get_lightpath.side_effect = [mock_lp, None]

        # Should not raise
        orchestrator.handle_release(mock_request, mock_network_state)

        assert mock_request.lightpath_ids == []

    def test_skips_grooming_when_disabled(
        self,
        orchestrator: "SDNOrchestrator",
        mock_request: MagicMock,
        mock_network_state: MagicMock,
    ) -> None:
        """handle_arrival skips grooming when disabled."""
        # Route fails immediately
        route_result = MagicMock()
        route_result.is_empty = True
        orchestrator.routing.find_routes.return_value = route_result

        result = orchestrator.handle_arrival(mock_request, mock_network_state)

        # Grooming should not be called
        assert orchestrator.grooming is None
        assert result.success is False
