"""Tests for StatsCollector."""

from __future__ import annotations

import pytest

from fusion.domain.config import SimulationConfig
from fusion.domain.request import BlockReason, Request
from fusion.domain.results import (
    AllocationResult,
    GroomingResult,
    ProtectionResult,
    RouteResult,
    SNRResult,
    SpectrumResult,
)
from fusion.stats.collector import StatsCollector

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> SimulationConfig:
    """Create a standard config for testing."""
    return SimulationConfig(
        network_name="NSFNET",
        cores_per_link=7,
        band_list=("c",),
        band_slots={"c": 320},
        guard_slots=1,
        num_requests=1000,
        erlang=50.0,
        holding_time=5.0,
        route_method="k_shortest_path",
        k_paths=3,
        allocation_method="first_fit",
        grooming_enabled=True,
        slicing_enabled=True,
        snr_enabled=True,
    )


@pytest.fixture
def collector(config: SimulationConfig) -> StatsCollector:
    """Create a fresh StatsCollector for each test."""
    return StatsCollector(config)


@pytest.fixture
def sample_request() -> Request:
    """Create a sample request for testing."""
    return Request(
        request_id=1,
        source="0",
        destination="5",
        bandwidth_gbps=100,
        arrive_time=0.0,
        depart_time=5.0,
    )


@pytest.fixture
def sample_spectrum_result() -> SpectrumResult:
    """Create a sample spectrum result."""
    return SpectrumResult(
        is_free=True,
        start_slot=0,
        end_slot=8,
        core=0,
        band="c",
        modulation="QPSK",
        slots_needed=8,
    )


@pytest.fixture
def sample_snr_result() -> SNRResult:
    """Create a sample SNR result."""
    return SNRResult.success(snr_db=18.5, required_snr_db=15.0)


@pytest.fixture
def sample_route_result() -> RouteResult:
    """Create a sample route result."""
    return RouteResult(
        paths=(("0", "2", "5"),),
        weights_km=(150.0,),
        modulations=(("QPSK", "16-QAM"),),
        strategy_name="k_shortest_path",
    )


# =============================================================================
# Test StatsCollector Creation
# =============================================================================


class TestStatsCollectorCreation:
    """Test StatsCollector initialization."""

    def test_create_collector(self, config: SimulationConfig) -> None:
        """Test basic collector creation."""
        collector = StatsCollector(config)

        assert collector.config is config
        assert collector.total_requests == 0
        assert collector.successful_requests == 0
        assert collector.blocked_requests == 0

    def test_initial_counters_are_zero(self, collector: StatsCollector) -> None:
        """Test that all initial counters are zero."""
        assert collector.total_requests == 0
        assert collector.successful_requests == 0
        assert collector.blocked_requests == 0
        assert collector.groomed_requests == 0
        assert collector.partially_groomed_requests == 0
        assert collector.sliced_requests == 0
        assert collector.protected_requests == 0
        assert collector.switchover_count == 0
        assert collector.protection_failures == 0
        assert collector.failure_induced_blocks == 0

    def test_initial_ratios_are_zero(self, collector: StatsCollector) -> None:
        """Test that initial ratios are zero (no division by zero)."""
        assert collector.blocking_probability == 0.0
        assert collector.success_rate == 1.0
        assert collector.average_snr == 0.0
        assert collector.grooming_ratio == 0.0
        assert collector.slicing_ratio == 0.0
        assert collector.protection_ratio == 0.0
        assert collector.bandwidth_utilization == 0.0

    def test_initial_lists_are_empty(self, collector: StatsCollector) -> None:
        """Test that initial lists are empty."""
        assert len(collector.snr_values) == 0
        assert len(collector.hop_counts) == 0
        assert len(collector.path_lengths_km) == 0
        assert len(collector.switchover_times) == 0
        assert len(collector.xt_values) == 0

    def test_initial_dicts_are_empty(self, collector: StatsCollector) -> None:
        """Test that initial dicts are empty."""
        assert len(collector.block_reasons) == 0
        assert len(collector.modulations_used) == 0
        assert len(collector.core_usage) == 0
        assert len(collector.band_usage) == 0


# =============================================================================
# Test record_arrival - Success Cases
# =============================================================================


class TestRecordArrivalSuccess:
    """Test record_arrival method for successful allocations."""

    def test_record_success_basic(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording successful allocation."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
        )

        collector.record_arrival(sample_request, result)

        assert collector.total_requests == 1
        assert collector.successful_requests == 1
        assert collector.blocked_requests == 0
        assert collector.blocking_probability == 0.0
        assert collector.success_rate == 1.0

    def test_record_success_bandwidth_tracking(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test bandwidth tracking on success."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
        )

        collector.record_arrival(sample_request, result)

        assert collector.total_bandwidth_requested_gbps == 100
        assert collector.total_bandwidth_allocated_gbps == 100
        assert collector.bandwidth_utilization == 1.0

    def test_record_success_with_spectrum(
        self,
        collector: StatsCollector,
        sample_request: Request,
        sample_spectrum_result: SpectrumResult,
    ) -> None:
        """Test modulation and core/band tracking from spectrum result."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
            spectrum_result=sample_spectrum_result,
        )

        collector.record_arrival(sample_request, result)

        assert collector.modulations_used["QPSK"] >= 1
        assert collector.core_usage[0] >= 1
        assert collector.band_usage["c"] >= 1

    def test_record_success_with_snr(
        self,
        collector: StatsCollector,
        sample_request: Request,
        sample_snr_result: SNRResult,
    ) -> None:
        """Test SNR tracking from result."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
            snr_result=sample_snr_result,
        )

        collector.record_arrival(sample_request, result)

        assert len(collector.snr_values) == 1
        assert collector.snr_values[0] == 18.5
        assert collector.average_snr == 18.5

    def test_record_success_with_route(
        self,
        collector: StatsCollector,
        sample_request: Request,
        sample_route_result: RouteResult,
    ) -> None:
        """Test path tracking from route result."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
            route_result=sample_route_result,
        )

        collector.record_arrival(sample_request, result)

        assert len(collector.hop_counts) == 1
        assert collector.hop_counts[0] == 2  # 3 nodes = 2 hops
        assert len(collector.path_lengths_km) == 1
        assert collector.path_lengths_km[0] == 150.0
        assert collector.average_hop_count == 2.0
        assert collector.average_path_length_km == 150.0

    def test_record_groomed_request(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording groomed request."""
        result = AllocationResult.success_groomed(
            lightpath_ids=[10],
            bandwidth_gbps=100,
        )

        collector.record_arrival(sample_request, result)

        assert collector.groomed_requests == 1
        assert collector.grooming_ratio == 1.0

    def test_record_partially_groomed_request(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording partially groomed request."""
        result = AllocationResult.success_partial_groom(
            groomed_ids=[10],
            new_lightpath_id=20,
            total_bandwidth=100,
        )

        collector.record_arrival(sample_request, result)

        assert collector.partially_groomed_requests == 1
        assert collector.grooming_ratio == 1.0  # Includes partial

    def test_record_sliced_request(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording sliced request."""
        result = AllocationResult.success_sliced(
            lightpath_ids=[1, 2, 3, 4],
            bandwidth_gbps=100,
        )

        collector.record_arrival(sample_request, result)

        assert collector.sliced_requests == 1
        assert collector.slicing_ratio == 1.0

    def test_record_protected_request(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording protected request."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
            is_protected=True,
        )

        collector.record_arrival(sample_request, result)

        assert collector.protected_requests == 1
        assert collector.protection_ratio == 1.0


# =============================================================================
# Test record_arrival - Blocked Cases
# =============================================================================


class TestRecordArrivalBlocked:
    """Test record_arrival method for blocked requests."""

    def test_record_blocked_basic(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording blocked request."""
        result = AllocationResult.blocked(BlockReason.CONGESTION)

        collector.record_arrival(sample_request, result)

        assert collector.total_requests == 1
        assert collector.successful_requests == 0
        assert collector.blocked_requests == 1
        assert collector.blocking_probability == 1.0
        assert collector.success_rate == 0.0

    def test_record_blocked_tracks_reason(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test that block reasons are tracked."""
        result = AllocationResult.blocked(BlockReason.CONGESTION)

        collector.record_arrival(sample_request, result)

        assert collector.block_reasons["congestion"] == 1

    def test_record_blocked_bandwidth_not_allocated(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test that blocked requests don't count as allocated bandwidth."""
        result = AllocationResult.blocked(BlockReason.CONGESTION)

        collector.record_arrival(sample_request, result)

        assert collector.total_bandwidth_requested_gbps == 100
        assert collector.total_bandwidth_allocated_gbps == 0
        assert collector.bandwidth_utilization == 0.0

    def test_record_blocked_no_path(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording blocked request due to no path."""
        result = AllocationResult.blocked(BlockReason.NO_PATH)

        collector.record_arrival(sample_request, result)

        assert collector.block_reasons["no_path"] == 1

    def test_record_blocked_snr_fail(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording blocked request due to SNR failure."""
        result = AllocationResult.blocked(BlockReason.SNR_THRESHOLD)

        collector.record_arrival(sample_request, result)

        assert collector.block_reasons["snr_fail"] == 1

    def test_record_blocked_failure_induced(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording failure-induced block."""
        result = AllocationResult.blocked(BlockReason.LINK_FAILURE)

        collector.record_arrival(sample_request, result)

        assert collector.block_reasons["link_failure"] == 1
        assert collector.failure_induced_blocks == 1

    def test_record_multiple_block_reasons(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test tracking multiple different block reasons."""
        reasons = [
            BlockReason.CONGESTION,
            BlockReason.CONGESTION,
            BlockReason.NO_PATH,
            BlockReason.SNR_THRESHOLD,
        ]

        for reason in reasons:
            result = AllocationResult.blocked(reason)
            collector.record_arrival(sample_request, result)

        assert collector.block_reasons["congestion"] == 2
        assert collector.block_reasons["no_path"] == 1
        assert collector.block_reasons["snr_fail"] == 1
        assert collector.blocked_requests == 4


# =============================================================================
# Test record_arrival - Multiple Arrivals
# =============================================================================


class TestRecordMultipleArrivals:
    """Test recording multiple arrivals."""

    def test_record_multiple_arrivals_mixed(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test recording multiple arrivals with mixed results."""
        success = AllocationResult.success_new_lightpath(1, 100)
        blocked = AllocationResult.blocked(BlockReason.CONGESTION)

        for _ in range(8):
            collector.record_arrival(sample_request, success)
        for _ in range(2):
            collector.record_arrival(sample_request, blocked)

        assert collector.total_requests == 10
        assert collector.successful_requests == 8
        assert collector.blocked_requests == 2
        assert collector.blocking_probability == 0.2
        assert collector.success_rate == 0.8

    def test_record_multiple_modulations(
        self,
        collector: StatsCollector,
        sample_request: Request,
    ) -> None:
        """Test tracking multiple modulation formats."""
        modulations = ["QPSK", "16-QAM", "QPSK", "8-QAM", "QPSK"]

        for mod in modulations:
            spectrum = SpectrumResult(
                is_free=True,
                start_slot=0,
                end_slot=8,
                modulation=mod,
                slots_needed=8,
            )
            result = AllocationResult.success_new_lightpath(
                lightpath_id=1,
                bandwidth_gbps=100,
                spectrum_result=spectrum,
            )
            collector.record_arrival(sample_request, result)

        assert collector.modulations_used["QPSK"] == 3
        assert collector.modulations_used["16-QAM"] == 1
        assert collector.modulations_used["8-QAM"] == 1


# =============================================================================
# Test record_snr
# =============================================================================


class TestRecordSNR:
    """Test record_snr method."""

    def test_record_single_snr(self, collector: StatsCollector) -> None:
        """Test recording single SNR value."""
        collector.record_snr(18.5)

        assert len(collector.snr_values) == 1
        assert collector.average_snr == 18.5
        assert collector.min_snr == 18.5
        assert collector.max_snr == 18.5

    def test_record_multiple_snr(self, collector: StatsCollector) -> None:
        """Test recording multiple SNR values."""
        collector.record_snr(18.0)
        collector.record_snr(16.0)
        collector.record_snr(20.0)

        assert len(collector.snr_values) == 3
        assert collector.average_snr == 18.0
        assert collector.min_snr == 16.0
        assert collector.max_snr == 20.0


# =============================================================================
# Test record_xt
# =============================================================================


class TestRecordXT:
    """Test record_xt method."""

    def test_record_single_xt(self, collector: StatsCollector) -> None:
        """Test recording single XT value."""
        collector.record_xt(-30.5)

        assert len(collector.xt_values) == 1
        assert collector.xt_values[0] == -30.5

    def test_record_multiple_xt(self, collector: StatsCollector) -> None:
        """Test recording multiple XT values."""
        collector.record_xt(-30.0)
        collector.record_xt(-28.0)
        collector.record_xt(-32.0)

        assert len(collector.xt_values) == 3


# =============================================================================
# Test record_release
# =============================================================================


class TestRecordRelease:
    """Test record_release method."""

    def test_record_release_noop(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test that record_release is a no-op in Phase 1."""
        # Should not raise
        collector.record_release(sample_request)

        # No state changes
        assert collector.total_requests == 0


# =============================================================================
# Test record_switchover
# =============================================================================


class TestRecordSwitchover:
    """Test record_switchover method."""

    def test_record_switchover_success(self, collector: StatsCollector) -> None:
        """Test recording successful switchover."""
        collector.record_switchover(50.0, success=True)

        assert collector.switchover_count == 1
        assert len(collector.switchover_times) == 1
        assert collector.switchover_times[0] == 50.0
        assert collector.protection_failures == 0

    def test_record_switchover_failure(self, collector: StatsCollector) -> None:
        """Test recording failed switchover."""
        collector.record_switchover(0.0, success=False)

        assert collector.switchover_count == 1
        assert len(collector.switchover_times) == 0
        assert collector.protection_failures == 1


# =============================================================================
# Test Computed Properties
# =============================================================================


class TestComputedProperties:
    """Test computed properties."""

    def test_blocking_probability_calculation(
        self, collector: StatsCollector
    ) -> None:
        """Test blocking probability calculation."""
        collector.total_requests = 100
        collector.blocked_requests = 15

        assert collector.blocking_probability == 0.15

    def test_success_rate_calculation(self, collector: StatsCollector) -> None:
        """Test success rate calculation."""
        collector.total_requests = 100
        collector.blocked_requests = 15

        assert collector.success_rate == 0.85

    def test_average_snr_empty(self, collector: StatsCollector) -> None:
        """Test average SNR with no values."""
        assert collector.average_snr == 0.0
        assert collector.min_snr == 0.0
        assert collector.max_snr == 0.0

    def test_grooming_ratio_includes_partial(
        self, collector: StatsCollector
    ) -> None:
        """Test that grooming ratio includes partially groomed."""
        collector.successful_requests = 100
        collector.groomed_requests = 20
        collector.partially_groomed_requests = 10

        assert collector.grooming_ratio == 0.3

    def test_bandwidth_utilization(self, collector: StatsCollector) -> None:
        """Test bandwidth utilization calculation."""
        collector.total_bandwidth_requested_gbps = 1000
        collector.total_bandwidth_allocated_gbps = 850

        assert collector.bandwidth_utilization == 0.85


# =============================================================================
# Test to_comparison_format
# =============================================================================


class TestToComparisonFormat:
    """Test to_comparison_format method."""

    def test_basic_export(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test basic export format."""
        result = AllocationResult.success_new_lightpath(42, 100)
        collector.record_arrival(sample_request, result)

        output = collector.to_comparison_format()

        assert "blocking_probability" in output
        assert "success_rate" in output
        assert "total_requests" in output
        assert "successful_requests" in output
        assert "blocked_requests" in output
        assert "block_reasons" in output
        assert "modulations_used" in output
        assert "config" in output

    def test_export_contains_config(self, collector: StatsCollector) -> None:
        """Test that export contains config context."""
        output = collector.to_comparison_format()

        assert output["config"]["network_name"] == "NSFNET"
        assert output["config"]["erlang"] == 50.0
        assert output["config"]["grooming_enabled"] is True
        assert output["config"]["slicing_enabled"] is True
        assert output["config"]["snr_enabled"] is True
        assert output["config"]["cores_per_link"] == 7

    def test_export_block_reasons_is_regular_dict(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test that block_reasons is exported as regular dict."""
        result = AllocationResult.blocked(BlockReason.NO_PATH)
        collector.record_arrival(sample_request, result)

        output = collector.to_comparison_format()

        # Should be regular dict, not defaultdict
        assert type(output["block_reasons"]) is dict

    def test_export_snr_values_is_copy(
        self, collector: StatsCollector
    ) -> None:
        """Test that snr_values is a copy."""
        collector.record_snr(18.5)
        output = collector.to_comparison_format()

        # Modify the output
        output["snr_values"].append(999.0)

        # Original should not be affected
        assert len(collector.snr_values) == 1

    def test_export_contains_protection_metrics(
        self, collector: StatsCollector
    ) -> None:
        """Test that export contains protection metrics."""
        collector.record_switchover(50.0, success=True)
        collector.record_switchover(0.0, success=False)

        output = collector.to_comparison_format()

        assert output["switchover_count"] == 2
        assert output["protection_failures"] == 1
        assert len(output["switchover_times"]) == 1

    def test_export_contains_path_metrics(
        self,
        collector: StatsCollector,
        sample_request: Request,
        sample_route_result: RouteResult,
    ) -> None:
        """Test that export contains path metrics."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
            route_result=sample_route_result,
        )
        collector.record_arrival(sample_request, result)

        output = collector.to_comparison_format()

        assert output["average_hop_count"] == 2.0
        assert output["average_path_length_km"] == 150.0

    def test_export_contains_xt_metrics(
        self, collector: StatsCollector
    ) -> None:
        """Test that export contains crosstalk metrics."""
        collector.record_xt(-30.0)
        collector.record_xt(-28.0)

        output = collector.to_comparison_format()

        assert "xt_values" in output
        assert output["xt_values"] == [-30.0, -28.0]
        assert output["average_xt"] == -29.0


# =============================================================================
# Test to_legacy_stats_dict
# =============================================================================


class TestToLegacyStatsDict:
    """Test to_legacy_stats_dict method."""

    def test_legacy_format(self, collector: StatsCollector) -> None:
        """Test legacy export format."""
        collector.total_requests = 100
        collector.blocked_requests = 20

        legacy = collector.to_legacy_stats_dict()

        assert "simulation_blocking_list" in legacy
        assert legacy["simulation_blocking_list"] == [0.2]
        assert "block_reasons_dict" in legacy
        assert "modulations_used_dict" in legacy
        assert "snr_list" in legacy

    def test_legacy_contains_protection_stats(
        self, collector: StatsCollector
    ) -> None:
        """Test legacy format contains protection stats."""
        collector.switchover_count = 5
        collector.protection_failures = 2

        legacy = collector.to_legacy_stats_dict()

        assert legacy["protection_switchovers"] == 5
        assert legacy["protection_failures"] == 2

    def test_legacy_contains_crosstalk_list(
        self, collector: StatsCollector
    ) -> None:
        """Test legacy format contains crosstalk_list."""
        collector.record_xt(-30.0)
        collector.record_xt(-28.0)

        legacy = collector.to_legacy_stats_dict()

        assert "crosstalk_list" in legacy
        assert legacy["crosstalk_list"] == [-30.0, -28.0]


# =============================================================================
# Test reset
# =============================================================================


class TestReset:
    """Test reset method."""

    def test_reset_counters(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test that reset clears counters."""
        result = AllocationResult.success_new_lightpath(42, 100)
        collector.record_arrival(sample_request, result)
        collector.record_snr(18.5)

        collector.reset()

        assert collector.total_requests == 0
        assert collector.successful_requests == 0
        assert collector.blocked_requests == 0
        assert len(collector.snr_values) == 0
        assert len(collector.block_reasons) == 0
        assert len(collector.modulations_used) == 0

    def test_reset_preserves_config(
        self, collector: StatsCollector, config: SimulationConfig
    ) -> None:
        """Test that reset preserves config."""
        collector.reset()

        assert collector.config is config

    def test_reset_clears_all_lists(
        self, collector: StatsCollector
    ) -> None:
        """Test that reset clears all lists."""
        collector.snr_values.append(18.5)
        collector.hop_counts.append(3)
        collector.path_lengths_km.append(100.0)
        collector.switchover_times.append(50.0)
        collector.xt_values.append(-30.0)

        collector.reset()

        assert len(collector.snr_values) == 0
        assert len(collector.hop_counts) == 0
        assert len(collector.path_lengths_km) == 0
        assert len(collector.switchover_times) == 0
        assert len(collector.xt_values) == 0

    def test_reset_clears_all_dicts(
        self, collector: StatsCollector
    ) -> None:
        """Test that reset clears all dicts."""
        collector.block_reasons["congestion"] = 5
        collector.modulations_used["QPSK"] = 10
        collector.core_usage[0] = 20
        collector.band_usage["c"] = 30

        collector.reset()

        assert len(collector.block_reasons) == 0
        assert len(collector.modulations_used) == 0
        assert len(collector.core_usage) == 0
        assert len(collector.band_usage) == 0


# =============================================================================
# Test merge
# =============================================================================


class TestMerge:
    """Test merge method."""

    def test_merge_counters(self, config: SimulationConfig) -> None:
        """Test merging counters from two collectors."""
        collector1 = StatsCollector(config)
        collector2 = StatsCollector(config)

        collector1.total_requests = 50
        collector1.successful_requests = 40
        collector1.blocked_requests = 10

        collector2.total_requests = 50
        collector2.successful_requests = 45
        collector2.blocked_requests = 5

        collector1.merge(collector2)

        assert collector1.total_requests == 100
        assert collector1.successful_requests == 85
        assert collector1.blocked_requests == 15

    def test_merge_block_reasons(self, config: SimulationConfig) -> None:
        """Test merging block reasons."""
        collector1 = StatsCollector(config)
        collector2 = StatsCollector(config)

        collector1.block_reasons["congestion"] = 5
        collector1.block_reasons["no_path"] = 3

        collector2.block_reasons["congestion"] = 7
        collector2.block_reasons["snr_fail"] = 2

        collector1.merge(collector2)

        assert collector1.block_reasons["congestion"] == 12
        assert collector1.block_reasons["no_path"] == 3
        assert collector1.block_reasons["snr_fail"] == 2

    def test_merge_lists(self, config: SimulationConfig) -> None:
        """Test merging lists."""
        collector1 = StatsCollector(config)
        collector2 = StatsCollector(config)

        collector1.snr_values = [18.0, 19.0]
        collector2.snr_values = [17.0, 20.0]

        collector1.merge(collector2)

        assert collector1.snr_values == [18.0, 19.0, 17.0, 20.0]

    def test_merge_xt_values(self, config: SimulationConfig) -> None:
        """Test merging XT values."""
        collector1 = StatsCollector(config)
        collector2 = StatsCollector(config)

        collector1.xt_values = [-30.0, -28.0]
        collector2.xt_values = [-32.0, -29.0]

        collector1.merge(collector2)

        assert collector1.xt_values == [-30.0, -28.0, -32.0, -29.0]


# =============================================================================
# Test String Representations
# =============================================================================


class TestStringRepresentations:
    """Test __str__ and __repr__ methods."""

    def test_str(self, collector: StatsCollector) -> None:
        """Test string representation."""
        collector.total_requests = 100
        collector.successful_requests = 85
        collector.blocked_requests = 15

        s = str(collector)

        assert "100" in s
        assert "85" in s
        assert "15" in s
        assert "15.00%" in s

    def test_repr(self, collector: StatsCollector) -> None:
        """Test detailed representation."""
        r = repr(collector)

        assert "StatsCollector" in r
        assert "NSFNET" in r


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for StatsCollector."""

    def test_full_simulation_flow(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test full simulation recording flow."""
        # Create spectrum result with modulation
        spectrum = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        snr = SNRResult.success(18.0, 15.0)
        route = RouteResult(
            paths=(("0", "2", "5"),),
            weights_km=(150.0,),
            modulations=(("QPSK",),),
            strategy_name="ksp",
        )

        for i in range(100):
            if i < 70:
                # Success
                result = AllocationResult.success_new_lightpath(
                    i,
                    100,
                    spectrum_result=spectrum,
                    snr_result=snr,
                    route_result=route,
                )
            elif i < 85:
                # Blocked - congestion
                result = AllocationResult.blocked(BlockReason.CONGESTION)
            elif i < 95:
                # Blocked - SNR failure
                result = AllocationResult.blocked(BlockReason.SNR_THRESHOLD)
            else:
                # Blocked - no path
                result = AllocationResult.blocked(BlockReason.NO_PATH)

            collector.record_arrival(sample_request, result)

        # Verify final statistics
        assert collector.total_requests == 100
        assert collector.successful_requests == 70
        assert collector.blocked_requests == 30
        assert collector.blocking_probability == 0.3

        assert collector.block_reasons["congestion"] == 15
        assert collector.block_reasons["snr_fail"] == 10
        assert collector.block_reasons["no_path"] == 5

        assert collector.modulations_used["QPSK"] >= 70
        assert len(collector.snr_values) >= 70

        # Export and verify
        output = collector.to_comparison_format()
        assert output["blocking_probability"] == 0.3
        assert output["block_reasons"]["congestion"] == 15

    def test_reset_and_reuse(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test resetting and reusing collector."""
        # First run
        for _ in range(50):
            result = AllocationResult.success_new_lightpath(1, 100)
            collector.record_arrival(sample_request, result)

        assert collector.total_requests == 50

        # Reset
        collector.reset()
        assert collector.total_requests == 0

        # Second run
        for _ in range(30):
            result = AllocationResult.blocked(BlockReason.CONGESTION)
            collector.record_arrival(sample_request, result)

        assert collector.total_requests == 30
        assert collector.blocked_requests == 30
        assert collector.blocking_probability == 1.0

    def test_full_features_request(
        self,
        collector: StatsCollector,
        sample_request: Request,
        sample_spectrum_result: SpectrumResult,
        sample_snr_result: SNRResult,
        sample_route_result: RouteResult,
    ) -> None:
        """Test request with all features enabled."""
        # Create a complex result with all features
        grooming = GroomingResult.partial(
            bandwidth_groomed=50,
            remaining=50,
            lightpath_ids=[1],
        )
        _protection = ProtectionResult.established(
            primary_spectrum=sample_spectrum_result,
            backup_spectrum=sample_spectrum_result,
        )

        result = AllocationResult.success_partial_groom(
            groomed_ids=[1],
            new_lightpath_id=2,
            total_bandwidth=100,
            grooming_result=grooming,
            route_result=sample_route_result,
            spectrum_result=sample_spectrum_result,
            snr_result=sample_snr_result,
        )

        collector.record_arrival(sample_request, result)

        assert collector.successful_requests == 1
        assert collector.partially_groomed_requests == 1
        assert collector.grooming_ratio == 1.0

    def test_per_segment_tracking(
        self, collector: StatsCollector, sample_request: Request
    ) -> None:
        """Test tracking of per-segment allocation data."""
        # Create result with per-segment data
        result = AllocationResult(
            success=True,
            lightpaths_created=(1, 2),
            total_bandwidth_allocated_gbps=100,
            is_sliced=True,
            modulations=("QPSK", "16-QAM"),
            cores=(0, 1),
            bands=("c", "l"),
            xt_costs=(0.001, 0.002),
            xt_values=(0.0015, 0.0025),
            snr_values=(18.0, 16.0),
        )

        collector.record_arrival(sample_request, result)

        # Check modulations tracked from per-segment
        assert "QPSK" in collector.modulations_used
        assert "16-QAM" in collector.modulations_used

        # Check cores tracked
        assert 0 in collector.core_usage
        assert 1 in collector.core_usage

        # Check bands tracked
        assert "c" in collector.band_usage
        assert "l" in collector.band_usage

        # Check XT values tracked
        assert len(collector.xt_values) == 4  # 2 from xt_costs + 2 from xt_values

        # Check SNR values tracked
        assert 18.0 in collector.snr_values
        assert 16.0 in collector.snr_values

    def test_protection_result_tracking(
        self,
        collector: StatsCollector,
        sample_request: Request,
        sample_spectrum_result: SpectrumResult,
    ) -> None:
        """Test protection result tracking."""
        # Create result with switchover protection result
        protection = ProtectionResult.switchover(
            success=True,
            switchover_time_ms=50.0,
            failure_type="link_failure",
        )

        result = AllocationResult(
            success=True,
            lightpaths_created=(1,),
            total_bandwidth_allocated_gbps=100,
            is_protected=True,
            protection_result=protection,
        )

        collector.record_arrival(sample_request, result)

        assert collector.protected_requests == 1
        assert collector.switchover_count == 1
        assert 50.0 in collector.switchover_times

    def test_protection_failure_tracking(
        self,
        collector: StatsCollector,
        sample_request: Request,
    ) -> None:
        """Test protection failure tracking when not fully protected."""
        # Create result with partial protection (backup failed)
        protection = ProtectionResult.primary_only(
            primary_spectrum=SpectrumResult(
                is_free=True, start_slot=0, end_slot=8, slots_needed=8
            )
        )

        result = AllocationResult(
            success=True,
            lightpaths_created=(1,),
            total_bandwidth_allocated_gbps=100,
            is_protected=True,
            protection_result=protection,
        )

        collector.record_arrival(sample_request, result)

        assert collector.protected_requests == 1
        assert collector.protection_failures == 1
