"""Tests for result dataclasses."""

from __future__ import annotations

import pytest

from fusion.domain.request import BlockReason
from fusion.domain.results import (
    AllocationResult,
    GroomingResult,
    ProtectionResult,
    RouteResult,
    SlicingResult,
    SNRRecheckResult,
    SNRResult,
    SpectrumResult,
)

# =============================================================================
# RouteResult Tests
# =============================================================================


class TestRouteResult:
    """Test RouteResult dataclass."""

    def test_create_basic_route_result(self) -> None:
        """Test creating basic route result."""
        result = RouteResult(
            paths=(("0", "2", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK", "16-QAM"),),
            strategy_name="k_shortest_path",
        )
        assert result.num_paths == 1
        assert not result.is_empty
        assert result.best_path == ("0", "2", "5")
        assert result.best_weight == 100.0
        assert result.strategy_name == "k_shortest_path"

    def test_create_multi_path_route_result(self) -> None:
        """Test creating route result with multiple paths."""
        result = RouteResult(
            paths=(("0", "2", "5"), ("0", "3", "5")),
            weights_km=(100.0, 120.0),
            modulations=(("QPSK", "16-QAM"), ("QPSK",)),
        )
        assert result.num_paths == 2
        assert result.best_path == ("0", "2", "5")
        assert result.get_path(1) == ("0", "3", "5")

    def test_create_empty_route_result(self) -> None:
        """Test creating empty route result."""
        result = RouteResult.empty(strategy_name="ksp")
        assert result.is_empty
        assert result.num_paths == 0
        assert result.best_path is None
        assert result.best_weight is None
        assert result.strategy_name == "ksp"

    def test_route_result_with_protection(self) -> None:
        """Test route result with backup paths."""
        result = RouteResult(
            paths=(("0", "2", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK",),),
            backup_paths=(("0", "3", "5"),),
            backup_weights_km=(120.0,),
            backup_modulations=(("QPSK",),),
        )
        assert result.has_protection
        assert result.backup_paths == (("0", "3", "5"),)

    def test_route_result_no_protection(self) -> None:
        """Test route result without backup paths."""
        result = RouteResult(
            paths=(("0", "5"),),
            weights_km=(50.0,),
            modulations=(("QPSK",),),
        )
        assert not result.has_protection

    def test_route_result_immutable(self) -> None:
        """Test route result is frozen."""
        result = RouteResult.empty()
        with pytest.raises(AttributeError):
            result.strategy_name = "modified"  # type: ignore[misc]

    def test_route_result_validation_length_mismatch_weights(self) -> None:
        """Test validation catches length mismatch with weights."""
        with pytest.raises(ValueError, match="same length"):
            RouteResult(
                paths=(("0", "5"),),
                weights_km=(100.0, 200.0),  # Wrong length
                modulations=(("QPSK",),),
            )

    def test_route_result_validation_length_mismatch_modulations(self) -> None:
        """Test validation catches length mismatch with modulations."""
        with pytest.raises(ValueError, match="same length"):
            RouteResult(
                paths=(("0", "5"),),
                weights_km=(100.0,),
                modulations=(("QPSK",), ("BPSK",)),  # Wrong length
            )

    def test_route_result_validation_short_path(self) -> None:
        """Test validation catches single-node path."""
        with pytest.raises(ValueError, match="at least 2 nodes"):
            RouteResult(
                paths=(("0",),),  # Too short
                weights_km=(100.0,),
                modulations=(("QPSK",),),
            )

    def test_route_result_validation_negative_weight(self) -> None:
        """Test validation catches negative weight."""
        with pytest.raises(ValueError, match="non-negative"):
            RouteResult(
                paths=(("0", "5"),),
                weights_km=(-100.0,),  # Negative
                modulations=(("QPSK",),),
            )

    def test_route_result_validation_backup_length_mismatch(self) -> None:
        """Test validation catches backup paths length mismatch."""
        with pytest.raises(ValueError, match="backup_paths must match"):
            RouteResult(
                paths=(("0", "5"),),
                weights_km=(100.0,),
                modulations=(("QPSK",),),
                backup_paths=(("0", "3", "5"), ("0", "4", "5")),  # Wrong length
            )

    def test_route_result_get_modulations(self) -> None:
        """Test get_modulations_for_path method."""
        result = RouteResult(
            paths=(("0", "5"), ("0", "3", "5")),
            weights_km=(100.0, 120.0),
            modulations=(("QPSK", "16-QAM"), ("QPSK",)),
        )
        assert result.get_modulations_for_path(0) == ("QPSK", "16-QAM")
        assert result.get_modulations_for_path(1) == ("QPSK",)

    def test_route_result_with_metadata(self) -> None:
        """Test route result with metadata."""
        result = RouteResult(
            paths=(("0", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK",),),
            metadata={"k": 3, "algorithm": "yen"},
        )
        assert result.metadata["k"] == 3
        assert result.metadata["algorithm"] == "yen"


# =============================================================================
# SpectrumResult Tests
# =============================================================================


class TestSpectrumResult:
    """Test SpectrumResult dataclass."""

    def test_create_successful_spectrum(self) -> None:
        """Test creating successful spectrum result."""
        result = SpectrumResult(
            is_free=True,
            start_slot=100,
            end_slot=108,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        assert result.is_free
        assert result.num_slots == 8
        assert result.start_slot == 100
        assert result.end_slot == 108
        assert result.core == 0
        assert result.band == "c"
        assert result.modulation == "QPSK"

    def test_create_failed_spectrum(self) -> None:
        """Test creating failed spectrum result."""
        result = SpectrumResult.not_found(slots_needed=8)
        assert not result.is_free
        assert result.num_slots == 0
        assert result.slots_needed == 8

    def test_spectrum_with_backup(self) -> None:
        """Test spectrum result with backup allocation."""
        result = SpectrumResult(
            is_free=True,
            start_slot=100,
            end_slot=108,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
            backup_start_slot=200,
            backup_end_slot=208,
            backup_core=1,
            backup_band="c",
        )
        assert result.has_backup
        assert result.backup_start_slot == 200
        assert result.backup_end_slot == 208

    def test_spectrum_no_backup(self) -> None:
        """Test spectrum result without backup."""
        result = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        assert not result.has_backup

    def test_spectrum_validation_invalid_slots(self) -> None:
        """Test validation when is_free but invalid slots."""
        with pytest.raises(ValueError, match="end_slot must be > start_slot"):
            SpectrumResult(
                is_free=True,
                start_slot=100,
                end_slot=100,  # Invalid
                slots_needed=8,
            )

    def test_spectrum_validation_zero_slots_needed(self) -> None:
        """Test validation when is_free but zero slots_needed."""
        with pytest.raises(ValueError, match="slots_needed must be > 0"):
            SpectrumResult(
                is_free=True,
                start_slot=0,
                end_slot=8,
                slots_needed=0,  # Invalid
            )

    def test_spectrum_to_allocation_dict(self) -> None:
        """Test conversion to allocation dict."""
        result = SpectrumResult(
            is_free=True,
            start_slot=100,
            end_slot=108,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        d = result.to_allocation_dict()
        assert d["is_free"] is True
        assert d["start_slot"] == 100
        assert d["end_slot"] == 108
        assert d["core_number"] == 0
        assert d["band"] == "c"
        assert d["modulation"] == "QPSK"
        assert d["slots_needed"] == 8

    def test_spectrum_from_dict(self) -> None:
        """Test creation from dict."""
        d = {
            "is_free": True,
            "start_slot": 50,
            "end_slot": 58,
            "core_number": 1,
            "band": "l",
            "modulation": "16-QAM",
            "slots_needed": 8,
        }
        result = SpectrumResult.from_spectrum_props(d)
        assert result.is_free
        assert result.start_slot == 50
        assert result.core == 1
        assert result.band == "l"

    def test_spectrum_immutable(self) -> None:
        """Test spectrum result is frozen."""
        result = SpectrumResult.not_found()
        with pytest.raises(AttributeError):
            result.is_free = True  # type: ignore[misc]


# =============================================================================
# GroomingResult Tests
# =============================================================================


class TestGroomingResult:
    """Test GroomingResult dataclass."""

    def test_no_grooming(self) -> None:
        """Test no grooming result."""
        result = GroomingResult.no_grooming(100)
        assert not result.was_groomed
        assert result.needs_new_lightpath
        assert result.remaining_bandwidth_gbps == 100
        assert result.bandwidth_groomed_gbps == 0
        assert result.lightpaths_used == ()

    def test_full_grooming(self) -> None:
        """Test fully groomed result."""
        result = GroomingResult.full(100, [1, 2])
        assert result.fully_groomed
        assert not result.partially_groomed
        assert result.was_groomed
        assert not result.needs_new_lightpath
        assert result.lightpaths_used == (1, 2)
        assert result.bandwidth_groomed_gbps == 100
        assert result.remaining_bandwidth_gbps == 0

    def test_partial_grooming(self) -> None:
        """Test partially groomed result."""
        result = GroomingResult.partial(
            bandwidth_groomed=60,
            remaining=40,
            lightpath_ids=[1],
            forced_path=["0", "2", "5"],
        )
        assert result.partially_groomed
        assert not result.fully_groomed
        assert result.was_groomed
        assert result.needs_new_lightpath
        assert result.forced_path == ("0", "2", "5")
        assert result.bandwidth_groomed_gbps == 60
        assert result.remaining_bandwidth_gbps == 40

    def test_partial_grooming_no_forced_path(self) -> None:
        """Test partially groomed result without forced path."""
        result = GroomingResult.partial(
            bandwidth_groomed=60,
            remaining=40,
            lightpath_ids=[1],
        )
        assert result.forced_path is None

    def test_grooming_validation_mutually_exclusive(self) -> None:
        """Test fully and partially groomed are mutually exclusive."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            GroomingResult(
                fully_groomed=True,
                partially_groomed=True,
                lightpaths_used=(1,),
            )

    def test_grooming_validation_full_requires_zero_remaining(self) -> None:
        """Test fully groomed requires zero remaining."""
        with pytest.raises(ValueError, match="remaining_bandwidth_gbps == 0"):
            GroomingResult(
                fully_groomed=True,
                remaining_bandwidth_gbps=10,
                lightpaths_used=(1,),
            )

    def test_grooming_validation_requires_lightpaths(self) -> None:
        """Test groomed results require lightpaths."""
        with pytest.raises(ValueError, match="must have lightpaths_used"):
            GroomingResult(
                fully_groomed=True,
                remaining_bandwidth_gbps=0,
                lightpaths_used=(),  # Empty
            )

    def test_grooming_immutable(self) -> None:
        """Test grooming result is frozen."""
        result = GroomingResult.no_grooming(100)
        with pytest.raises(AttributeError):
            result.fully_groomed = True  # type: ignore[misc]


# =============================================================================
# SlicingResult Tests
# =============================================================================


class TestSlicingResult:
    """Test SlicingResult dataclass."""

    def test_failed_slicing(self) -> None:
        """Test failed slicing result."""
        result = SlicingResult.failed()
        assert not result.success
        assert not result.is_sliced
        assert result.num_slices == 0

    def test_single_lightpath(self) -> None:
        """Test single lightpath (no actual slicing)."""
        result = SlicingResult.single_lightpath(100, 42)
        assert result.success
        assert not result.is_sliced  # Only 1 slice doesn't count as sliced
        assert result.num_slices == 1
        assert result.lightpaths_created == (42,)
        assert result.total_bandwidth_gbps == 100
        assert result.slice_bandwidth_gbps == 100

    def test_sliced(self) -> None:
        """Test actual slicing."""
        result = SlicingResult.sliced(
            num_slices=4,
            slice_bandwidth=25,
            lightpath_ids=[1, 2, 3, 4],
        )
        assert result.success
        assert result.is_sliced
        assert result.num_slices == 4
        assert result.total_bandwidth_gbps == 100
        assert result.slice_bandwidth_gbps == 25
        assert result.lightpaths_created == (1, 2, 3, 4)

    def test_slicing_validation_success_requires_slices(self) -> None:
        """Test success requires num_slices > 0."""
        with pytest.raises(ValueError, match="num_slices > 0"):
            SlicingResult(success=True, num_slices=0)

    def test_slicing_validation_lightpath_count(self) -> None:
        """Test lightpaths count must match num_slices."""
        with pytest.raises(ValueError, match="must match num_slices"):
            SlicingResult(
                success=True,
                num_slices=4,
                lightpaths_created=(1, 2),  # Wrong count
            )

    def test_slicing_immutable(self) -> None:
        """Test slicing result is frozen."""
        result = SlicingResult.failed()
        with pytest.raises(AttributeError):
            result.success = True  # type: ignore[misc]


# =============================================================================
# SNRResult Tests
# =============================================================================


class TestSNRResult:
    """Test SNRResult dataclass."""

    def test_snr_success(self) -> None:
        """Test successful SNR validation."""
        result = SNRResult.success(snr_db=18.5, required_snr_db=15.0)
        assert result.passed
        assert result.snr_db == 18.5
        assert result.required_snr_db == 15.0
        assert result.margin_db == 3.5
        assert not result.is_degraded
        assert result.failure_reason is None

    def test_snr_failure(self) -> None:
        """Test failed SNR validation."""
        result = SNRResult.failure(
            snr_db=12.0,
            required_snr_db=15.0,
            reason="SNR too low",
        )
        assert not result.passed
        assert result.snr_db == 12.0
        assert result.margin_db == -3.0
        assert result.failure_reason == "SNR too low"

    def test_snr_skipped(self) -> None:
        """Test SNR validation skipped."""
        result = SNRResult.skipped()
        assert result.passed
        assert result.snr_db == 0.0
        assert result.required_snr_db == 0.0
        assert result.margin_db == 0.0

    def test_snr_is_degraded(self) -> None:
        """Test marginal SNR is flagged as degraded."""
        result = SNRResult.success(snr_db=15.5, required_snr_db=15.0)
        assert result.passed
        assert result.margin_db == 0.5
        assert result.is_degraded  # margin < 1.0

    def test_snr_not_degraded_when_failed(self) -> None:
        """Test failed SNR is not marked as degraded."""
        result = SNRResult.failure(snr_db=14.5, required_snr_db=15.0)
        assert not result.passed
        assert not result.is_degraded

    def test_snr_not_degraded_when_good_margin(self) -> None:
        """Test SNR with good margin is not degraded."""
        result = SNRResult.success(snr_db=20.0, required_snr_db=15.0)
        assert result.passed
        assert result.margin_db == 5.0
        assert not result.is_degraded

    def test_snr_with_link_breakdown(self) -> None:
        """Test SNR with per-link values."""
        link_values = {("0", "2"): 18.5, ("2", "5"): 15.0}
        result = SNRResult.success(
            snr_db=15.0,
            required_snr_db=15.0,
            link_snr_values=link_values,
        )
        assert result.has_link_breakdown
        weakest = result.get_weakest_link()
        assert weakest == (("2", "5"), 15.0)

    def test_snr_no_link_breakdown(self) -> None:
        """Test SNR without link breakdown."""
        result = SNRResult.success(snr_db=18.0, required_snr_db=15.0)
        assert not result.has_link_breakdown
        assert result.get_weakest_link() is None

    def test_snr_immutable(self) -> None:
        """Test SNR result is frozen."""
        result = SNRResult.skipped()
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]


# =============================================================================
# SNRRecheckResult Tests
# =============================================================================


class TestSNRRecheckResult:
    """Test SNRRecheckResult dataclass."""

    def test_snr_recheck_success(self) -> None:
        """Test successful SNR recheck (all lightpaths pass)."""
        result = SNRRecheckResult.success(checked_count=5)
        assert result.all_pass
        assert result.num_degraded == 0
        assert not result.has_violations
        assert result.degraded_lightpath_ids == ()
        assert result.checked_count == 5

    def test_snr_recheck_success_no_count(self) -> None:
        """Test successful SNR recheck without explicit count."""
        result = SNRRecheckResult.success()
        assert result.all_pass
        assert result.checked_count == 0

    def test_snr_recheck_degraded(self) -> None:
        """Test SNR recheck with degraded lightpaths."""
        violations = {10: -2.5, 20: -0.5}
        result = SNRRecheckResult.degraded(
            degraded_ids=[10, 20],
            violations=violations,
            checked_count=5,
        )
        assert not result.all_pass
        assert result.num_degraded == 2
        assert result.has_violations
        assert result.degraded_lightpath_ids == (10, 20)
        assert result.violations == violations
        assert result.checked_count == 5

    def test_snr_recheck_degraded_default_count(self) -> None:
        """Test degraded result defaults count to degraded count."""
        result = SNRRecheckResult.degraded(
            degraded_ids=[10],
            violations={10: -1.0},
        )
        assert result.checked_count == 1

    def test_snr_recheck_get_worst_violation(self) -> None:
        """Test get_worst_violation method."""
        violations = {10: -2.5, 20: -0.5, 30: -3.0}
        result = SNRRecheckResult.degraded(
            degraded_ids=[10, 20, 30],
            violations=violations,
        )
        worst = result.get_worst_violation()
        assert worst == (30, -3.0)  # Most negative is worst

    def test_snr_recheck_get_worst_violation_empty(self) -> None:
        """Test get_worst_violation when no violations."""
        result = SNRRecheckResult.success()
        assert result.get_worst_violation() is None

    def test_snr_recheck_validation_all_pass_with_degraded(self) -> None:
        """Test validation catches all_pass with degraded lightpaths."""
        with pytest.raises(ValueError, match="all_pass=True requires no degraded"):
            SNRRecheckResult(
                all_pass=True,
                degraded_lightpath_ids=(10,),
            )

    def test_snr_recheck_validation_not_pass_without_degraded(self) -> None:
        """Test validation catches all_pass=False without degraded lightpaths."""
        with pytest.raises(ValueError, match="all_pass=False requires at least one"):
            SNRRecheckResult(
                all_pass=False,
                degraded_lightpath_ids=(),
            )

    def test_snr_recheck_immutable(self) -> None:
        """Test SNR recheck result is frozen."""
        result = SNRRecheckResult.success()
        with pytest.raises(AttributeError):
            result.all_pass = False  # type: ignore[misc]


# =============================================================================
# AllocationResult Tests
# =============================================================================


class TestAllocationResult:
    """Test AllocationResult dataclass."""

    def test_blocked_result(self) -> None:
        """Test blocked allocation result."""
        result = AllocationResult.blocked(BlockReason.NO_PATH)
        assert not result.success
        assert result.block_reason == BlockReason.NO_PATH
        assert result.num_lightpaths == 0

    def test_blocked_with_nested_results(self) -> None:
        """Test blocked result with nested results."""
        route = RouteResult.empty()
        result = AllocationResult.blocked(
            BlockReason.NO_PATH,
            route_result=route,
        )
        assert result.route_result is route

    def test_success_new_lightpath(self) -> None:
        """Test successful new lightpath allocation."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
        )
        assert result.success
        assert result.lightpaths_created == (42,)
        assert result.lightpaths_groomed == ()
        assert result.total_bandwidth_allocated_gbps == 100
        assert result.num_lightpaths == 1
        assert not result.is_groomed
        assert not result.is_sliced

    def test_success_new_lightpath_with_protection(self) -> None:
        """Test successful protected allocation."""
        result = AllocationResult.success_new_lightpath(
            lightpath_id=42,
            bandwidth_gbps=100,
            is_protected=True,
        )
        assert result.success
        assert result.is_protected

    def test_success_new_lightpath_with_nested_results(self) -> None:
        """Test successful allocation with nested results."""
        route = RouteResult(
            paths=(("0", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK",),),
        )
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

        result = AllocationResult.success_new_lightpath(
            lightpath_id=1,
            bandwidth_gbps=100,
            route_result=route,
            spectrum_result=spectrum,
            snr_result=snr,
        )
        assert result.route_result is route
        assert result.spectrum_result is spectrum
        assert result.snr_result is snr

    def test_success_groomed(self) -> None:
        """Test successful groomed allocation."""
        result = AllocationResult.success_groomed(
            lightpath_ids=[10, 20],
            bandwidth_gbps=100,
        )
        assert result.success
        assert result.is_groomed
        assert result.lightpaths_groomed == (10, 20)
        assert result.lightpaths_created == ()
        assert result.num_lightpaths == 2
        assert result.used_grooming

    def test_success_groomed_with_result(self) -> None:
        """Test groomed allocation with grooming result."""
        grooming = GroomingResult.full(100, [10, 20])
        result = AllocationResult.success_groomed(
            lightpath_ids=[10, 20],
            bandwidth_gbps=100,
            grooming_result=grooming,
        )
        assert result.grooming_result is grooming

    def test_success_partial_groom(self) -> None:
        """Test successful partial grooming."""
        result = AllocationResult.success_partial_groom(
            groomed_ids=[10],
            new_lightpath_id=42,
            total_bandwidth=100,
        )
        assert result.success
        assert result.is_partially_groomed
        assert result.lightpaths_groomed == (10,)
        assert result.lightpaths_created == (42,)
        assert result.num_lightpaths == 2
        assert result.used_grooming

    def test_success_sliced(self) -> None:
        """Test successful sliced allocation."""
        result = AllocationResult.success_sliced(
            lightpath_ids=[1, 2, 3, 4],
            bandwidth_gbps=400,
        )
        assert result.success
        assert result.is_sliced
        assert result.num_lightpaths == 4
        assert result.lightpaths_created == (1, 2, 3, 4)
        assert result.total_bandwidth_allocated_gbps == 400

    def test_all_lightpath_ids(self) -> None:
        """Test all_lightpath_ids property."""
        result = AllocationResult.success_partial_groom(
            groomed_ids=[10, 20],
            new_lightpath_id=42,
            total_bandwidth=100,
        )
        # Created first, then groomed
        assert result.all_lightpath_ids == (42, 10, 20)

    def test_used_grooming(self) -> None:
        """Test used_grooming property."""
        full = AllocationResult.success_groomed([1], 100)
        assert full.used_grooming

        partial = AllocationResult.success_partial_groom([1], 2, 100)
        assert partial.used_grooming

        new_lp = AllocationResult.success_new_lightpath(1, 100)
        assert not new_lp.used_grooming

    def test_validation_success_requires_lightpath(self) -> None:
        """Test success requires at least one lightpath."""
        with pytest.raises(ValueError, match="at least one lightpath"):
            AllocationResult(success=True, total_bandwidth_allocated_gbps=100)

    def test_validation_success_requires_bandwidth(self) -> None:
        """Test success requires positive bandwidth."""
        with pytest.raises(ValueError, match="total_bandwidth > 0"):
            AllocationResult(
                success=True,
                lightpaths_created=(1,),
                total_bandwidth_allocated_gbps=0,
            )

    def test_validation_failure_requires_reason(self) -> None:
        """Test failure requires block_reason."""
        with pytest.raises(ValueError, match="requires block_reason"):
            AllocationResult(success=False)

    def test_allocation_immutable(self) -> None:
        """Test allocation result is frozen."""
        result = AllocationResult.blocked(BlockReason.CONGESTION)
        with pytest.raises(AttributeError):
            result.success = True  # type: ignore[misc]

    def test_per_segment_tracking_fields(self) -> None:
        """Test per-segment tracking fields for sliced allocations."""
        result = AllocationResult(
            success=True,
            lightpaths_created=(1, 2, 3, 4),
            total_bandwidth_allocated_gbps=400,
            is_sliced=True,
            bandwidth_allocations=(100, 100, 100, 100),
            modulations=("QPSK", "QPSK", "16-QAM", "16-QAM"),
            cores=(0, 1, 0, 1),
            bands=("c", "c", "c", "c"),
            start_slots=(0, 0, 100, 100),
            end_slots=(8, 8, 6, 6),
            xt_costs=(0.1, 0.2, 0.1, 0.15),
            xt_values=(0.05, 0.08, 0.06, 0.07),
            snr_values=(18.0, 17.5, 19.0, 18.5),
            lightpath_bandwidths=(100, 100, 100, 100),
        )
        assert result.success
        assert len(result.bandwidth_allocations) == 4
        assert result.modulations == ("QPSK", "QPSK", "16-QAM", "16-QAM")
        assert result.cores == (0, 1, 0, 1)
        assert result.bands == ("c", "c", "c", "c")
        assert result.start_slots == (0, 0, 100, 100)
        assert result.end_slots == (8, 8, 6, 6)
        assert len(result.xt_costs) == 4
        assert len(result.xt_values) == 4
        assert len(result.snr_values) == 4

    def test_per_segment_defaults_empty(self) -> None:
        """Test per-segment fields default to empty tuples."""
        result = AllocationResult.success_new_lightpath(1, 100)
        assert result.bandwidth_allocations == ()
        assert result.modulations == ()
        assert result.cores == ()
        assert result.bands == ()
        assert result.start_slots == ()
        assert result.end_slots == ()
        assert result.xt_costs == ()
        assert result.xt_values == ()
        assert result.snr_values == ()
        assert result.lightpath_bandwidths == ()

    def test_with_protection_result(self) -> None:
        """Test AllocationResult with nested ProtectionResult."""
        spectrum = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        protection = ProtectionResult.established(
            primary_spectrum=spectrum,
            backup_spectrum=spectrum,
        )
        result = AllocationResult(
            success=True,
            lightpaths_created=(1,),
            total_bandwidth_allocated_gbps=100,
            is_protected=True,
            protection_result=protection,
        )
        assert result.protection_result is protection
        assert result.protection_result.is_fully_protected


# =============================================================================
# ProtectionResult Tests
# =============================================================================


class TestProtectionResult:
    """Test ProtectionResult dataclass."""

    def test_established_full_protection(self) -> None:
        """Test fully established protection."""
        primary = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        backup = SpectrumResult(
            is_free=True,
            start_slot=100,
            end_slot=108,
            core=1,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        result = ProtectionResult.established(primary, backup)
        assert result.primary_established
        assert result.backup_established
        assert result.is_fully_protected
        assert result.primary_spectrum is primary
        assert result.backup_spectrum is backup

    def test_established_primary_only(self) -> None:
        """Test protection with only primary established."""
        primary = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        result = ProtectionResult.established(primary)
        assert result.primary_established
        assert not result.backup_established
        assert not result.is_fully_protected
        assert result.primary_spectrum is primary
        assert result.backup_spectrum is None

    def test_primary_only_factory(self) -> None:
        """Test primary_only factory method."""
        primary = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        result = ProtectionResult.primary_only(primary)
        assert result.primary_established
        assert not result.backup_established
        assert not result.is_fully_protected

    def test_failed_protection(self) -> None:
        """Test failed protection establishment."""
        result = ProtectionResult.failed()
        assert not result.primary_established
        assert not result.backup_established
        assert not result.is_fully_protected
        assert result.primary_spectrum is None
        assert result.backup_spectrum is None

    def test_switchover_success(self) -> None:
        """Test successful switchover event."""
        result = ProtectionResult.switchover(
            success=True,
            switchover_time_ms=50.0,
            failure_type="link",
            recovery_type="protection",
        )
        assert result.switchover_triggered
        assert result.switchover_success
        assert result.switchover_time_ms == 50.0
        assert result.failure_type == "link"
        assert result.recovery_type == "protection"

    def test_switchover_failure(self) -> None:
        """Test failed switchover event."""
        result = ProtectionResult.switchover(
            success=False,
            switchover_time_ms=100.0,
            failure_type="node",
        )
        assert result.switchover_triggered
        assert not result.switchover_success
        assert result.switchover_time_ms is None  # Not set when failed
        assert result.failure_type == "node"

    def test_recovery_duration(self) -> None:
        """Test recovery duration calculation."""
        result = ProtectionResult(
            primary_established=True,
            backup_established=True,
            recovery_start=10.0,
            recovery_end=10.05,  # 50ms in simulation time
            recovery_type="protection",
        )
        assert result.recovery_duration_ms == pytest.approx(50.0)

    def test_recovery_duration_incomplete(self) -> None:
        """Test recovery duration when incomplete."""
        result = ProtectionResult(
            primary_established=True,
            recovery_start=10.0,
            recovery_end=None,
        )
        assert result.recovery_duration_ms is None

    def test_protection_immutable(self) -> None:
        """Test protection result is frozen."""
        result = ProtectionResult.failed()
        with pytest.raises(AttributeError):
            result.primary_established = True  # type: ignore[misc]

    def test_default_values(self) -> None:
        """Test default field values."""
        result = ProtectionResult()
        assert not result.primary_established
        assert not result.backup_established
        assert result.primary_spectrum is None
        assert result.backup_spectrum is None
        assert not result.switchover_triggered
        assert not result.switchover_success
        assert result.switchover_time_ms is None
        assert result.failure_type is None
        assert result.recovery_start is None
        assert result.recovery_end is None
        assert result.recovery_type is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestResultIntegration:
    """Integration tests for result objects."""

    def test_result_chain_success(self) -> None:
        """Test successful result chain."""
        # Routing succeeded
        route = RouteResult(
            paths=(("0", "2", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK",),),
        )
        assert not route.is_empty

        # Spectrum found
        spectrum = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        assert spectrum.is_free

        # SNR passed
        snr = SNRResult.success(18.0, 15.0)
        assert snr.passed

        # Final result
        result = AllocationResult.success_new_lightpath(
            lightpath_id=1,
            bandwidth_gbps=100,
            route_result=route,
            spectrum_result=spectrum,
            snr_result=snr,
        )
        assert result.success

    def test_result_chain_routing_failure(self) -> None:
        """Test failure at routing stage."""
        route = RouteResult.empty()
        assert route.is_empty

        result = AllocationResult.blocked(
            BlockReason.NO_PATH,
            route_result=route,
        )
        assert not result.success
        assert result.block_reason == BlockReason.NO_PATH

    def test_result_chain_spectrum_failure(self) -> None:
        """Test failure at spectrum stage."""
        route = RouteResult(
            paths=(("0", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK",),),
        )
        spectrum = SpectrumResult.not_found(slots_needed=8)

        result = AllocationResult.blocked(
            BlockReason.CONGESTION,
            route_result=route,
            spectrum_result=spectrum,
        )
        assert not result.success
        assert result.block_reason == BlockReason.CONGESTION

    def test_result_chain_snr_failure(self) -> None:
        """Test failure at SNR validation stage."""
        route = RouteResult(
            paths=(("0", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK",),),
        )
        spectrum = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        snr = SNRResult.failure(12.0, 15.0, "SNR too low for QPSK")

        result = AllocationResult.blocked(
            BlockReason.SNR_THRESHOLD,
            route_result=route,
            spectrum_result=spectrum,
            snr_result=snr,
        )
        assert not result.success
        assert result.block_reason == BlockReason.SNR_THRESHOLD

    def test_grooming_flow(self) -> None:
        """Test grooming result flow."""
        # Full grooming
        grooming = GroomingResult.full(100, [1, 2])
        assert grooming.fully_groomed
        assert not grooming.needs_new_lightpath

        result = AllocationResult.success_groomed(
            lightpath_ids=[1, 2],
            bandwidth_gbps=100,
            grooming_result=grooming,
        )
        assert result.success
        assert result.is_groomed

    def test_slicing_flow(self) -> None:
        """Test slicing result flow."""
        # Sliced request
        slicing = SlicingResult.sliced(
            num_slices=4,
            slice_bandwidth=25,
            lightpath_ids=[1, 2, 3, 4],
        )
        assert slicing.is_sliced

        route = RouteResult(
            paths=(("0", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK",),),
        )

        result = AllocationResult.success_sliced(
            lightpath_ids=[1, 2, 3, 4],
            bandwidth_gbps=100,
            slicing_result=slicing,
            route_result=route,
        )
        assert result.success
        assert result.is_sliced

    def test_all_block_reasons_can_be_used(self) -> None:
        """Test that all block reasons work with AllocationResult."""
        for reason in BlockReason:
            result = AllocationResult.blocked(reason)
            assert not result.success
            assert result.block_reason == reason

    def test_protection_flow(self) -> None:
        """Test protection result flow with AllocationResult."""
        # Create spectrum results for primary and backup
        primary_spectrum = SpectrumResult(
            is_free=True,
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )
        backup_spectrum = SpectrumResult(
            is_free=True,
            start_slot=100,
            end_slot=108,
            core=1,
            band="c",
            modulation="QPSK",
            slots_needed=8,
        )

        # Protection established
        protection = ProtectionResult.established(primary_spectrum, backup_spectrum)
        assert protection.is_fully_protected

        # Create allocation result with protection
        result = AllocationResult(
            success=True,
            lightpaths_created=(1, 2),  # Primary and backup lightpaths
            total_bandwidth_allocated_gbps=100,
            is_protected=True,
            protection_result=protection,
        )
        assert result.success
        assert result.is_protected
        assert result.protection_result is not None
        assert result.protection_result.is_fully_protected

    def test_sliced_allocation_with_per_segment_data(self) -> None:
        """Test sliced allocation with complete per-segment tracking."""
        route = RouteResult(
            paths=(("0", "5"),),
            weights_km=(100.0,),
            modulations=(("QPSK", "16-QAM"),),
        )

        slicing = SlicingResult.sliced(
            num_slices=4,
            slice_bandwidth=25,
            lightpath_ids=[1, 2, 3, 4],
        )

        result = AllocationResult(
            success=True,
            lightpaths_created=(1, 2, 3, 4),
            total_bandwidth_allocated_gbps=100,
            is_sliced=True,
            bandwidth_allocations=(25, 25, 25, 25),
            modulations=("QPSK", "QPSK", "QPSK", "QPSK"),
            cores=(0, 1, 2, 3),
            bands=("c", "c", "c", "c"),
            start_slots=(0, 0, 0, 0),
            end_slots=(4, 4, 4, 4),
            snr_values=(18.0, 17.5, 18.2, 17.8),
            xt_costs=(0.1, 0.12, 0.11, 0.13),
            lightpath_bandwidths=(25, 25, 25, 25),
            slicing_result=slicing,
            route_result=route,
        )

        assert result.success
        assert result.is_sliced
        assert len(result.bandwidth_allocations) == 4
        assert sum(result.bandwidth_allocations) == 100
        assert all(b == "c" for b in result.bands)
        assert result.slicing_result is not None
        assert result.slicing_result.num_slices == 4
