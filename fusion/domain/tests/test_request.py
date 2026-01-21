"""Tests for Request, RequestType, RequestStatus, ProtectionStatus, and BlockReason."""

from __future__ import annotations

import pytest

from fusion.domain.request import (
    BlockReason,
    ProtectionStatus,
    Request,
    RequestStatus,
    RequestType,
)


class TestRequestType:
    """Test RequestType enum."""

    def test_all_types_exist(self) -> None:
        """Verify all expected types are defined."""
        assert RequestType.ARRIVAL
        assert RequestType.RELEASE

    def test_values(self) -> None:
        """Test enum values."""
        assert RequestType.ARRIVAL.value == "arrival"
        assert RequestType.RELEASE.value == "release"

    def test_from_legacy(self) -> None:
        """Test conversion from legacy strings."""
        assert RequestType.from_legacy("arrival") == RequestType.ARRIVAL
        assert RequestType.from_legacy("release") == RequestType.RELEASE
        assert RequestType.from_legacy("ARRIVAL") == RequestType.ARRIVAL
        assert RequestType.from_legacy("RELEASE") == RequestType.RELEASE

    def test_to_legacy(self) -> None:
        """Test conversion to legacy strings."""
        assert RequestType.ARRIVAL.to_legacy() == "arrival"
        assert RequestType.RELEASE.to_legacy() == "release"


class TestRequestStatus:
    """Test RequestStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Verify all expected statuses are defined."""
        # Initial state
        assert RequestStatus.PENDING
        # Processing states
        assert RequestStatus.ROUTING
        assert RequestStatus.SPECTRUM_SEARCH
        assert RequestStatus.SNR_CHECK
        # Success states
        assert RequestStatus.ALLOCATED
        assert RequestStatus.GROOMED
        assert RequestStatus.PARTIALLY_GROOMED
        # Failure state
        assert RequestStatus.BLOCKED
        # Release states
        assert RequestStatus.RELEASING
        assert RequestStatus.RELEASED

    def test_is_terminal(self) -> None:
        """Test terminal state detection."""
        # Non-terminal states
        assert not RequestStatus.PENDING.is_terminal()
        assert not RequestStatus.ROUTING.is_terminal()
        assert not RequestStatus.SPECTRUM_SEARCH.is_terminal()
        assert not RequestStatus.SNR_CHECK.is_terminal()
        assert not RequestStatus.RELEASING.is_terminal()
        # Terminal states
        assert RequestStatus.ALLOCATED.is_terminal()
        assert RequestStatus.GROOMED.is_terminal()
        assert RequestStatus.PARTIALLY_GROOMED.is_terminal()
        assert RequestStatus.BLOCKED.is_terminal()
        assert RequestStatus.RELEASED.is_terminal()

    def test_is_success(self) -> None:
        """Test success state detection."""
        assert not RequestStatus.PENDING.is_success()
        assert not RequestStatus.ROUTING.is_success()
        assert not RequestStatus.BLOCKED.is_success()
        assert not RequestStatus.RELEASED.is_success()
        # Success states
        assert RequestStatus.ALLOCATED.is_success()
        assert RequestStatus.GROOMED.is_success()
        assert RequestStatus.PARTIALLY_GROOMED.is_success()

    def test_is_processing(self) -> None:
        """Test processing state detection."""
        assert not RequestStatus.PENDING.is_processing()
        assert not RequestStatus.ALLOCATED.is_processing()
        assert not RequestStatus.BLOCKED.is_processing()
        assert not RequestStatus.RELEASED.is_processing()
        # Processing states
        assert RequestStatus.ROUTING.is_processing()
        assert RequestStatus.SPECTRUM_SEARCH.is_processing()
        assert RequestStatus.SNR_CHECK.is_processing()
        assert RequestStatus.RELEASING.is_processing()

    def test_valid_transitions_from_pending(self) -> None:
        """Test valid transitions from PENDING."""
        assert RequestStatus.PENDING.can_transition_to(RequestStatus.ROUTING)
        assert RequestStatus.PENDING.can_transition_to(RequestStatus.BLOCKED)
        # Direct to success states (simplified flows)
        assert RequestStatus.PENDING.can_transition_to(RequestStatus.ALLOCATED)
        assert RequestStatus.PENDING.can_transition_to(RequestStatus.GROOMED)
        assert RequestStatus.PENDING.can_transition_to(RequestStatus.PARTIALLY_GROOMED)
        # Invalid
        assert not RequestStatus.PENDING.can_transition_to(RequestStatus.RELEASED)
        assert not RequestStatus.PENDING.can_transition_to(RequestStatus.PENDING)

    def test_valid_transitions_from_routing(self) -> None:
        """Test valid transitions from ROUTING."""
        assert RequestStatus.ROUTING.can_transition_to(RequestStatus.SPECTRUM_SEARCH)
        assert RequestStatus.ROUTING.can_transition_to(RequestStatus.BLOCKED)
        assert not RequestStatus.ROUTING.can_transition_to(RequestStatus.PENDING)

    def test_valid_transitions_from_spectrum_search(self) -> None:
        """Test valid transitions from SPECTRUM_SEARCH."""
        assert RequestStatus.SPECTRUM_SEARCH.can_transition_to(RequestStatus.SNR_CHECK)
        assert RequestStatus.SPECTRUM_SEARCH.can_transition_to(RequestStatus.ALLOCATED)
        assert RequestStatus.SPECTRUM_SEARCH.can_transition_to(RequestStatus.GROOMED)
        assert RequestStatus.SPECTRUM_SEARCH.can_transition_to(RequestStatus.BLOCKED)

    def test_valid_transitions_from_snr_check(self) -> None:
        """Test valid transitions from SNR_CHECK."""
        assert RequestStatus.SNR_CHECK.can_transition_to(RequestStatus.ALLOCATED)
        assert RequestStatus.SNR_CHECK.can_transition_to(RequestStatus.GROOMED)
        assert RequestStatus.SNR_CHECK.can_transition_to(RequestStatus.PARTIALLY_GROOMED)
        assert RequestStatus.SNR_CHECK.can_transition_to(RequestStatus.BLOCKED)

    def test_valid_transitions_from_success_states(self) -> None:
        """Test valid transitions from success states."""
        assert RequestStatus.ALLOCATED.can_transition_to(RequestStatus.RELEASING)
        assert RequestStatus.GROOMED.can_transition_to(RequestStatus.RELEASING)
        assert RequestStatus.PARTIALLY_GROOMED.can_transition_to(RequestStatus.RELEASING)
        # Cannot go back
        assert not RequestStatus.ALLOCATED.can_transition_to(RequestStatus.PENDING)
        assert not RequestStatus.ALLOCATED.can_transition_to(RequestStatus.BLOCKED)

    def test_valid_transitions_from_releasing(self) -> None:
        """Test valid transitions from RELEASING."""
        assert RequestStatus.RELEASING.can_transition_to(RequestStatus.RELEASED)
        assert not RequestStatus.RELEASING.can_transition_to(RequestStatus.PENDING)

    def test_no_transitions_from_terminal(self) -> None:
        """Test no transitions from terminal states."""
        for target in RequestStatus:
            assert not RequestStatus.BLOCKED.can_transition_to(target)
            assert not RequestStatus.RELEASED.can_transition_to(target)


class TestBlockReason:
    """Test BlockReason enum."""

    def test_all_reasons_exist(self) -> None:
        """Verify all expected block reasons are defined."""
        # Path-related
        assert BlockReason.NO_PATH
        assert BlockReason.DISTANCE
        # Spectrum-related
        assert BlockReason.CONGESTION
        # Quality-related
        assert BlockReason.SNR_THRESHOLD
        assert BlockReason.XT_THRESHOLD
        # Feature-specific
        assert BlockReason.GROOMING_FAIL
        assert BlockReason.SLICING_FAIL
        assert BlockReason.PROTECTION_FAIL
        # Failure-related
        assert BlockReason.LINK_FAILURE
        assert BlockReason.NODE_FAILURE
        assert BlockReason.FAILURE
        # Resource limits
        assert BlockReason.TRANSPONDER_LIMIT
        assert BlockReason.MAX_SEGMENTS

    def test_from_legacy_string_standard(self) -> None:
        """Test conversion from standard legacy strings."""
        assert BlockReason.from_legacy_string("no_path") == BlockReason.NO_PATH
        assert BlockReason.from_legacy_string("congestion") == BlockReason.CONGESTION
        assert BlockReason.from_legacy_string("distance") == BlockReason.DISTANCE

    def test_from_legacy_string_mapped(self) -> None:
        """Test conversion from mapped legacy strings."""
        assert BlockReason.from_legacy_string("no_route") == BlockReason.NO_PATH
        assert BlockReason.from_legacy_string("xt_threshold") == BlockReason.XT_THRESHOLD
        assert BlockReason.from_legacy_string("failure") == BlockReason.FAILURE
        assert BlockReason.from_legacy_string("snr_fail") == BlockReason.SNR_THRESHOLD
        assert BlockReason.from_legacy_string("snr_failure") == BlockReason.SNR_THRESHOLD
        assert BlockReason.from_legacy_string("grooming_fail") == BlockReason.GROOMING_FAIL
        assert BlockReason.from_legacy_string("slicing_fail") == BlockReason.SLICING_FAIL
        assert BlockReason.from_legacy_string("protection_fail") == BlockReason.PROTECTION_FAIL
        assert BlockReason.from_legacy_string("link_failure") == BlockReason.LINK_FAILURE
        assert BlockReason.from_legacy_string("node_failure") == BlockReason.NODE_FAILURE
        assert BlockReason.from_legacy_string("transponder_limit") == BlockReason.TRANSPONDER_LIMIT
        assert BlockReason.from_legacy_string("max_segments") == BlockReason.MAX_SEGMENTS

    def test_from_legacy_string_none(self) -> None:
        """Test conversion from None or empty string."""
        assert BlockReason.from_legacy_string(None) is None
        assert BlockReason.from_legacy_string("") is None

    def test_from_legacy_string_unknown(self) -> None:
        """Test unknown legacy string defaults to FAILURE."""
        assert BlockReason.from_legacy_string("unknown_reason") == BlockReason.FAILURE

    def test_to_legacy_string(self) -> None:
        """Test conversion to legacy string."""
        assert BlockReason.NO_PATH.to_legacy_string() == "no_path"
        assert BlockReason.DISTANCE.to_legacy_string() == "distance"
        assert BlockReason.CONGESTION.to_legacy_string() == "congestion"
        assert BlockReason.SNR_THRESHOLD.to_legacy_string() == "snr_fail"
        assert BlockReason.XT_THRESHOLD.to_legacy_string() == "xt_threshold"
        assert BlockReason.FAILURE.to_legacy_string() == "failure"

    def test_is_path_related(self) -> None:
        """Test is_path_related helper."""
        assert BlockReason.NO_PATH.is_path_related()
        assert BlockReason.DISTANCE.is_path_related()
        assert not BlockReason.CONGESTION.is_path_related()

    def test_is_spectrum_related(self) -> None:
        """Test is_spectrum_related helper."""
        assert BlockReason.CONGESTION.is_spectrum_related()
        assert not BlockReason.NO_PATH.is_spectrum_related()

    def test_is_quality_related(self) -> None:
        """Test is_quality_related helper."""
        assert BlockReason.SNR_THRESHOLD.is_quality_related()
        assert BlockReason.XT_THRESHOLD.is_quality_related()
        assert not BlockReason.CONGESTION.is_quality_related()

    def test_is_feature_related(self) -> None:
        """Test is_feature_related helper."""
        assert BlockReason.GROOMING_FAIL.is_feature_related()
        assert BlockReason.SLICING_FAIL.is_feature_related()
        assert BlockReason.PROTECTION_FAIL.is_feature_related()
        assert not BlockReason.CONGESTION.is_feature_related()

    def test_is_failure_related(self) -> None:
        """Test is_failure_related helper."""
        assert BlockReason.LINK_FAILURE.is_failure_related()
        assert BlockReason.NODE_FAILURE.is_failure_related()
        assert BlockReason.FAILURE.is_failure_related()
        assert not BlockReason.CONGESTION.is_failure_related()

    def test_is_resource_limit(self) -> None:
        """Test is_resource_limit helper."""
        assert BlockReason.TRANSPONDER_LIMIT.is_resource_limit()
        assert BlockReason.MAX_SEGMENTS.is_resource_limit()
        assert not BlockReason.CONGESTION.is_resource_limit()


class TestProtectionStatus:
    """Test ProtectionStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Verify all expected statuses are defined."""
        assert ProtectionStatus.UNPROTECTED
        assert ProtectionStatus.ESTABLISHING
        assert ProtectionStatus.ACTIVE_PRIMARY
        assert ProtectionStatus.ACTIVE_BACKUP
        assert ProtectionStatus.SWITCHOVER_IN_PROGRESS
        assert ProtectionStatus.PRIMARY_FAILED
        assert ProtectionStatus.BACKUP_FAILED
        assert ProtectionStatus.BOTH_FAILED

    def test_is_active(self) -> None:
        """Test is_active helper."""
        assert ProtectionStatus.ACTIVE_PRIMARY.is_active()
        assert ProtectionStatus.ACTIVE_BACKUP.is_active()
        assert ProtectionStatus.PRIMARY_FAILED.is_active()
        assert ProtectionStatus.BACKUP_FAILED.is_active()
        assert not ProtectionStatus.UNPROTECTED.is_active()
        assert not ProtectionStatus.BOTH_FAILED.is_active()

    def test_is_degraded(self) -> None:
        """Test is_degraded helper."""
        assert ProtectionStatus.PRIMARY_FAILED.is_degraded()
        assert ProtectionStatus.BACKUP_FAILED.is_degraded()
        assert not ProtectionStatus.ACTIVE_PRIMARY.is_degraded()
        assert not ProtectionStatus.BOTH_FAILED.is_degraded()

    def test_is_failed(self) -> None:
        """Test is_failed helper."""
        assert ProtectionStatus.BOTH_FAILED.is_failed()
        assert not ProtectionStatus.ACTIVE_PRIMARY.is_failed()
        assert not ProtectionStatus.PRIMARY_FAILED.is_failed()

    def test_is_protected(self) -> None:
        """Test is_protected helper."""
        assert not ProtectionStatus.UNPROTECTED.is_protected()
        assert ProtectionStatus.ACTIVE_PRIMARY.is_protected()
        assert ProtectionStatus.ACTIVE_BACKUP.is_protected()
        assert ProtectionStatus.BOTH_FAILED.is_protected()

    def test_from_legacy(self) -> None:
        """Test conversion from legacy fields."""
        assert ProtectionStatus.from_legacy(False, None) == ProtectionStatus.UNPROTECTED
        assert ProtectionStatus.from_legacy(True, "primary") == ProtectionStatus.ACTIVE_PRIMARY
        assert ProtectionStatus.from_legacy(True, "backup") == ProtectionStatus.ACTIVE_BACKUP
        assert ProtectionStatus.from_legacy(True, None) == ProtectionStatus.ACTIVE_PRIMARY

    def test_to_legacy_active_path(self) -> None:
        """Test conversion to legacy active_path string."""
        assert ProtectionStatus.UNPROTECTED.to_legacy_active_path() == "primary"
        assert ProtectionStatus.ACTIVE_PRIMARY.to_legacy_active_path() == "primary"
        assert ProtectionStatus.ACTIVE_BACKUP.to_legacy_active_path() == "backup"
        assert ProtectionStatus.PRIMARY_FAILED.to_legacy_active_path() == "backup"
        assert ProtectionStatus.BACKUP_FAILED.to_legacy_active_path() == "primary"


class TestRequestCreation:
    """Test Request instantiation."""

    def test_create_minimal_request(self) -> None:
        """Test creating request with required fields only."""
        request = Request(
            request_id=1,
            source="0",
            destination="5",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )

        assert request.request_id == 1
        assert request.source == "0"
        assert request.destination == "5"
        assert request.bandwidth_gbps == 100
        assert request.arrive_time == 0.0
        assert request.depart_time == 5.0
        assert request.status == RequestStatus.PENDING
        assert request.lightpath_ids == []
        assert request.block_reason is None
        assert request.modulation_formats == {}

    def test_create_full_request(self) -> None:
        """Test creating request with all fields."""
        request = Request(
            request_id=42,
            source="NYC",
            destination="LAX",
            bandwidth_gbps=400,
            arrive_time=10.5,
            depart_time=20.5,
            modulation_formats={"QPSK": {"slots": 4}},
            protection_status=ProtectionStatus.ACTIVE_PRIMARY,
        )

        assert request.modulation_formats == {"QPSK": {"slots": 4}}
        assert request.is_protected is True  # Computed from protection_status
        assert request.is_groomed is False

    def test_default_feature_flags(self) -> None:
        """Test default values for feature flags."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=50,
            arrive_time=0.0,
            depart_time=1.0,
        )

        assert request.is_groomed is False
        assert request.is_partially_groomed is False
        assert request.is_sliced is False
        assert request.is_protected is False


class TestRequestValidation:
    """Test Request validation in __post_init__."""

    def test_same_source_destination(self) -> None:
        """Test that same source and destination raises ValueError."""
        with pytest.raises(ValueError, match="source and destination cannot be the same"):
            Request(
                request_id=1,
                source="0",
                destination="0",
                bandwidth_gbps=100,
                arrive_time=0.0,
                depart_time=5.0,
            )

    def test_zero_bandwidth(self) -> None:
        """Test that zero bandwidth raises ValueError."""
        with pytest.raises(ValueError, match="bandwidth_gbps must be > 0"):
            Request(
                request_id=1,
                source="0",
                destination="1",
                bandwidth_gbps=0,
                arrive_time=0.0,
                depart_time=5.0,
            )

    def test_negative_bandwidth(self) -> None:
        """Test that negative bandwidth raises ValueError."""
        with pytest.raises(ValueError, match="bandwidth_gbps must be > 0"):
            Request(
                request_id=1,
                source="0",
                destination="1",
                bandwidth_gbps=-100,
                arrive_time=0.0,
                depart_time=5.0,
            )

    def test_depart_before_arrive(self) -> None:
        """Test that depart_time <= arrive_time raises ValueError."""
        with pytest.raises(ValueError, match="depart_time must be > arrive_time"):
            Request(
                request_id=1,
                source="0",
                destination="1",
                bandwidth_gbps=100,
                arrive_time=10.0,
                depart_time=5.0,
            )

    def test_depart_equals_arrive(self) -> None:
        """Test that depart_time == arrive_time raises ValueError."""
        with pytest.raises(ValueError, match="depart_time must be > arrive_time"):
            Request(
                request_id=1,
                source="0",
                destination="1",
                bandwidth_gbps=100,
                arrive_time=5.0,
                depart_time=5.0,
            )


class TestRequestComputedProperties:
    """Test Request computed properties."""

    @pytest.fixture
    def sample_request(self) -> Request:
        """Create a standard request for testing."""
        return Request(
            request_id=42,
            source="A",
            destination="B",
            bandwidth_gbps=100,
            arrive_time=10.0,
            depart_time=25.0,
        )

    def test_is_arrival(self, sample_request: Request) -> None:
        """Test is_arrival property."""
        assert sample_request.is_arrival is True
        sample_request.mark_routed([1])
        assert sample_request.is_arrival is False

    def test_is_successful(self, sample_request: Request) -> None:
        """Test is_successful property."""
        assert sample_request.is_successful is False
        sample_request.mark_routed([1])
        assert sample_request.is_successful is True

    def test_is_blocked(self, sample_request: Request) -> None:
        """Test is_blocked property."""
        assert sample_request.is_blocked is False
        sample_request.mark_blocked(BlockReason.NO_PATH)
        assert sample_request.is_blocked is True

    def test_is_released(self, sample_request: Request) -> None:
        """Test is_released property."""
        assert sample_request.is_released is False
        sample_request.mark_routed([1])
        sample_request.mark_released()
        assert sample_request.is_released is True

    def test_is_terminal(self, sample_request: Request) -> None:
        """Test is_terminal property."""
        assert sample_request.is_terminal is False
        sample_request.mark_blocked(BlockReason.CONGESTION)
        assert sample_request.is_terminal is True

    def test_is_terminal_after_release(self, sample_request: Request) -> None:
        """Test is_terminal property after release."""
        sample_request.mark_routed([1])
        sample_request.mark_released()
        assert sample_request.is_terminal is True

    def test_is_processing(self, sample_request: Request) -> None:
        """Test is_processing property."""
        assert sample_request.is_processing is False
        sample_request.set_status(RequestStatus.ROUTING)
        assert sample_request.is_processing is True

    def test_endpoint_key(self, sample_request: Request) -> None:
        """Test endpoint_key is sorted."""
        assert sample_request.endpoint_key == ("A", "B")

        # Verify reverse direction produces same key
        reverse = Request(
            request_id=43,
            source="B",
            destination="A",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=1.0,
        )
        assert reverse.endpoint_key == ("A", "B")

    def test_endpoint_key_numeric_strings(self) -> None:
        """Test endpoint_key with numeric string node IDs."""
        request = Request(
            request_id=1,
            source="10",
            destination="2",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=1.0,
        )
        # String sorting: "10" < "2"
        assert request.endpoint_key == ("10", "2")

    def test_holding_time(self, sample_request: Request) -> None:
        """Test holding_time calculation."""
        assert sample_request.holding_time == 15.0  # 25.0 - 10.0

    def test_num_lightpaths(self, sample_request: Request) -> None:
        """Test num_lightpaths property."""
        assert sample_request.num_lightpaths == 0
        sample_request.mark_routed([1, 2, 3])
        assert sample_request.num_lightpaths == 3


class TestRequestStateTransitions:
    """Test Request state transition methods."""

    @pytest.fixture
    def sample_request(self) -> Request:
        """Create a fresh request for each test."""
        return Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )

    def test_mark_allocated(self, sample_request: Request) -> None:
        """Test successful allocation transition."""
        sample_request.mark_allocated([10, 20])

        assert sample_request.status == RequestStatus.ALLOCATED
        assert sample_request.lightpath_ids == [10, 20]
        assert sample_request.block_reason is None

    def test_mark_routed(self, sample_request: Request) -> None:
        """Test mark_routed as alias for mark_allocated."""
        sample_request.mark_routed([10, 20])

        assert sample_request.status == RequestStatus.ALLOCATED
        assert sample_request.lightpath_ids == [10, 20]
        assert sample_request.block_reason is None

    def test_mark_routed_clears_block_reason(self, sample_request: Request) -> None:
        """Test that mark_routed clears any block_reason."""
        # Simulate some intermediate state with block_reason
        sample_request.block_reason = BlockReason.CONGESTION
        sample_request.mark_routed([1])
        assert sample_request.block_reason is None

    def test_mark_routed_copies_list(self, sample_request: Request) -> None:
        """Test that lightpath_ids is copied, not aliased."""
        ids = [1, 2, 3]
        sample_request.mark_routed(ids)
        ids.append(4)
        assert sample_request.lightpath_ids == [1, 2, 3]  # Not affected

    def test_mark_routed_empty_ids_raises(self, sample_request: Request) -> None:
        """Test that empty lightpath_ids raises ValueError."""
        with pytest.raises(ValueError, match="Must provide at least one lightpath_id"):
            sample_request.mark_routed([])

    def test_mark_routed_from_invalid_state(self, sample_request: Request) -> None:
        """Test invalid transition to ALLOCATED."""
        sample_request.mark_blocked(BlockReason.NO_PATH)
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_routed([1])

    def test_mark_routed_from_allocated_invalid(self, sample_request: Request) -> None:
        """Test cannot transition from ALLOCATED to ALLOCATED."""
        sample_request.mark_routed([1])
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_routed([2])

    def test_mark_groomed(self, sample_request: Request) -> None:
        """Test grooming transition."""
        sample_request.mark_groomed([5])

        assert sample_request.status == RequestStatus.GROOMED
        assert sample_request.lightpath_ids == [5]
        assert sample_request.is_groomed is True

    def test_mark_groomed_empty_ids_raises(self, sample_request: Request) -> None:
        """Test that empty lightpath_ids raises ValueError for grooming."""
        with pytest.raises(ValueError, match="Must provide at least one lightpath_id"):
            sample_request.mark_groomed([])

    def test_mark_partially_groomed(self, sample_request: Request) -> None:
        """Test partial grooming transition."""
        sample_request.mark_partially_groomed([5, 6])

        assert sample_request.status == RequestStatus.PARTIALLY_GROOMED
        assert sample_request.lightpath_ids == [5, 6]
        assert sample_request.is_partially_groomed is True

    def test_mark_blocked(self, sample_request: Request) -> None:
        """Test blocking transition."""
        sample_request.mark_blocked(BlockReason.SNR_THRESHOLD)

        assert sample_request.status == RequestStatus.BLOCKED
        assert sample_request.block_reason == BlockReason.SNR_THRESHOLD

    def test_mark_blocked_from_invalid_state(self, sample_request: Request) -> None:
        """Test invalid transition to BLOCKED."""
        sample_request.mark_routed([1])
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_blocked(BlockReason.CONGESTION)

    def test_mark_blocked_from_released_invalid(self, sample_request: Request) -> None:
        """Test cannot transition from RELEASED to BLOCKED."""
        sample_request.mark_routed([1])
        sample_request.mark_released()
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_blocked(BlockReason.CONGESTION)

    def test_mark_releasing(self, sample_request: Request) -> None:
        """Test releasing transition."""
        sample_request.mark_allocated([1])
        sample_request.mark_releasing()

        assert sample_request.status == RequestStatus.RELEASING

    def test_mark_released(self, sample_request: Request) -> None:
        """Test release transition."""
        sample_request.mark_routed([1])
        sample_request.mark_released()

        assert sample_request.status == RequestStatus.RELEASED

    def test_mark_released_from_pending(self, sample_request: Request) -> None:
        """Test invalid release from PENDING."""
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_released()

    def test_mark_released_from_blocked(self, sample_request: Request) -> None:
        """Test cannot release from BLOCKED."""
        sample_request.mark_blocked(BlockReason.NO_PATH)
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_released()

    def test_mark_groomed_from_invalid_state(self, sample_request: Request) -> None:
        """Test invalid transition to GROOMED."""
        sample_request.mark_allocated([1])  # Already allocated
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_groomed([2])

    def test_mark_partially_groomed_from_invalid_state(self, sample_request: Request) -> None:
        """Test invalid transition to PARTIALLY_GROOMED."""
        sample_request.mark_allocated([1])  # Already allocated
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_partially_groomed([2])

    def test_mark_partially_groomed_empty_lightpaths(self, sample_request: Request) -> None:
        """Test mark_partially_groomed requires lightpath_ids."""
        with pytest.raises(ValueError, match="Must provide at least one lightpath_id"):
            sample_request.mark_partially_groomed([])

    def test_mark_releasing_from_invalid_state(self, sample_request: Request) -> None:
        """Test invalid transition to RELEASING from PENDING."""
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.mark_releasing()

    def test_full_lifecycle(self, sample_request: Request) -> None:
        """Test complete request lifecycle."""
        assert sample_request.status == RequestStatus.PENDING
        sample_request.mark_routed([1, 2])
        assert sample_request.status == RequestStatus.ALLOCATED
        sample_request.mark_released()
        assert sample_request.status == RequestStatus.RELEASED

    def test_full_lifecycle_with_processing(self, sample_request: Request) -> None:
        """Test complete request lifecycle with processing states."""
        assert sample_request.status == RequestStatus.PENDING
        sample_request.set_status(RequestStatus.ROUTING)
        assert sample_request.status == RequestStatus.ROUTING
        sample_request.set_status(RequestStatus.SPECTRUM_SEARCH)
        assert sample_request.status == RequestStatus.SPECTRUM_SEARCH
        sample_request.set_status(RequestStatus.ALLOCATED)
        assert sample_request.status == RequestStatus.ALLOCATED
        sample_request.mark_releasing()
        assert sample_request.status == RequestStatus.RELEASING
        sample_request.set_status(RequestStatus.RELEASED)
        assert sample_request.status == RequestStatus.RELEASED

    def test_set_status(self, sample_request: Request) -> None:
        """Test set_status method."""
        sample_request.set_status(RequestStatus.ROUTING)
        assert sample_request.status == RequestStatus.ROUTING

    def test_set_status_invalid(self, sample_request: Request) -> None:
        """Test set_status with invalid transition."""
        with pytest.raises(ValueError, match="Cannot transition from"):
            sample_request.set_status(RequestStatus.RELEASED)


class TestRequestLegacyConversion:
    """Test Request legacy adapter methods."""

    def test_from_legacy_dict_basic(self) -> None:
        """Test basic conversion from legacy dict."""
        legacy = {
            "req_id": 42,
            "source": "0",
            "destination": "5",
            "arrive": 12.345,
            "depart": 17.890,
            "bandwidth": "100Gbps",
        }
        request = Request.from_legacy_dict((42, 12.345), legacy)

        assert request.request_id == 42
        assert request.source == "0"
        assert request.destination == "5"
        assert request.bandwidth_gbps == 100
        assert request.arrive_time == 12.345
        assert request.depart_time == 17.890
        assert request.status == RequestStatus.PENDING

    def test_from_legacy_dict_with_modformats(self) -> None:
        """Test conversion with modulation formats."""
        legacy = {
            "req_id": 1,
            "source": "A",
            "destination": "B",
            "arrive": 0.0,
            "depart": 5.0,
            "bandwidth": "50Gbps",
            "mod_formats": {"QPSK": {"slots": 4, "reach": 2000}},
        }
        request = Request.from_legacy_dict((1, 0.0), legacy)

        assert request.modulation_formats == {"QPSK": {"slots": 4, "reach": 2000}}

    def test_from_legacy_dict_bandwidth_formats(self) -> None:
        """Test various bandwidth string formats."""
        base = {
            "req_id": 1,
            "source": "0",
            "destination": "1",
            "arrive": 0.0,
            "depart": 1.0,
        }

        # "100Gbps"
        base["bandwidth"] = "100Gbps"
        assert Request.from_legacy_dict((1, 0.0), base).bandwidth_gbps == 100

        # "100 Gbps" (with space)
        base["bandwidth"] = "100 Gbps"
        assert Request.from_legacy_dict((1, 0.0), base).bandwidth_gbps == 100

        # "100gbps" (lowercase)
        base["bandwidth"] = "100gbps"
        assert Request.from_legacy_dict((1, 0.0), base).bandwidth_gbps == 100

        # "50GBPS" (uppercase)
        base["bandwidth"] = "50GBPS"
        assert Request.from_legacy_dict((1, 0.0), base).bandwidth_gbps == 50

        # Integer passthrough
        base["bandwidth"] = 100
        assert Request.from_legacy_dict((1, 0.0), base).bandwidth_gbps == 100

    def test_from_legacy_dict_numeric_nodes(self) -> None:
        """Test that numeric node IDs are converted to strings."""
        legacy = {
            "req_id": 1,
            "source": 0,
            "destination": 5,
            "arrive": 0.0,
            "depart": 1.0,
            "bandwidth": "100Gbps",
        }
        request = Request.from_legacy_dict((1, 0.0), legacy)

        assert request.source == "0"
        assert request.destination == "5"

    def test_from_legacy_dict_request_id_priority(self) -> None:
        """Test request_id parameter takes priority."""
        legacy = {
            "req_id": 42,
            "source": "0",
            "destination": "1",
            "arrive": 0.0,
            "depart": 1.0,
            "bandwidth": "100Gbps",
        }
        # Override with parameter
        request = Request.from_legacy_dict((42, 0.0), legacy, request_id=99)
        assert request.request_id == 99

    def test_from_legacy_dict_request_id_from_dict(self) -> None:
        """Test request_id from dict when not in time_key."""
        legacy = {
            "req_id": 42,
            "source": "0",
            "destination": "1",
            "arrive": 0.0,
            "depart": 1.0,
            "bandwidth": "100Gbps",
        }
        request = Request.from_legacy_dict((0, 0.0), legacy)
        assert request.request_id == 42

    def test_from_legacy_dict_request_id_from_time_key(self) -> None:
        """Test request_id from time_key when not in dict."""
        legacy = {
            "source": "0",
            "destination": "1",
            "arrive": 0.0,
            "depart": 1.0,
            "bandwidth": "100Gbps",
        }
        request = Request.from_legacy_dict((99, 0.0), legacy)
        assert request.request_id == 99

    def test_from_legacy_dict_arrive_from_time_key(self) -> None:
        """Test arrive_time from time_key when not in dict."""
        legacy = {
            "source": "0",
            "destination": "1",
            "depart": 5.0,
            "bandwidth": "100Gbps",
        }
        request = Request.from_legacy_dict((1, 2.5), legacy)
        assert request.arrive_time == 2.5

    def test_from_legacy_dict_none_modformats(self) -> None:
        """Test handling of None mod_formats."""
        legacy = {
            "req_id": 1,
            "source": "0",
            "destination": "1",
            "arrive": 0.0,
            "depart": 1.0,
            "bandwidth": "100Gbps",
            "mod_formats": None,
        }
        request = Request.from_legacy_dict((1, 0.0), legacy)
        assert request.modulation_formats == {}

    def test_to_legacy_dict_basic(self) -> None:
        """Test basic conversion to legacy dict."""
        request = Request(
            request_id=42,
            source="0",
            destination="5",
            bandwidth_gbps=100,
            arrive_time=12.345,
            depart_time=17.890,
        )
        legacy = request.to_legacy_dict()

        assert legacy["req_id"] == 42
        assert legacy["source"] == "0"
        assert legacy["destination"] == "5"
        assert legacy["arrive"] == 12.345
        assert legacy["depart"] == 17.890
        assert legacy["bandwidth"] == "100Gbps"
        assert legacy["request_type"] == "arrival"
        assert legacy["mod_formats"] == {}

    def test_to_legacy_dict_released(self) -> None:
        """Test request_type is 'release' for RELEASED status."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )
        request.mark_routed([1])
        request.mark_released()
        legacy = request.to_legacy_dict()

        assert legacy["request_type"] == "release"

    def test_to_legacy_dict_routed(self) -> None:
        """Test request_type for ROUTED status."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )
        request.mark_routed([1])
        legacy = request.to_legacy_dict()

        assert legacy["request_type"] == "arrival"

    def test_to_legacy_dict_blocked(self) -> None:
        """Test request_type for BLOCKED status."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )
        request.mark_blocked(BlockReason.CONGESTION)
        legacy = request.to_legacy_dict()

        assert legacy["request_type"] == "arrival"

    def test_to_legacy_dict_with_modformats(self) -> None:
        """Test conversion with modulation formats."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=50,
            arrive_time=0.0,
            depart_time=5.0,
            modulation_formats={"QPSK": {"slots": 4}},
        )
        legacy = request.to_legacy_dict()

        assert legacy["mod_formats"] == {"QPSK": {"slots": 4}}

    def test_to_legacy_time_key_pending(self) -> None:
        """Test legacy time key for PENDING status."""
        request = Request(
            request_id=42,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=10.0,
            depart_time=20.0,
        )
        assert request.to_legacy_time_key() == (42, 10.0)

    def test_to_legacy_time_key_routed(self) -> None:
        """Test legacy time key for ROUTED status."""
        request = Request(
            request_id=42,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=10.0,
            depart_time=20.0,
        )
        request.mark_routed([1])
        assert request.to_legacy_time_key() == (42, 10.0)

    def test_to_legacy_time_key_released(self) -> None:
        """Test legacy time key for RELEASED status uses depart_time."""
        request = Request(
            request_id=42,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=10.0,
            depart_time=20.0,
        )
        request.mark_routed([1])
        request.mark_released()
        assert request.to_legacy_time_key() == (42, 20.0)

    def test_to_legacy_time_key_blocked(self) -> None:
        """Test legacy time key for BLOCKED status."""
        request = Request(
            request_id=42,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=10.0,
            depart_time=20.0,
        )
        request.mark_blocked(BlockReason.NO_PATH)
        assert request.to_legacy_time_key() == (42, 10.0)


class TestRequestRoundtrip:
    """Test roundtrip conversion preserves data."""

    def test_roundtrip_basic(self) -> None:
        """Test basic roundtrip conversion."""
        original = {
            "req_id": 42,
            "source": "0",
            "destination": "5",
            "arrive": 12.345,
            "depart": 17.890,
            "bandwidth": "100Gbps",
            "mod_formats": {},
        }
        request = Request.from_legacy_dict((42, 12.345), original)
        roundtrip = request.to_legacy_dict()

        assert roundtrip["req_id"] == original["req_id"]
        assert roundtrip["source"] == original["source"]
        assert roundtrip["destination"] == original["destination"]
        assert roundtrip["arrive"] == original["arrive"]
        assert roundtrip["depart"] == original["depart"]
        assert roundtrip["bandwidth"] == original["bandwidth"]

    def test_roundtrip_with_modformats(self) -> None:
        """Test roundtrip with modulation formats."""
        original = {
            "req_id": 1,
            "source": "A",
            "destination": "B",
            "arrive": 0.0,
            "depart": 5.0,
            "bandwidth": "50Gbps",
            "mod_formats": {"QPSK": {"slots": 4}},
        }
        request = Request.from_legacy_dict((1, 0.0), original)
        roundtrip = request.to_legacy_dict()

        assert roundtrip["mod_formats"] == original["mod_formats"]

    def test_roundtrip_numeric_node_ids(self) -> None:
        """Test roundtrip with numeric node IDs in original."""
        original = {
            "req_id": 1,
            "source": 0,
            "destination": 5,
            "arrive": 0.0,
            "depart": 5.0,
            "bandwidth": "100Gbps",
            "mod_formats": {},
        }
        request = Request.from_legacy_dict((1, 0.0), original)
        roundtrip = request.to_legacy_dict()

        # Node IDs become strings
        assert roundtrip["source"] == "0"
        assert roundtrip["destination"] == "5"


class TestRequestMutability:
    """Test that Request fields can be mutated as expected."""

    def test_feature_flags_mutable(self) -> None:
        """Test that feature flags can be set after creation."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )

        request.is_groomed = True
        request.is_partially_groomed = True
        request.is_sliced = True
        # is_protected is now a computed property from protection_status
        request.protection_status = ProtectionStatus.ACTIVE_PRIMARY

        assert request.is_groomed is True
        assert request.is_partially_groomed is True
        assert request.is_sliced is True
        assert request.is_protected is True  # Computed from protection_status

    def test_lightpath_ids_mutable(self) -> None:
        """Test that lightpath_ids can be modified."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )

        request.lightpath_ids.append(1)
        request.lightpath_ids.append(2)

        assert request.lightpath_ids == [1, 2]

    def test_protection_fields_mutable(self) -> None:
        """Test that protection fields can be set after creation."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )

        # Default values
        assert request.protection_status == ProtectionStatus.UNPROTECTED
        assert request.primary_path is None
        assert request.backup_path is None
        assert request.active_path == "primary"  # Computed property
        assert request.last_switchover_time is None

        # Set protection fields
        request.protection_status = ProtectionStatus.ACTIVE_BACKUP
        request.primary_path = ["0", "2", "1"]
        request.backup_path = ["0", "3", "1"]
        request.last_switchover_time = 12.345

        assert request.is_protected is True  # Computed property
        assert request.primary_path == ["0", "2", "1"]
        assert request.backup_path == ["0", "3", "1"]
        assert request.active_path == "backup"  # Computed property
        assert request.last_switchover_time == 12.345


class TestRequestProtectionFields:
    """Test Request protection field functionality."""

    def test_default_protection_values(self) -> None:
        """Test default values for protection fields."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
        )

        assert request.protection_status == ProtectionStatus.UNPROTECTED
        assert request.is_protected is False  # Computed
        assert request.primary_path is None
        assert request.backup_path is None
        assert request.active_path == "primary"  # Computed
        assert request.last_switchover_time is None
        assert request.is_protection_degraded is False
        assert request.is_protection_failed is False

    def test_create_protected_request(self) -> None:
        """Test creating a protected request with paths."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
            protection_status=ProtectionStatus.ACTIVE_PRIMARY,
            primary_path=["0", "2", "1"],
            backup_path=["0", "3", "1"],
        )

        assert request.protection_status == ProtectionStatus.ACTIVE_PRIMARY
        assert request.is_protected is True
        assert request.primary_path == ["0", "2", "1"]
        assert request.backup_path == ["0", "3", "1"]
        assert request.active_path == "primary"

    def test_switchover(self) -> None:
        """Test switching from primary to backup path."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
            protection_status=ProtectionStatus.ACTIVE_PRIMARY,
            primary_path=["0", "2", "1"],
            backup_path=["0", "3", "1"],
        )

        # Simulate switchover
        request.protection_status = ProtectionStatus.ACTIVE_BACKUP
        request.last_switchover_time = 10.5

        assert request.active_path == "backup"
        assert request.last_switchover_time == 10.5

    def test_protection_degraded(self) -> None:
        """Test degraded protection state."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
            protection_status=ProtectionStatus.PRIMARY_FAILED,
            primary_path=["0", "2", "1"],
            backup_path=["0", "3", "1"],
        )

        assert request.is_protected is True
        assert request.is_protection_degraded is True
        assert request.is_protection_failed is False
        assert request.active_path == "backup"

    def test_protection_failed(self) -> None:
        """Test failed protection state."""
        request = Request(
            request_id=1,
            source="0",
            destination="1",
            bandwidth_gbps=100,
            arrive_time=0.0,
            depart_time=5.0,
            protection_status=ProtectionStatus.BOTH_FAILED,
        )

        assert request.is_protected is True
        assert request.is_protection_degraded is False
        assert request.is_protection_failed is True
