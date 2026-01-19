"""
Unit tests for recovery time tracking in SimStats.

Tests the Phase 3 survivability extensions to the SimStats class including:
- Recovery event recording
- Recovery statistics computation
- Failure window blocking probability
- CSV export of recovery metrics
"""

import pytest

from fusion.core.metrics import SimStats


@pytest.fixture
def stats_with_recovery() -> SimStats:
    """Create a SimStats instance with recovery tracking enabled."""
    engine_props = {
        "seed": 42,
        "recovery_timing_settings": {"failure_window_size": 1000},
        "num_requests": 2000,
        "save_snapshots": False,
    }
    return SimStats(engine_props, sim_info="test_simulation")


class TestRecoveryEventRecording:
    """Test suite for recovery event recording."""

    def test_record_protection_switchover(self, stats_with_recovery: SimStats) -> None:
        """Test recording of protection switchover event."""
        stats = stats_with_recovery

        stats.record_recovery_event(
            failure_time=100.0,
            recovery_time=100.05,  # 50ms later
            affected_requests=1,
            recovery_type="protection",
        )

        assert len(stats.recovery_times_ms) == 1
        assert stats.recovery_times_ms[0] == 50.0
        assert len(stats.recovery_events) == 1
        assert stats.recovery_events[0]["recovery_type"] == "protection"

    def test_record_restoration_event(self, stats_with_recovery: SimStats) -> None:
        """Test recording of restoration event."""
        stats = stats_with_recovery

        stats.record_recovery_event(
            failure_time=100.0,
            recovery_time=100.1,  # 100ms later
            affected_requests=1,
            recovery_type="restoration",
        )

        assert len(stats.recovery_times_ms) == 1
        assert stats.recovery_times_ms[0] == 100.0
        assert stats.recovery_events[0]["recovery_type"] == "restoration"

    def test_record_multiple_events(self, stats_with_recovery: SimStats) -> None:
        """Test recording multiple recovery events."""
        stats = stats_with_recovery

        # Record protection switchover
        stats.record_recovery_event(100.0, 100.05, 1, "protection")
        # Record restoration
        stats.record_recovery_event(100.0, 100.1, 1, "restoration")
        # Record another protection
        stats.record_recovery_event(100.0, 100.055, 1, "protection")

        assert len(stats.recovery_times_ms) == 3
        assert len(stats.recovery_events) == 3
        assert stats.recovery_times_ms == [50.0, 100.0, 55.0]

    def test_recovery_event_details_stored(self, stats_with_recovery: SimStats) -> None:
        """Test that full event details are stored."""
        stats = stats_with_recovery

        stats.record_recovery_event(
            failure_time=100.0,
            recovery_time=100.05,
            affected_requests=5,
            recovery_type="protection",
        )

        event = stats.recovery_events[0]
        assert event["failure_time"] == 100.0
        assert event["recovery_time"] == 100.05
        assert event["recovery_duration_ms"] == 50.0
        assert event["affected_requests"] == 5
        assert event["recovery_type"] == "protection"


class TestRecoveryStatistics:
    """Test suite for recovery statistics computation."""

    def test_get_recovery_stats_empty(self, stats_with_recovery: SimStats) -> None:
        """Test recovery stats with no events."""
        stats = stats_with_recovery
        recovery_stats = stats.get_recovery_stats()

        assert recovery_stats["mean_ms"] == 0.0
        assert recovery_stats["p95_ms"] == 0.0
        assert recovery_stats["max_ms"] == 0.0
        assert recovery_stats["count"] == 0

    def test_get_recovery_stats_single_event(self, stats_with_recovery: SimStats) -> None:
        """Test recovery stats with single event."""
        stats = stats_with_recovery

        stats.record_recovery_event(100.0, 100.05, 1, "protection")

        recovery_stats = stats.get_recovery_stats()
        assert recovery_stats["mean_ms"] == 50.0
        assert recovery_stats["p95_ms"] == 50.0
        assert recovery_stats["max_ms"] == 50.0
        assert recovery_stats["count"] == 1

    def test_get_recovery_stats_multiple_events(self, stats_with_recovery: SimStats) -> None:
        """Test recovery stats with multiple events."""
        stats = stats_with_recovery

        # Add recovery times: 50, 55, 60, ..., 100 (11 values)
        for i in range(11):
            recovery_time_ms = 50 + (i * 5)
            recovery_time_s = recovery_time_ms / 1000.0
            stats.record_recovery_event(
                failure_time=100.0,
                recovery_time=100.0 + recovery_time_s,
                affected_requests=1,
                recovery_type="protection",
            )

        recovery_stats = stats.get_recovery_stats()

        # Mean should be 75ms (middle of 50-100)
        assert recovery_stats["mean_ms"] == pytest.approx(75.0, rel=0.01)

        # P95 should be near 97.5ms (95th percentile of 50-100)
        assert 95 <= recovery_stats["p95_ms"] <= 100

        # Max should be 100ms
        assert recovery_stats["max_ms"] == 100.0

        # Count should be 11
        assert recovery_stats["count"] == 11

    def test_recovery_stats_mixed_types(self, stats_with_recovery: SimStats) -> None:
        """Test recovery stats with mixed protection and restoration events."""
        stats = stats_with_recovery

        # Add protection events (50ms each)
        for _ in range(5):
            stats.record_recovery_event(100.0, 100.05, 1, "protection")

        # Add restoration events (100ms each)
        for _ in range(5):
            stats.record_recovery_event(100.0, 100.1, 1, "restoration")

        recovery_stats = stats.get_recovery_stats()

        # Mean should be 75ms (average of 50 and 100)
        assert recovery_stats["mean_ms"] == pytest.approx(75.0, rel=0.01)
        assert recovery_stats["count"] == 10


class TestFailureWindowBP:
    """Test suite for failure window blocking probability."""

    def test_compute_failure_window_bp_no_blocks(self, stats_with_recovery: SimStats) -> None:
        """Test failure window BP with no blocks."""
        stats = stats_with_recovery

        # Create arrival times (one per second for 2000 requests)
        arrival_times = [float(i) for i in range(2000)]

        # No blocked requests
        blocked_requests: list[int] = []

        bp = stats.compute_failure_window_bp(
            failure_time=1000.0,
            arrival_times=arrival_times,
            blocked_requests=blocked_requests,
        )

        assert bp == 0.0
        assert len(stats.failure_window_bp) == 1

    def test_compute_failure_window_bp_with_blocks(self, stats_with_recovery: SimStats) -> None:
        """Test failure window BP with blocked requests."""
        stats = stats_with_recovery

        # Create arrival times
        arrival_times = [float(i) for i in range(2000)]

        # Failure at t=1000, blocks 50 requests in next 1000 arrivals
        blocked_requests = list(range(1000, 1050))

        bp = stats.compute_failure_window_bp(
            failure_time=1000.0,
            arrival_times=arrival_times,
            blocked_requests=blocked_requests,
        )

        # Expected: 50 blocks / 1000 arrivals = 0.05
        assert bp == pytest.approx(0.05, rel=0.01)

    def test_compute_failure_window_bp_window_size(self, stats_with_recovery: SimStats) -> None:
        """Test that window size parameter is respected."""
        stats = stats_with_recovery
        stats.failure_window_size = 500  # Set smaller window

        arrival_times = [float(i) for i in range(2000)]

        # Blocks in range [1000, 1050]
        blocked_requests = list(range(1000, 1050))

        bp = stats.compute_failure_window_bp(
            failure_time=1000.0,
            arrival_times=arrival_times,
            blocked_requests=blocked_requests,
        )

        # Only first 500 arrivals after failure: 50 blocks / 500 arrivals = 0.1
        assert bp == pytest.approx(0.1, rel=0.01)

    def test_compute_failure_window_bp_at_end(self, stats_with_recovery: SimStats) -> None:
        """Test failure window BP when failure is near end of simulation."""
        stats = stats_with_recovery

        arrival_times = [float(i) for i in range(2000)]

        # Failure at t=1500, only 500 arrivals left
        blocked_requests = list(range(1500, 1550))

        bp = stats.compute_failure_window_bp(
            failure_time=1500.0,
            arrival_times=arrival_times,
            blocked_requests=blocked_requests,
        )

        # Window should be truncated to 500 arrivals
        # 50 blocks / 500 arrivals = 0.1
        assert bp == pytest.approx(0.1, rel=0.01)


class TestFailureWindowStatistics:
    """Test suite for failure window statistics."""

    def test_get_failure_window_stats_empty(self, stats_with_recovery: SimStats) -> None:
        """Test failure window stats with no data."""
        stats = stats_with_recovery
        window_stats = stats.get_failure_window_stats()

        assert window_stats["mean"] == 0.0
        assert window_stats["p95"] == 0.0
        assert window_stats["count"] == 0

    def test_get_failure_window_stats_single(self, stats_with_recovery: SimStats) -> None:
        """Test failure window stats with single measurement."""
        stats = stats_with_recovery

        arrival_times = [float(i) for i in range(2000)]
        blocked_requests = list(range(1000, 1100))  # 100 blocks

        stats.compute_failure_window_bp(
            failure_time=1000.0,
            arrival_times=arrival_times,
            blocked_requests=blocked_requests,
        )

        window_stats = stats.get_failure_window_stats()

        assert window_stats["mean"] == pytest.approx(0.1, rel=0.01)
        assert window_stats["p95"] == pytest.approx(0.1, rel=0.01)
        assert window_stats["count"] == 1

    def test_get_failure_window_stats_multiple(self, stats_with_recovery: SimStats) -> None:
        """Test failure window stats with multiple measurements."""
        stats = stats_with_recovery

        arrival_times = [float(i) for i in range(5000)]

        # Add multiple failure window measurements with different BPs
        # BP = 0.05
        stats.compute_failure_window_bp(
            failure_time=1000.0,
            arrival_times=arrival_times,
            blocked_requests=list(range(1000, 1050)),
        )
        # BP = 0.10
        stats.compute_failure_window_bp(
            failure_time=2000.0,
            arrival_times=arrival_times,
            blocked_requests=list(range(2000, 2100)),
        )
        # BP = 0.15
        stats.compute_failure_window_bp(
            failure_time=3000.0,
            arrival_times=arrival_times,
            blocked_requests=list(range(3000, 3150)),
        )

        window_stats = stats.get_failure_window_stats()

        # Mean should be (0.05 + 0.10 + 0.15) / 3 = 0.10
        assert window_stats["mean"] == pytest.approx(0.1, rel=0.01)
        assert window_stats["count"] == 3


class TestRecoveryCSVExport:
    """Test suite for CSV export of recovery metrics."""

    def test_get_recovery_csv_row_empty(self, stats_with_recovery: SimStats) -> None:
        """Test CSV export with no recovery data."""
        stats = stats_with_recovery
        csv_row = stats.get_recovery_csv_row()

        assert csv_row["recovery_time_mean_ms"] == 0.0
        assert csv_row["recovery_time_p95_ms"] == 0.0
        assert csv_row["recovery_time_max_ms"] == 0.0
        assert csv_row["recovery_event_count"] == 0
        assert csv_row["bp_window_fail_mean"] == 0.0
        assert csv_row["bp_window_fail_p95"] == 0.0
        assert csv_row["failure_window_count"] == 0

    def test_get_recovery_csv_row_with_data(self, stats_with_recovery: SimStats) -> None:
        """Test CSV export with recovery data."""
        stats = stats_with_recovery

        # Add recovery events
        stats.record_recovery_event(100.0, 100.05, 1, "protection")
        stats.record_recovery_event(100.0, 100.1, 1, "restoration")

        # Add failure window BP
        arrival_times = [float(i) for i in range(2000)]
        blocked_requests = list(range(1000, 1100))
        stats.compute_failure_window_bp(
            failure_time=1000.0,
            arrival_times=arrival_times,
            blocked_requests=blocked_requests,
        )

        csv_row = stats.get_recovery_csv_row()

        # Check recovery metrics (use approx for floating point)
        assert csv_row["recovery_time_mean_ms"] == pytest.approx(75.0, abs=0.01)
        assert csv_row["recovery_event_count"] == 2

        # Check failure window metrics
        assert csv_row["bp_window_fail_mean"] == pytest.approx(0.1, rel=0.01)
        assert csv_row["failure_window_count"] == 1

        # Verify all expected keys are present
        expected_keys = [
            "recovery_time_mean_ms",
            "recovery_time_p95_ms",
            "recovery_time_max_ms",
            "recovery_event_count",
            "bp_window_fail_mean",
            "bp_window_fail_p95",
            "failure_window_count",
        ]
        for key in expected_keys:
            assert key in csv_row


class TestIntegration:
    """Integration tests for recovery tracking."""

    def test_full_recovery_tracking_workflow(self, stats_with_recovery: SimStats) -> None:
        """Test complete recovery tracking workflow."""
        stats = stats_with_recovery

        # Simulate failure scenario
        arrival_times = [float(i) for i in range(3000)]

        # Record failure and protection switchover (5 requests affected)
        stats.record_recovery_event(
            failure_time=1000.0,
            recovery_time=1000.05,
            affected_requests=5,
            recovery_type="protection",
        )

        # Record failure window BP
        blocked_requests = list(range(1000, 1050))  # 50 blocked
        bp = stats.compute_failure_window_bp(
            failure_time=1000.0,
            arrival_times=arrival_times,
            blocked_requests=blocked_requests,
        )

        # Verify recovery stats (use approx for floating point)
        recovery_stats = stats.get_recovery_stats()
        assert recovery_stats["mean_ms"] == pytest.approx(50.0, abs=0.01)
        assert recovery_stats["count"] == 1

        # Verify window stats
        assert bp == pytest.approx(0.05, rel=0.01)

        # Verify CSV export includes all data
        csv_row = stats.get_recovery_csv_row()
        assert csv_row["recovery_time_mean_ms"] == pytest.approx(50.0, abs=0.01)
        assert csv_row["bp_window_fail_mean"] == pytest.approx(0.05, rel=0.01)
