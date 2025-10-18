"""
Unit tests for grooming statistics tracking.
"""

import os
import tempfile

import pytest

from fusion.reporting.statistics import (
    GroomingStatistics,
    SimulationStatistics,
    export_grooming_stats_csv,
    generate_grooming_report,
)


def test_grooming_statistics_init() -> None:
    """Test grooming statistics initialization."""
    stats = GroomingStatistics()

    assert stats.fully_groomed_count == 0
    assert stats.total_requests == 0
    assert stats.lightpaths_created == 0
    assert stats.partially_groomed_count == 0
    assert stats.not_groomed_count == 0
    assert stats.bandwidth_groomed == 0.0
    assert stats.bandwidth_new_lightpath == 0.0


def test_update_grooming_outcome_fully_groomed() -> None:
    """Test grooming outcome update for fully groomed request."""
    stats = GroomingStatistics()

    stats.update_grooming_outcome(
        was_groomed=True, was_partially_groomed=False, bandwidth=100.0, new_lightpaths=0
    )

    assert stats.fully_groomed_count == 1
    assert stats.total_requests == 1
    assert stats.bandwidth_groomed == 100.0
    assert stats.lightpaths_created == 0


def test_update_grooming_outcome_partially_groomed() -> None:
    """Test grooming outcome update for partially groomed request."""
    stats = GroomingStatistics()

    stats.update_grooming_outcome(
        was_groomed=False,
        was_partially_groomed=True,
        bandwidth=100.0,
        new_lightpaths=1,
    )

    assert stats.partially_groomed_count == 1
    assert stats.total_requests == 1
    assert stats.lightpaths_created == 1


def test_update_grooming_outcome_not_groomed() -> None:
    """Test grooming outcome update for not groomed request."""
    stats = GroomingStatistics()

    stats.update_grooming_outcome(
        was_groomed=False,
        was_partially_groomed=False,
        bandwidth=100.0,
        new_lightpaths=1,
    )

    assert stats.not_groomed_count == 1
    assert stats.total_requests == 1
    assert stats.bandwidth_new_lightpath == 100.0
    assert stats.lightpaths_created == 1


def test_calculate_grooming_rate() -> None:
    """Test grooming rate calculation."""
    stats = GroomingStatistics()

    # Add some requests
    stats.update_grooming_outcome(True, False, 100, 0)
    stats.update_grooming_outcome(False, True, 100, 1)
    stats.update_grooming_outcome(False, False, 100, 1)

    rate = stats.calculate_grooming_rate()
    assert rate == pytest.approx(66.67, rel=0.01)  # 2 out of 3 groomed


def test_calculate_grooming_rate_zero_requests() -> None:
    """Test grooming rate calculation with zero requests."""
    stats = GroomingStatistics()

    rate = stats.calculate_grooming_rate()
    assert rate == 0.0


def test_calculate_bandwidth_savings() -> None:
    """Test bandwidth savings calculation."""
    stats = GroomingStatistics()

    stats.bandwidth_groomed = 300.0
    stats.bandwidth_new_lightpath = 100.0

    savings = stats.calculate_bandwidth_savings()
    assert savings == 75.0  # 300 / (300 + 100) * 100


def test_calculate_bandwidth_savings_zero() -> None:
    """Test bandwidth savings calculation with zero bandwidth."""
    stats = GroomingStatistics()

    savings = stats.calculate_bandwidth_savings()
    assert savings == 0.0


def test_update_lightpath_release() -> None:
    """Test lightpath release update."""
    stats = GroomingStatistics()
    stats.active_lightpaths = 5

    stats.update_lightpath_release(123, 75.5, 100.0)

    assert stats.lightpaths_released == 1
    assert stats.active_lightpaths == 4
    assert len(stats.lightpath_utilization_list) == 1
    assert stats.lightpath_utilization_list[0] == 75.5


def test_get_average_lightpath_utilization() -> None:
    """Test average lightpath utilization calculation."""
    stats = GroomingStatistics()

    stats.lightpath_utilization_list = [50.0, 60.0, 70.0]

    avg = stats.get_average_lightpath_utilization()
    assert avg == 60.0


def test_get_average_lightpath_utilization_empty() -> None:
    """Test average lightpath utilization with no data."""
    stats = GroomingStatistics()

    avg = stats.get_average_lightpath_utilization()
    assert avg == 0.0


def test_to_dict() -> None:
    """Test statistics serialization to dictionary."""
    stats = GroomingStatistics()
    stats.update_grooming_outcome(True, False, 100, 0)
    stats.update_grooming_outcome(False, False, 50, 1)

    data = stats.to_dict()

    assert "grooming_outcomes" in data
    assert "lightpaths" in data
    assert "bandwidth" in data
    assert "transponders" in data

    assert data["grooming_outcomes"]["fully_groomed"] == 1
    assert data["grooming_outcomes"]["not_groomed"] == 1
    assert data["grooming_outcomes"]["total_requests"] == 2
    assert data["lightpaths"]["created"] == 1
    assert data["bandwidth"]["groomed"] == 100.0
    assert data["bandwidth"]["new_lightpath"] == 50.0


def test_simulation_statistics_with_grooming() -> None:
    """Test SimulationStatistics initialization with grooming enabled."""
    engine_props = {"is_grooming_enabled": True}

    stats = SimulationStatistics(engine_props)

    assert hasattr(stats, "grooming_stats")
    assert stats.grooming_stats is not None
    assert isinstance(stats.grooming_stats, GroomingStatistics)


def test_simulation_statistics_without_grooming() -> None:
    """Test SimulationStatistics initialization with grooming disabled."""
    engine_props = {"is_grooming_enabled": False}

    stats = SimulationStatistics(engine_props)

    assert hasattr(stats, "grooming_stats")
    assert stats.grooming_stats is None


def test_generate_grooming_report() -> None:
    """Test report generation."""
    stats = GroomingStatistics()
    stats.total_requests = 100
    stats.fully_groomed_count = 50
    stats.partially_groomed_count = 30
    stats.not_groomed_count = 20
    stats.lightpaths_created = 70
    stats.bandwidth_groomed = 5000.0
    stats.bandwidth_new_lightpath = 2000.0

    report = generate_grooming_report(stats)

    assert "Traffic Grooming Statistics" in report
    assert "Total Requests: 100" in report
    assert "Fully Groomed: 50" in report
    assert "Grooming Success Rate: 80.00%" in report
    assert "Bandwidth Groomed: 5000.00 Gbps" in report


def test_export_grooming_stats_csv() -> None:
    """Test CSV export functionality."""
    stats = GroomingStatistics()
    stats.total_requests = 10
    stats.fully_groomed_count = 5

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        temp_path = f.name

    try:
        export_grooming_stats_csv(stats, temp_path)

        # Verify file was created and contains data
        assert os.path.exists(temp_path)
        with open(temp_path, encoding="utf-8") as f:
            content = f.read()
            assert "Metric,Value" in content
            assert "GROOMING_OUTCOMES" in content
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_multiple_grooming_outcomes() -> None:
    """Test statistics with multiple mixed outcomes."""
    stats = GroomingStatistics()

    # Simulate 10 requests with different outcomes
    for _ in range(4):
        stats.update_grooming_outcome(True, False, 100, 0)  # Fully groomed
    for _ in range(3):
        stats.update_grooming_outcome(False, True, 100, 1)  # Partially groomed
    for _ in range(3):
        stats.update_grooming_outcome(False, False, 100, 1)  # Not groomed

    assert stats.total_requests == 10
    assert stats.fully_groomed_count == 4
    assert stats.partially_groomed_count == 3
    assert stats.not_groomed_count == 3
    assert stats.calculate_grooming_rate() == 70.0  # 7 out of 10
    assert stats.lightpaths_created == 6  # 3 partially + 3 not groomed
