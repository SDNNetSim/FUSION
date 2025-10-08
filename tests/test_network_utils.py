"""Tests for grooming utility functions."""

import pytest
from fusion.utils.network import average_bandwidth_usage


def test_average_bandwidth_usage_single_period():
    """Test bandwidth calculation with single time period."""
    bw_dict = {0.0: 50.0}
    departure_time = 100.0

    avg = average_bandwidth_usage(bw_dict, departure_time)

    assert avg == 50.0


def test_average_bandwidth_usage_multiple_periods():
    """Test bandwidth calculation with multiple time periods."""
    bw_dict = {
        0.0: 0.0,      # Start at 0% utilization
        100.0: 50.0,   # At t=100, increase to 50%
        200.0: 75.0,   # At t=200, increase to 75%
        300.0: 100.0   # At t=300, increase to 100%
    }
    departure_time = 400.0

    avg = average_bandwidth_usage(bw_dict, departure_time)

    # Manual calculation:
    # Period 1: 0% * 100 = 0
    # Period 2: 50% * 100 = 5000
    # Period 3: 75% * 100 = 7500
    # Period 4: 100% * 100 = 10000
    # Total: 22500 / 400 = 56.25%
    assert avg == pytest.approx(56.25, rel=1e-5)


def test_average_bandwidth_usage_empty():
    """Test with empty bandwidth dictionary."""
    bw_dict = {}
    departure_time = 100.0

    avg = average_bandwidth_usage(bw_dict, departure_time)

    assert avg == 0.0


def test_average_bandwidth_usage_invalid_time():
    """Test with invalid time progression."""
    bw_dict = {100.0: 50.0}
    departure_time = 50.0  # Earlier than start time

    with pytest.raises(ValueError, match="Invalid time progression"):
        average_bandwidth_usage(bw_dict, departure_time)


def test_average_bandwidth_usage_constant():
    """Test with constant bandwidth throughout."""
    bw_dict = {0.0: 80.0}
    departure_time = 500.0

    avg = average_bandwidth_usage(bw_dict, departure_time)

    assert avg == 80.0


def test_average_bandwidth_usage_fluctuating():
    """Test with fluctuating bandwidth."""
    bw_dict = {
        0.0: 100.0,   # Full utilization
        50.0: 25.0,   # Drop to 25%
        100.0: 75.0,  # Increase to 75%
        150.0: 50.0   # Drop to 50%
    }
    departure_time = 200.0

    avg = average_bandwidth_usage(bw_dict, departure_time)

    # (100*50 + 25*50 + 75*50 + 50*50) / 200 = 12500/200 = 62.5
    assert avg == pytest.approx(62.5, rel=1e-5)
