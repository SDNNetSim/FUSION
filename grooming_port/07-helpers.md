# Component 7: Helper Functions

**File:** `fusion/utils/network.py` (or new `fusion/utils/grooming.py`)
**Estimated Time:** 15 minutes
**Dependencies:** None

## Overview

Add helper function for calculating time-weighted average bandwidth utilization. This is used for lightpath bandwidth utilization statistics.

## Implementation

Add to `fusion/utils/network.py`:

```python
def average_bandwidth_usage(bw_dict: dict[float, float], departure_time: float) -> float:
    """
    Calculate time-weighted average bandwidth utilization.

    Given a dictionary mapping timestamps to bandwidth utilization percentages,
    calculates the weighted average utilization over the lifetime of a lightpath.

    :param bw_dict: Dictionary mapping arrival times to bandwidth usage percentages
    :type bw_dict: dict[float, float]
    :param departure_time: Time when the lightpath is released
    :type departure_time: float
    :return: Average bandwidth utilization percentage (0-100)
    :rtype: float

    Example:
        >>> bw_dict = {0.0: 50.0, 100.0: 75.0, 200.0: 100.0}
        >>> average_bandwidth_usage(bw_dict, 300.0)
        75.0
    """
    if not bw_dict:
        return 0.0

    sorted_times = sorted(bw_dict.keys())

    total_bw_time = 0.0  # Accumulated bandwidth * time
    total_time = 0.0     # Total time duration

    for i in range(len(sorted_times)):
        start_time = sorted_times[i]
        bw = bw_dict[start_time]

        # Determine end time for this period
        if i < len(sorted_times) - 1:
            end_time = sorted_times[i + 1]
        else:
            end_time = departure_time

        # Calculate duration and weighted bandwidth
        duration = end_time - start_time

        if duration < 0:
            # Sanity check
            raise ValueError(
                f"Invalid time progression: start_time={start_time}, end_time={end_time}"
            )

        total_bw_time += bw * duration
        total_time += duration

    # Calculate weighted average
    if total_time == 0:
        return 0.0

    average_utilization = total_bw_time / total_time

    return average_utilization
```

## Alternative: Create Separate Grooming Utils

If you prefer to organize grooming utilities separately, create:

`fusion/utils/grooming.py`:

```python
"""
Utility functions for traffic grooming operations.
"""

from typing import Any


def average_bandwidth_usage(bw_dict: dict[float, float], departure_time: float) -> float:
    """Calculate time-weighted average bandwidth utilization."""
    # ... implementation above ...


def calculate_fragmentation_entropy(spectrum_dict: dict[tuple, dict[str, Any]]) -> float:
    """
    Calculate spectrum fragmentation using entropy metric.

    :param spectrum_dict: Network spectrum dictionary
    :type spectrum_dict: dict[tuple, dict[str, Any]]
    :return: Fragmentation entropy value
    :rtype: float
    """
    # TODO: Implement if needed for fragmentation_metrics config
    pass


def calculate_external_fragmentation(
    spectrum_dict: dict[tuple, dict[str, Any]],
    band: str,
    core: int
) -> float:
    """
    Calculate external fragmentation metric.

    :param spectrum_dict: Network spectrum dictionary
    :type spectrum_dict: dict[tuple, dict[str, Any]]
    :param band: Spectral band
    :type band: str
    :param core: Core number
    :type core: int
    :return: External fragmentation value (0-1)
    :rtype: float
    """
    # TODO: Implement if needed for fragmentation_metrics config
    pass
```

## Testing

Create tests in `tests/test_network_utils.py` or `tests/test_grooming_utils.py`:

```python
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
```

Run tests:

```bash
# Lint
python -m pylint fusion/utils/network.py

# Type check
mypy fusion/utils/network.py

# Unit tests
python -m pytest tests/test_network_utils.py -v -k bandwidth
```

## Usage Example

```python
from fusion.utils.network import average_bandwidth_usage

# Example lightpath utilization over time
time_bw_usage = {
    0.0: 0.0,      # Lightpath created, no requests yet
    10.5: 33.3,    # First request groomed (33% utilization)
    25.7: 66.6,    # Second request groomed (66% utilization)
    40.2: 100.0    # Third request groomed (100% utilization)
}

departure_time = 100.0  # Lightpath released at t=100

avg_utilization = average_bandwidth_usage(time_bw_usage, departure_time)
print(f"Average utilization: {avg_utilization:.2f}%")
```

## Validation Checklist

- [ ] `average_bandwidth_usage()` function implemented
- [ ] Function added to appropriate utils module
- [ ] Type hints included
- [ ] Docstring with examples
- [ ] Error handling for invalid inputs
- [ ] Unit tests created with edge cases
- [ ] Tests cover: single period, multiple periods, empty dict, invalid times
- [ ] Code passes pylint and mypy
- [ ] All tests passing

## Next Component

After completing this component, proceed to: [Component 8: Simulation](08-simulation.md)
