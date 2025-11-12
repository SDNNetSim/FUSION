# Component 9: Statistics & Reporting

**File:** `fusion/reporting/statistics.py`
**Estimated Time:** 45 minutes
**Dependencies:** Components 1, 8 (Properties, Simulation)

## Overview

Add grooming-specific statistics collection and reporting. This includes tracking grooming success rates, lightpath utilization, and transponder usage.

## New Statistics to Track

### Grooming Metrics

1. **Grooming Success Rate**
   - Fully groomed requests (no new lightpath needed)
   - Partially groomed requests (some grooming + new lightpath)
   - Not groomed requests (new lightpath only)

2. **Lightpath Utilization**
   - Average bandwidth utilization per lightpath
   - Distribution of lightpath lifetimes
   - Number of requests per lightpath

3. **Transponder Usage**
   - Peak transponder usage per node
   - Average transponder availability
   - Blocking due to transponder exhaustion

4. **Resource Efficiency**
   - Bandwidth savings from grooming
   - Spectrum efficiency improvement
   - Reduction in new lightpath establishments

## Implementation

### 1. Extend Statistics Class

Add grooming statistics to the main statistics class:

```python
class SimulationStatistics:
    """
    Comprehensive statistics tracking for network simulations.
    """

    def __init__(self, engine_props: dict[str, Any]) -> None:
        """Initialize statistics collector."""
        self.engine_props = engine_props

        # Existing statistics
        # ... existing attributes ...

        # NEW: Grooming statistics
        if engine_props.get('is_grooming_enabled', False):
            self.grooming_stats = GroomingStatistics()


class GroomingStatistics:
    """
    Statistics specific to traffic grooming operations.
    """

    def __init__(self) -> None:
        """Initialize grooming statistics."""
        # Request-level grooming outcomes
        self.fully_groomed_count: int = 0
        self.partially_groomed_count: int = 0
        self.not_groomed_count: int = 0
        self.total_requests: int = 0

        # Lightpath tracking
        self.lightpaths_created: int = 0
        self.lightpaths_released: int = 0
        self.active_lightpaths: int = 0
        self.lightpath_utilization_list: list[float] = []

        # Bandwidth savings
        self.bandwidth_groomed: float = 0.0  # Total bandwidth groomed
        self.bandwidth_new_lightpath: float = 0.0  # Bandwidth on new lightpaths
        self.spectrum_saved: int = 0  # Slots saved by grooming

        # Transponder usage
        self.transponder_blocking_count: int = 0
        self.peak_transponder_usage_per_node: dict[str, int] = {}
        self.avg_transponder_availability_per_node: dict[str, list[float]] = {}

        # Time-series data
        self.grooming_rate_over_time: list[tuple[float, float]] = []  # (time, rate)
        self.lightpath_count_over_time: list[tuple[float, int]] = []  # (time, count)

    def update_grooming_outcome(
        self,
        was_groomed: bool,
        was_partially_groomed: bool,
        bandwidth: float,
        new_lightpaths: int
    ) -> None:
        """
        Update statistics for a request allocation attempt.

        :param was_groomed: Request was fully groomed
        :type was_groomed: bool
        :param was_partially_groomed: Request was partially groomed
        :type was_partially_groomed: bool
        :param bandwidth: Request bandwidth
        :type bandwidth: float
        :param new_lightpaths: Number of new lightpaths created
        :type new_lightpaths: int
        """
        self.total_requests += 1

        if was_groomed:
            self.fully_groomed_count += 1
            self.bandwidth_groomed += bandwidth
        elif was_partially_groomed:
            self.partially_groomed_count += 1
            # Partial: some groomed, some new lightpath
        else:
            self.not_groomed_count += 1
            self.bandwidth_new_lightpath += bandwidth

        self.lightpaths_created += new_lightpaths

    def update_lightpath_release(
        self,
        lightpath_id: int,
        utilization: float,
        lifetime: float
    ) -> None:
        """
        Update statistics when a lightpath is released.

        :param lightpath_id: ID of released lightpath
        :type lightpath_id: int
        :param utilization: Average utilization percentage
        :type utilization: float
        :param lifetime: Lightpath lifetime in seconds
        :type lifetime: float
        """
        self.lightpaths_released += 1
        self.active_lightpaths -= 1
        self.lightpath_utilization_list.append(utilization)

    def calculate_grooming_rate(self) -> float:
        """
        Calculate the overall grooming success rate.

        :return: Percentage of requests that were groomed (fully or partially)
        :rtype: float
        """
        if self.total_requests == 0:
            return 0.0

        groomed = self.fully_groomed_count + self.partially_groomed_count
        return (groomed / self.total_requests) * 100.0

    def calculate_bandwidth_savings(self) -> float:
        """
        Calculate bandwidth savings from grooming.

        :return: Percentage of bandwidth that was groomed vs new lightpaths
        :rtype: float
        """
        total_bw = self.bandwidth_groomed + self.bandwidth_new_lightpath
        if total_bw == 0:
            return 0.0

        return (self.bandwidth_groomed / total_bw) * 100.0

    def get_average_lightpath_utilization(self) -> float:
        """
        Get average utilization across all released lightpaths.

        :return: Average utilization percentage
        :rtype: float
        """
        if not self.lightpath_utilization_list:
            return 0.0

        return sum(self.lightpath_utilization_list) / len(self.lightpath_utilization_list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert statistics to dictionary for serialization.

        :return: Dictionary of all statistics
        :rtype: dict[str, Any]
        """
        return {
            'grooming_outcomes': {
                'fully_groomed': self.fully_groomed_count,
                'partially_groomed': self.partially_groomed_count,
                'not_groomed': self.not_groomed_count,
                'total_requests': self.total_requests,
                'grooming_rate': self.calculate_grooming_rate()
            },
            'lightpaths': {
                'created': self.lightpaths_created,
                'released': self.lightpaths_released,
                'active': self.active_lightpaths,
                'avg_utilization': self.get_average_lightpath_utilization()
            },
            'bandwidth': {
                'groomed': self.bandwidth_groomed,
                'new_lightpath': self.bandwidth_new_lightpath,
                'savings_percentage': self.calculate_bandwidth_savings()
            },
            'transponders': {
                'blocking_count': self.transponder_blocking_count,
                'peak_usage_per_node': self.peak_transponder_usage_per_node
            }
        }
```

### 2. Add Statistics Update Hooks

Add hooks in SDN controller to update statistics:

```python
# In fusion/core/sdn_controller.py

def allocate_request(...) -> None:
    """Main request allocation."""
    # ... allocation logic ...

    # Update grooming statistics
    if self.engine_props.get('is_grooming_enabled', False):
        self._update_grooming_stats()


def _update_grooming_stats(self) -> None:
    """Update grooming statistics after request processing."""
    if not hasattr(self, 'stats') or self.stats is None:
        return

    if not hasattr(self.stats, 'grooming_stats'):
        return

    self.stats.grooming_stats.update_grooming_outcome(
        was_groomed=self.sdn_props.was_groomed or False,
        was_partially_groomed=self.sdn_props.was_partially_groomed,
        bandwidth=float(self.sdn_props.bandwidth) if self.sdn_props.bandwidth else 0.0,
        new_lightpaths=len(self.sdn_props.was_new_lp_established)
    )
```

### 3. Add Reporting Functions

Add functions to generate grooming reports:

```python
def generate_grooming_report(stats: GroomingStatistics) -> str:
    """
    Generate human-readable grooming statistics report.

    :param stats: Grooming statistics object
    :type stats: GroomingStatistics
    :return: Formatted report string
    :rtype: str
    """
    report = []
    report.append("=" * 60)
    report.append("Traffic Grooming Statistics Report")
    report.append("=" * 60)
    report.append("")

    # Grooming outcomes
    report.append("Grooming Outcomes:")
    report.append(f"  Total Requests: {stats.total_requests}")
    report.append(f"  Fully Groomed: {stats.fully_groomed_count} "
                  f"({stats.fully_groomed_count/stats.total_requests*100:.1f}%)")
    report.append(f"  Partially Groomed: {stats.partially_groomed_count} "
                  f"({stats.partially_groomed_count/stats.total_requests*100:.1f}%)")
    report.append(f"  Not Groomed: {stats.not_groomed_count} "
                  f"({stats.not_groomed_count/stats.total_requests*100:.1f}%)")
    report.append(f"  Grooming Success Rate: {stats.calculate_grooming_rate():.2f}%")
    report.append("")

    # Lightpath statistics
    report.append("Lightpath Statistics:")
    report.append(f"  Lightpaths Created: {stats.lightpaths_created}")
    report.append(f"  Lightpaths Released: {stats.lightpaths_released}")
    report.append(f"  Active Lightpaths: {stats.active_lightpaths}")
    report.append(f"  Avg Utilization: {stats.get_average_lightpath_utilization():.2f}%")
    report.append("")

    # Bandwidth savings
    report.append("Bandwidth Efficiency:")
    report.append(f"  Bandwidth Groomed: {stats.bandwidth_groomed:.2f} Gbps")
    report.append(f"  Bandwidth New LP: {stats.bandwidth_new_lightpath:.2f} Gbps")
    report.append(f"  Savings: {stats.calculate_bandwidth_savings():.2f}%")
    report.append("")

    # Transponder usage
    if stats.transponder_blocking_count > 0:
        report.append("Transponder Usage:")
        report.append(f"  Transponder Blocking: {stats.transponder_blocking_count}")
        report.append("")

    report.append("=" * 60)

    return "\n".join(report)


def export_grooming_stats_csv(stats: GroomingStatistics, filepath: str) -> None:
    """
    Export grooming statistics to CSV file.

    :param stats: Grooming statistics object
    :type stats: GroomingStatistics
    :param filepath: Output CSV file path
    :type filepath: str
    """
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['Metric', 'Value'])

        # Write statistics
        data = stats.to_dict()

        for category, metrics in data.items():
            writer.writerow([f"--- {category.upper()} ---", ""])
            for key, value in metrics.items():
                writer.writerow([key, value])
            writer.writerow([])  # Empty row between categories
```

## Testing

Create tests in `fusion/reporting/tests/test_statistics.py`:

```python
def test_grooming_statistics_init():
    """Test grooming statistics initialization."""
    stats = GroomingStatistics()

    assert stats.fully_groomed_count == 0
    assert stats.total_requests == 0
    assert stats.lightpaths_created == 0


def test_update_grooming_outcome():
    """Test grooming outcome update."""
    stats = GroomingStatistics()

    stats.update_grooming_outcome(
        was_groomed=True,
        was_partially_groomed=False,
        bandwidth=100.0,
        new_lightpaths=0
    )

    assert stats.fully_groomed_count == 1
    assert stats.total_requests == 1
    assert stats.bandwidth_groomed == 100.0


def test_calculate_grooming_rate():
    """Test grooming rate calculation."""
    stats = GroomingStatistics()

    # Add some requests
    stats.update_grooming_outcome(True, False, 100, 0)
    stats.update_grooming_outcome(False, True, 100, 1)
    stats.update_grooming_outcome(False, False, 100, 1)

    rate = stats.calculate_grooming_rate()
    assert rate == pytest.approx(66.67, rel=0.01)  # 2 out of 3 groomed


def test_generate_grooming_report():
    """Test report generation."""
    stats = GroomingStatistics()
    stats.total_requests = 100
    stats.fully_groomed_count = 50

    report = generate_grooming_report(stats)

    assert "Traffic Grooming Statistics" in report
    assert "Total Requests: 100" in report
```

Run tests:

```bash
python -m pylint fusion/reporting/statistics.py
python -m mypy fusion/reporting/statistics.py
python -m pytest fusion/reporting/tests/test_statistics.py -v
```

## Validation Checklist

- [ ] `GroomingStatistics` class created
- [ ] Statistics update hooks added to SDN controller
- [ ] Grooming outcomes tracked
- [ ] Lightpath utilization tracked
- [ ] Bandwidth savings calculated
- [ ] Report generation function created
- [ ] CSV export function created
- [ ] All methods have proper type hints
- [ ] Code passes pylint and mypy
- [ ] Unit tests created and passing

## Next Component

After completing this component, proceed to: [Component 10: Tests](10-tests.md)
