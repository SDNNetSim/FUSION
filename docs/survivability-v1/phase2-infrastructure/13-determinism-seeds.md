# Phase 2: Core Infrastructure

## 13 - Determinism & Seeds

**Section Reference**: 1.9 - Determinism & Seeds

**Purpose**: Ensure reproducible simulation results across runs by seeding all random number generators and enforcing deterministic behavior in all stochastic components.

**Location**: `fusion/core/simulation.py` (extensions)

**Estimated Effort**: 0.5 days

---

## Overview

Reproducibility is critical for:
- **Statistical analysis**: Measuring variance across seeds vs. experimental conditions
- **Debugging**: Reproducing specific failure scenarios
- **Paper claims**: Validating results across multiple trials
- **Comparison**: Ensuring fair baseline vs. RL policy comparisons

This module ensures all randomness sources are properly seeded and all non-deterministic behavior is controlled.

---

## 1. Seed Management

### Core Seeding Function

**Location**: `fusion/core/simulation.py`

```python
"""
Seed management for reproducible simulations.
"""

import random
import numpy as np
import time
from typing import Any


def seed_all_rngs(seed: int) -> None:
    """
    Seed all random number generators for reproducibility.

    Seeds:
    - Python's built-in random module
    - NumPy's random state
    - PyTorch's random state (CPU and CUDA)

    Also sets PyTorch to deterministic mode to prevent
    non-deterministic operations.

    :param seed: Random seed (integer)
    :type seed: int

    Example:
        >>> seed_all_rngs(42)
        >>> # All subsequent random operations are reproducible
        >>> random.random()
        0.6394267984578837
        >>> np.random.rand()
        0.3745401188473625
    """
    # Seed Python's random module
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Seed PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Enforce deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)

    except ImportError:
        # PyTorch not installed, skip
        pass


def generate_seed_from_time() -> int:
    """
    Generate a seed from current time.

    Used when no seed is explicitly provided.

    :return: Seed value
    :rtype: int

    Example:
        >>> seed = generate_seed_from_time()
        >>> print(seed)
        1678901234
    """
    return int(time.time() * 1000) % (2**31 - 1)


def validate_seed(seed: int) -> int:
    """
    Validate and normalize seed value.

    Ensures seed is in valid range for all RNGs.

    :param seed: Seed value
    :type seed: int
    :return: Validated seed
    :rtype: int
    :raises ValueError: If seed is out of valid range

    Example:
        >>> validate_seed(42)
        42
        >>> validate_seed(-1)
        ValueError: Seed must be non-negative
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    if seed > 2**31 - 1:
        raise ValueError(f"Seed must be < 2^31, got {seed}")

    return seed
```

---

## 2. Integration with SimulationEngine

### Extension to `fusion/core/simulation.py`

```python
class SimulationEngine:
    """
    Main simulation engine with deterministic behavior.
    """

    def __init__(self, engine_props: dict[str, Any]) -> None:
        """
        Initialize simulation engine with seed management.

        :param engine_props: Engine configuration
        :type engine_props: dict[str, Any]
        """
        # Seed management
        seed = engine_props.get('seed')

        if seed is None:
            # Generate seed from time if not provided
            seed = generate_seed_from_time()
            logger.info(f"No seed provided, generated: {seed}")
        else:
            # Validate provided seed
            seed = validate_seed(seed)
            logger.info(f"Using provided seed: {seed}")

        # Store seed in engine props for logging
        engine_props['seed'] = seed

        # Seed all RNGs
        seed_all_rngs(seed)
        logger.info("All RNGs seeded for reproducibility")

        # ... rest of initialization ...
        self.engine_props = engine_props

    def get_seed(self) -> int:
        """
        Get the seed used for this simulation.

        :return: Seed value
        :rtype: int
        """
        return self.engine_props['seed']
```

---

## 3. Deterministic Components

### Arrival Process

Ensure arrival times are deterministic given a seed:

```python
def generate_arrival_times(
    num_requests: int,
    erlang: float,
    holding_time: float,
    seed: int
) -> list[float]:
    """
    Generate deterministic arrival times.

    Uses seeded random state for Poisson arrival process.

    :param num_requests: Number of requests
    :type num_requests: int
    :param erlang: Load in Erlangs
    :type erlang: float
    :param holding_time: Mean holding time
    :type holding_time: float
    :param seed: Random seed
    :type seed: int
    :return: List of arrival times
    :rtype: list[float]
    """
    # Create local random state
    rng = np.random.RandomState(seed)

    # Arrival rate (lambda)
    arrival_rate = erlang / holding_time

    # Generate inter-arrival times (exponential distribution)
    inter_arrivals = rng.exponential(1.0 / arrival_rate, num_requests)

    # Compute cumulative arrival times
    arrival_times = np.cumsum(inter_arrivals)

    return arrival_times.tolist()
```

### Request Generation

Ensure request parameters (source, destination, bandwidth) are deterministic:

```python
def generate_requests(
    num_requests: int,
    topology: nx.Graph,
    bandwidth_distribution: dict[str, Any],
    seed: int
) -> list[dict[str, Any]]:
    """
    Generate deterministic request parameters.

    :param num_requests: Number of requests
    :type num_requests: int
    :param topology: Network topology
    :type topology: nx.Graph
    :param bandwidth_distribution: Bandwidth parameters
    :type bandwidth_distribution: dict[str, Any]
    :param seed: Random seed
    :type seed: int
    :return: List of request dictionaries
    :rtype: list[dict[str, Any]]
    """
    # Create local random state
    rng = np.random.RandomState(seed + 1)  # Offset to avoid correlation

    nodes = list(topology.nodes())
    requests = []

    for i in range(num_requests):
        # Select source and destination
        src, dst = rng.choice(nodes, size=2, replace=False)

        # Sample bandwidth
        bandwidth = rng.choice(
            bandwidth_distribution['values'],
            p=bandwidth_distribution['probabilities']
        )

        requests.append({
            'id': i,
            'source': src,
            'destination': dst,
            'bandwidth': bandwidth
        })

    return requests
```

---

## 4. Multi-Seed Execution

### Batch Runner

**Location**: `fusion/sim/batch_runner.py` (new or extend existing)

```python
"""
Batch simulation runner for multi-seed experiments.
"""

from typing import Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def run_multi_seed_experiment(
    config: dict[str, Any],
    seed_list: list[int],
    output_dir: str
) -> list[dict[str, Any]]:
    """
    Run simulation with multiple seeds for statistical analysis.

    :param config: Base configuration
    :type config: dict[str, Any]
    :param seed_list: List of seeds to run
    :type seed_list: list[int]
    :param output_dir: Output directory for results
    :type output_dir: str
    :return: List of result dictionaries
    :rtype: list[dict[str, Any]]

    Example:
        >>> config = load_config('survivability_experiment.ini')
        >>> seeds = [42, 43, 44, 45, 46]
        >>> results = run_multi_seed_experiment(config, seeds, 'results/')
        >>> print(len(results))
        5
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for seed in seed_list:
        logger.info(f"Running simulation with seed {seed}")

        # Create seed-specific config
        seed_config = config.copy()
        seed_config['seed'] = seed

        # Run simulation
        engine = SimulationEngine(seed_config)
        stats = engine.run()

        # Store results
        result = {
            'seed': seed,
            **stats.to_dict()
        }
        results.append(result)

        logger.info(f"Seed {seed} complete: BP={result.get('bp_overall', 0):.4f}")

    return results
```

---

## 5. Seed Recording

### Statistics Extension

**Location**: `fusion/reporting/statistics.py`

```python
class SimulationStatistics:
    """
    Statistics with seed tracking.
    """

    def __init__(self, engine_props: dict[str, Any]) -> None:
        """
        Initialize statistics with seed.

        :param engine_props: Engine configuration
        :type engine_props: dict[str, Any]
        """
        # ... existing initialization ...

        # Record seed
        self.seed = engine_props.get('seed', 0)

    def to_dict(self) -> dict[str, Any]:
        """
        Export statistics including seed.

        :return: Statistics dictionary
        :rtype: dict[str, Any]
        """
        return {
            'seed': self.seed,
            'bp_overall': self.compute_blocking_probability(),
            # ... other metrics ...
        }

    def to_csv_row(self) -> dict[str, Any]:
        """
        Export as CSV row with seed.

        :return: CSV row dictionary
        :rtype: dict[str, Any]
        """
        row = self.to_dict()
        row['seed'] = self.seed
        return row
```

---

## 6. Configuration

### Seed Configuration

```ini
[general_settings]
# Explicit seed for reproducibility
seed = 42

# Or leave unset to auto-generate
# seed =
```

### Multi-Seed Configuration

```ini
[reporting]
# Run with multiple seeds
seed_list = [42, 43, 44, 45, 46]
aggregate_seeds = true
```

---

## 7. Testing Requirements

### Unit Tests

```python
import pytest
import numpy as np
from fusion.core.simulation import seed_all_rngs, generate_seed_from_time, validate_seed


def test_reproducibility_with_same_seed():
    """Test that identical results with same seed."""
    seed = 42

    # First run
    seed_all_rngs(seed)
    result1 = [np.random.rand() for _ in range(10)]

    # Second run with same seed
    seed_all_rngs(seed)
    result2 = [np.random.rand() for _ in range(10)]

    # Should be identical
    np.testing.assert_array_equal(result1, result2)


def test_different_results_with_different_seeds():
    """Test that different results with different seeds."""
    # Run 1 with seed 42
    seed_all_rngs(42)
    result1 = [np.random.rand() for _ in range(10)]

    # Run 2 with seed 43
    seed_all_rngs(43)
    result2 = [np.random.rand() for _ in range(10)]

    # Should be different
    assert not np.allclose(result1, result2)


def test_seed_recorded_in_outputs():
    """Test that seed value logged in CSV and datasets."""
    from fusion.core.simulation import SimulationEngine

    config = {
        'seed': 42,
        'num_requests': 100,
        # ... other config ...
    }

    engine = SimulationEngine(config)
    assert engine.get_seed() == 42

    # Check statistics
    stats = engine.statistics
    assert stats.seed == 42

    # Check CSV export
    csv_row = stats.to_csv_row()
    assert csv_row['seed'] == 42


def test_seed_generation_from_time():
    """Test that seed can be generated from time."""
    seed1 = generate_seed_from_time()
    seed2 = generate_seed_from_time()

    # Should be valid integers
    assert isinstance(seed1, int)
    assert isinstance(seed2, int)

    # Should be different (unless called in same millisecond)
    assert seed1 >= 0
    assert seed2 >= 0


def test_seed_validation():
    """Test seed validation."""
    # Valid seeds
    assert validate_seed(0) == 0
    assert validate_seed(42) == 42
    assert validate_seed(2**31 - 1) == 2**31 - 1

    # Invalid seeds
    with pytest.raises(ValueError, match="non-negative"):
        validate_seed(-1)

    with pytest.raises(ValueError, match="2\\^31"):
        validate_seed(2**31)


def test_torch_determinism():
    """Test that PyTorch operations are deterministic."""
    try:
        import torch

        seed = 42

        # Run 1
        seed_all_rngs(seed)
        x = torch.randn(10, 10)
        result1 = torch.matmul(x, x.t())

        # Run 2
        seed_all_rngs(seed)
        x = torch.randn(10, 10)
        result2 = torch.matmul(x, x.t())

        # Should be identical
        assert torch.allclose(result1, result2)

    except ImportError:
        pytest.skip("PyTorch not installed")
```

### Integration Tests

```python
def test_multi_seed_variance():
    """Test that multi-seed runs produce expected variance."""
    from fusion.sim.batch_runner import run_multi_seed_experiment

    config = {
        'num_requests': 1000,
        'erlang': 150,
        # ... other config ...
    }

    seeds = [42, 43, 44, 45, 46]
    results = run_multi_seed_experiment(config, seeds, 'test_output/')

    # Extract BP values
    bp_values = [r['bp_overall'] for r in results]

    # Should have variance across seeds
    assert np.std(bp_values) > 0

    # But not too much (sanity check)
    assert np.std(bp_values) < 0.1  # Adjust based on expected variance
```

---

## 8. Performance Considerations

### Random State Management

- **Local RNG**: Use `np.random.RandomState(seed)` for isolated components
- **Global RNG**: Use `np.random.seed(seed)` for global operations
- **Thread Safety**: Each thread should have its own seeded RNG

### Determinism Overhead

- PyTorch deterministic mode has ~10-20% performance penalty
- Trade-off between reproducibility and speed
- Can be disabled for production runs if reproducibility not needed

---

## 9. Acceptance Criteria

- [x] `test_reproducibility_with_same_seed`: Identical results with same seed
- [x] `test_different_results_with_different_seeds`: Different results with different seeds
- [x] `test_seed_recorded_in_outputs`: Seed value logged in CSV and datasets
- [x] `test_seed_generation_from_time`: Auto-generation works when seed not provided
- [x] `test_seed_validation`: Invalid seeds rejected with clear errors
- [x] All stochastic components seeded (arrivals, requests, routing, failures)
- [x] Multi-seed batch execution supported

---

## 10. Best Practices

### For Experiments

1. **Always set explicit seeds** for published results
2. **Use seed lists** (e.g., [42, 43, 44, 45, 46]) for variance analysis
3. **Record seeds** in all outputs (CSV, logs, datasets)
4. **Document seed ranges** in paper/report

### For Development

1. **Use fixed seeds** during debugging
2. **Test with multiple seeds** to catch non-deterministic bugs
3. **Log seed warnings** when auto-generated

### For Production

1. **Allow seed override** via CLI
2. **Default to time-based** if not provided
3. **Validate seed range** before running

---

## 11. Troubleshooting

### Non-Reproducible Results

If results differ across runs with same seed:

1. **Check library versions**: NumPy/PyTorch versions may differ
2. **Check threading**: Multi-threaded operations may be non-deterministic
3. **Check file I/O**: Reading order may vary
4. **Check hashing**: Hash functions may be non-deterministic

### Solutions

```python
# Force single-threaded execution
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Disable multithreading in PyTorch
import torch
torch.set_num_threads(1)
```

---

## Notes

- **Seed Range**: Valid seeds are 0 to 2^31-1 (NumPy/PyTorch constraint)
- **Offset Seeds**: Use `seed + offset` for different components to avoid correlation
- **Cross-Platform**: Results may differ slightly across CPU architectures
- **Version Control**: Document library versions for perfect reproducibility

---

**Related Documents**:
- [40-metrics-reporting.md](../phase5-metrics/40-metrics-reporting.md) (Seed aggregation)
- [50-testing.md](../phase6-quality/50-testing.md) (Testing with seeds)
- [63-usage-workflow.md](../phase7-management/63-usage-workflow.md) (Multi-seed experiments)
