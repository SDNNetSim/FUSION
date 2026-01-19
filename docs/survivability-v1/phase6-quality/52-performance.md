# Phase 6: Quality Assurance

## 52 - Performance Budgets

**Section Reference**: Section 7 - Performance Budgets

**Purpose**: Define performance requirements and constraints for survivability v1 to ensure efficient execution and scalability.

---

## 1. Time Constraints

### Per-Request Processing
- **Decision-time overhead** (policy + masking + path features): **≤ 2 ms** per request (K≤5)
  - Path feature extraction: ≤ 0.5 ms
  - Action mask computation: ≤ 0.3 ms
  - Policy inference: ≤ 1.2 ms

### Failure Processing
- **Failure injection** (F4 radius=2 on NSFNet): **≤ 10 ms** amortized
- **Path feasibility check**: **≤ 0.1 ms** per path

### Initialization
- **K-path cache initialization**:
  - NSFNet (14 nodes): **≤ 5 seconds**
  - USbackbone60 (60 nodes): **≤ 30 seconds**

### Dataset Logging
- **Logging throughput**: **≥ 50k transitions/minute** to JSONL
- **Per-transition overhead**: ≤ 0.02 ms

---

## 2. Memory Constraints

### Cache Sizes
- **K-path cache size**:
  - NSFNet with K=4: **≤ 100 MB**
  - USbackbone60 with K=4: **≤ 500 MB**

### Runtime State
- **FailureManager state**: **≤ 10 MB** for worst-case F4 failure
- **Active request tracking**: **≤ 50 MB** for 10k concurrent requests

### Dataset Logging
- **Dataset logger buffer**: Use file flushing, no unbounded memory growth
- **Maximum buffer size**: 1000 transitions (≤ 1 MB)

---

## 3. Performance Testing

### Benchmark Script

**Location**: `tests/performance/test_survivability_performance.py`

```python
"""
Performance benchmarks for survivability features.
"""

import time
import pytest


def test_decision_time_overhead():
    """Test that decision time ≤ 2ms per request."""
    # Setup
    engine = create_test_engine()
    request = create_test_request()

    # Measure
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        engine.handle_arrival(request)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    # Verify
    mean_time = sum(times) / len(times)
    assert mean_time <= 2.0, f"Mean decision time {mean_time:.2f}ms exceeds 2ms budget"


def test_k_path_cache_initialization_time():
    """Test that K-path cache initializes within budget."""
    topology = load_nsfnet_topology()

    start = time.time()
    cache = KPathCache(topology, k=4)
    elapsed = time.time() - start

    assert elapsed <= 5.0, f"Cache init took {elapsed:.2f}s, exceeds 5s budget"


def test_dataset_logging_throughput():
    """Test that dataset logging ≥ 50k transitions/minute."""
    logger = DatasetLogger('test_output.jsonl', {})

    start = time.time()
    for i in range(10000):
        logger.log_transition(
            state={'src': 0, 'dst': 1, 'paths': []},
            action=0,
            reward=1.0,
            next_state=None,
            action_mask=[True],
            meta={}
        )
    elapsed = time.time() - start

    throughput = 10000 / elapsed * 60  # transitions/minute
    assert throughput >= 50000, f"Throughput {throughput:.0f}/min, expected ≥50k/min"


def test_memory_usage():
    """Test that memory usage within constraints."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Create cache
    topology = load_nsfnet_topology()
    cache = KPathCache(topology, k=4)

    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    cache_mem = mem_after - mem_before

    assert cache_mem <= 100, f"Cache uses {cache_mem:.1f}MB, exceeds 100MB budget"
```

---

## 4. Optimization Guidelines

### Hot Paths
Focus optimization on:
1. Path feature extraction (called K times per request)
2. Action mask computation (per request)
3. RL policy forward pass (per request)

### Caching Strategy
- Pre-compute K-paths at initialization
- Cache path features when possible
- Avoid repeated graph traversals

### Memory Management
- Use generators for large datasets
- Flush dataset logger regularly
- Clear old failure events from history

---

## 5. Profiling

### CPU Profiling

```bash
# Profile simulation with cProfile
python -m cProfile -o profile.stats \
  -m fusion.cli.run_sim --config survivability_experiment.ini

# Analyze
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

### Memory Profiling

```bash
# Profile with memory_profiler
python -m memory_profiler fusion/core/simulation.py

# Visualize with memray
memray run fusion/cli/run_sim.py --config config.ini
memray flamegraph memray-output.bin
```

---

## 6. Acceptance Criteria

- [x] All time budgets met for NSFNet and USbackbone60
- [x] Memory usage within constraints
- [x] Performance tests pass
- [x] No performance regressions vs. baseline FUSION
- [x] Profiling data shows no obvious bottlenecks

---

**Related Documents**:
- [11-k-path-cache.md](../phase2-infrastructure/11-k-path-cache.md) (Cache performance)
- [30-rl-policies.md](../phase4-rl-integration/30-rl-policies.md) (Policy inference time)
- [31-dataset-logging.md](../phase4-rl-integration/31-dataset-logging.md) (Logging throughput)
