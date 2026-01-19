# Writing a New Pipeline

This tutorial shows how to create a new pipeline that plugs into the V4 architecture. Pipelines are modular components that handle specific aspects of request processing.

## Prerequisites

- [Getting Started with Domain Model](./getting_started_with_domain_model.md)
- [Architecture: Pipeline Interfaces](../architecture/pipeline_interfaces.md)
- [Architecture: Orchestration](../architecture/orchestration.md)

---

## Overview: What is a Pipeline?

A pipeline is a stateless component that:

1. Implements a **protocol** (interface)
2. Receives `NetworkState` by reference per call
3. Returns a typed **result object**
4. Does NOT store or cache network state

### Pipeline Types

| Pipeline | Responsibility | Result Type |
|----------|---------------|-------------|
| `RoutingPipeline` | Find candidate paths | `RouteResult` |
| `SpectrumPipeline` | Assign spectrum slots | `SpectrumResult` |
| `GroomingPipeline` | Reuse existing lightpaths | `GroomingResult` |
| `SNRPipeline` | Validate signal quality | `SNRResult` |
| `SlicingPipeline` | Split request into slices | `AllocationResult` |

---

## Step 1: Define the Protocol

Before implementing, understand the protocol your pipeline must satisfy.

### Example: Creating a QoSPipeline

Let's create a QoS (Quality of Service) pipeline that prioritizes routes based on request class.

```python
# fusion/interfaces/pipelines.py (add to existing)

from typing import Protocol

class QoSPipeline(Protocol):
    """Protocol for QoS prioritization."""

    def prioritize(
        self,
        route_result: RouteResult,
        request: Request,
        network_state: NetworkState,
    ) -> RouteResult:
        """
        Reorder routes based on QoS requirements.

        Args:
            route_result: Candidate routes from RoutingPipeline
            request: The incoming request with QoS class
            network_state: Current network state (read-only)

        Returns:
            RouteResult with paths reordered by QoS priority
        """
        ...
```

### Protocol Design Rules

1. **Receive NetworkState per call** - Never store it
2. **Return immutable result objects** - Use `@dataclass(frozen=True)`
3. **Single responsibility** - Do one thing well
4. **No side effects on read** - Only orchestrator mutates state

---

## Step 2: Implement the Pipeline

Create the implementation file.

### File: `fusion/pipelines/qos_pipeline.py`

```python
"""QoS prioritization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.request import Request
from fusion.domain.results import RouteResult


@dataclass
class QoSConfig:
    """QoS-specific configuration."""
    gold_max_hops: int = 3
    silver_max_hops: int = 5
    bronze_max_hops: int = 10
    gold_congestion_threshold: float = 0.3
    silver_congestion_threshold: float = 0.6


class StandardQoSPipeline:
    """
    QoS pipeline that prioritizes routes based on request class.

    Priority classes:
    - gold: Shortest paths, low congestion required
    - silver: Balance of length and congestion
    - bronze: Any available path
    """

    def __init__(self, config: SimulationConfig, qos_config: QoSConfig | None = None):
        """
        Initialize QoS pipeline.

        Args:
            config: Simulation configuration
            qos_config: Optional QoS-specific configuration
        """
        self.config = config
        self.qos_config = qos_config or QoSConfig()

    def prioritize(
        self,
        route_result: RouteResult,
        request: Request,
        network_state: NetworkState,
    ) -> RouteResult:
        """
        Reorder routes based on QoS requirements.

        Args:
            route_result: Candidate routes from RoutingPipeline
            request: The incoming request with QoS class
            network_state: Current network state

        Returns:
            RouteResult with paths filtered and reordered by QoS
        """
        if route_result.is_empty:
            return route_result

        qos_class = getattr(request, "qos_class", "bronze")

        # Score and filter paths
        scored_paths = []
        for i, path in enumerate(route_result.paths):
            score = self._compute_qos_score(path, qos_class, network_state)
            if score is not None:
                scored_paths.append((
                    score,
                    path,
                    route_result.weights_km[i],
                    route_result.modulations[i],
                ))

        if not scored_paths:
            return RouteResult(paths=[], weights_km=[], modulations=[])

        # Sort by score (lower is better)
        scored_paths.sort(key=lambda x: x[0])

        # Build result
        return RouteResult(
            paths=[p[1] for p in scored_paths],
            weights_km=[p[2] for p in scored_paths],
            modulations=[p[3] for p in scored_paths],
            strategy_name=route_result.strategy_name,
            metadata={
                **route_result.metadata,
                "qos_class": qos_class,
                "qos_scores": [p[0] for p in scored_paths],
            },
        )

    def _compute_qos_score(
        self,
        path: list[str],
        qos_class: str,
        network_state: NetworkState,
    ) -> float | None:
        """
        Compute QoS score for a path.

        Returns None if path doesn't meet QoS requirements.
        Lower score is better.
        """
        hop_count = len(path) - 1
        congestion = self._compute_congestion(path, network_state)

        if qos_class == "gold":
            if hop_count > self.qos_config.gold_max_hops:
                return None
            if congestion > self.qos_config.gold_congestion_threshold:
                return None
            # Gold prioritizes minimal hops
            return hop_count + congestion * 0.5

        elif qos_class == "silver":
            if hop_count > self.qos_config.silver_max_hops:
                return None
            if congestion > self.qos_config.silver_congestion_threshold:
                return None
            # Silver balances hops and congestion
            return hop_count * 0.5 + congestion

        else:  # bronze
            if hop_count > self.qos_config.bronze_max_hops:
                return None
            # Bronze accepts any path, prefers low congestion
            return congestion + hop_count * 0.1

    def _compute_congestion(
        self,
        path: list[str],
        network_state: NetworkState,
    ) -> float:
        """Compute average congestion across path links."""
        total_slots = 0
        used_slots = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            link_spectrum = network_state.get_link_spectrum(link)

            for band in self.config.band_list:
                matrix = link_spectrum.cores_matrix[band]
                total_slots += matrix.size
                used_slots += (matrix != 0).sum()

        if total_slots == 0:
            return 0.0

        return used_slots / total_slots
```

---

## Step 3: Add to PipelineSet

Update `PipelineSet` to include your new pipeline.

### Update: `fusion/core/pipeline_factory.py`

```python
from dataclasses import dataclass
from fusion.interfaces.pipelines import (
    RoutingPipeline, SpectrumPipeline, GroomingPipeline,
    SNRPipeline, SlicingPipeline, QoSPipeline,
)

@dataclass
class PipelineSet:
    """Container for all pipeline instances."""
    routing: RoutingPipeline
    spectrum: SpectrumPipeline
    grooming: GroomingPipeline | None = None
    snr: SNRPipeline | None = None
    slicing: SlicingPipeline | None = None
    qos: QoSPipeline | None = None  # NEW
```

---

## Step 4: Update PipelineFactory

Add factory method for your pipeline.

```python
class PipelineFactory:
    """Factory for creating pipeline instances."""

    @staticmethod
    def create_qos(config: SimulationConfig) -> QoSPipeline | None:
        """
        Create QoS pipeline if enabled.

        Args:
            config: Simulation configuration

        Returns:
            QoSPipeline instance or None if disabled
        """
        if not getattr(config, "qos_enabled", False):
            return None

        from fusion.pipelines.qos_pipeline import StandardQoSPipeline, QoSConfig

        qos_config = QoSConfig(
            gold_max_hops=getattr(config, "qos_gold_max_hops", 3),
            silver_max_hops=getattr(config, "qos_silver_max_hops", 5),
            bronze_max_hops=getattr(config, "qos_bronze_max_hops", 10),
        )

        return StandardQoSPipeline(config, qos_config)

    @staticmethod
    def create_pipeline_set(config: SimulationConfig) -> PipelineSet:
        """Create complete pipeline set based on configuration."""
        return PipelineSet(
            routing=PipelineFactory.create_routing(config),
            spectrum=PipelineFactory.create_spectrum(config),
            grooming=PipelineFactory.create_grooming(config),
            snr=PipelineFactory.create_snr(config),
            slicing=PipelineFactory.create_slicing(config),
            qos=PipelineFactory.create_qos(config),  # NEW
        )
```

---

## Step 5: Integrate with SDNOrchestrator

Add pipeline invocation to the orchestrator. Remember: only 3-5 lines should be added.

### Update: `fusion/core/orchestrator.py`

```python
class SDNOrchestrator:
    def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
        self.config = config
        self.routing = pipelines.routing
        self.spectrum = pipelines.spectrum
        self.grooming = pipelines.grooming
        self.snr = pipelines.snr
        self.slicing = pipelines.slicing
        self.qos = pipelines.qos  # NEW: Add reference

    def handle_arrival(
        self,
        request: Request,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> AllocationResult:
        # ... grooming stage ...

        # STAGE 2: Routing
        route_result = self.routing.find_routes(
            request.source, request.destination,
            remaining_bw, network_state, forced_path
        )

        # NEW: QoS prioritization (3 lines added)
        if self.qos and self.config.qos_enabled:
            route_result = self.qos.prioritize(route_result, request, network_state)

        if route_result.is_empty:
            return self._handle_failure(...)

        # ... rest of flow unchanged ...
```

---

## Step 6: Write Unit Tests

### File: `fusion/tests/pipelines/test_qos_pipeline.py`

```python
"""Tests for QoS pipeline."""

import pytest
import networkx as nx

from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.request import Request
from fusion.domain.results import RouteResult
from fusion.pipelines.qos_pipeline import StandardQoSPipeline, QoSConfig


@pytest.fixture
def simple_config():
    """Create test configuration."""
    return SimulationConfig(
        network_name="test",
        cores_per_link=1,
        band_list=("c",),
        band_slots={"c": 320},
        # ... other required fields
    )


@pytest.fixture
def linear_topology():
    """Create A-B-C-D-E linear topology."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=100)
    G.add_edge("B", "C", weight=100)
    G.add_edge("C", "D", weight=100)
    G.add_edge("D", "E", weight=100)
    return G


@pytest.fixture
def network_state(linear_topology, simple_config):
    """Create network state."""
    return NetworkState(linear_topology, simple_config)


@pytest.fixture
def route_result():
    """Create sample route result with multiple paths."""
    return RouteResult(
        paths=[
            ["A", "B", "C", "D", "E"],  # 4 hops
            ["A", "B", "E"],             # 2 hops (if existed)
        ],
        weights_km=[400.0, 200.0],
        modulations=[["QPSK"], ["QPSK"]],
    )


class TestQoSPipelineGoldClass:
    """Tests for gold QoS class."""

    def test_gold_rejects_long_paths(self, simple_config, network_state):
        """Gold class rejects paths exceeding max hops."""
        qos_config = QoSConfig(gold_max_hops=2)
        pipeline = StandardQoSPipeline(simple_config, qos_config)

        route_result = RouteResult(
            paths=[["A", "B", "C", "D"]],  # 3 hops
            weights_km=[300.0],
            modulations=[["QPSK"]],
        )

        request = Request(
            request_id=1, source="A", destination="D",
            bandwidth_gbps=100, arrival_time=0.0, holding_time=1.0,
        )
        request.qos_class = "gold"

        result = pipeline.prioritize(route_result, request, network_state)

        assert result.is_empty

    def test_gold_accepts_short_paths(self, simple_config, network_state):
        """Gold class accepts paths within max hops."""
        qos_config = QoSConfig(gold_max_hops=3)
        pipeline = StandardQoSPipeline(simple_config, qos_config)

        route_result = RouteResult(
            paths=[["A", "B", "C"]],  # 2 hops
            weights_km=[200.0],
            modulations=[["QPSK"]],
        )

        request = Request(
            request_id=1, source="A", destination="C",
            bandwidth_gbps=100, arrival_time=0.0, holding_time=1.0,
        )
        request.qos_class = "gold"

        result = pipeline.prioritize(route_result, request, network_state)

        assert not result.is_empty
        assert result.paths[0] == ["A", "B", "C"]


class TestQoSPipelineBronzeClass:
    """Tests for bronze QoS class."""

    def test_bronze_accepts_long_paths(self, simple_config, network_state):
        """Bronze class accepts longer paths."""
        qos_config = QoSConfig(bronze_max_hops=10)
        pipeline = StandardQoSPipeline(simple_config, qos_config)

        route_result = RouteResult(
            paths=[["A", "B", "C", "D", "E"]],  # 4 hops
            weights_km=[400.0],
            modulations=[["QPSK"]],
        )

        request = Request(
            request_id=1, source="A", destination="E",
            bandwidth_gbps=100, arrival_time=0.0, holding_time=1.0,
        )
        request.qos_class = "bronze"

        result = pipeline.prioritize(route_result, request, network_state)

        assert not result.is_empty


class TestQoSPipelineMetadata:
    """Tests for QoS metadata."""

    def test_qos_class_in_metadata(self, simple_config, network_state):
        """QoS class should appear in result metadata."""
        pipeline = StandardQoSPipeline(simple_config)

        route_result = RouteResult(
            paths=[["A", "B"]],
            weights_km=[100.0],
            modulations=[["QPSK"]],
        )

        request = Request(
            request_id=1, source="A", destination="B",
            bandwidth_gbps=100, arrival_time=0.0, holding_time=1.0,
        )
        request.qos_class = "silver"

        result = pipeline.prioritize(route_result, request, network_state)

        assert result.metadata.get("qos_class") == "silver"
        assert "qos_scores" in result.metadata
```

---

## Step 7: Add Configuration Support

### Update: `fusion/configs/templates/simulation.ini`

```ini
[qos]
# Enable QoS prioritization
qos_enabled = false

# Maximum hops for gold class
qos_gold_max_hops = 3

# Maximum hops for silver class
qos_silver_max_hops = 5

# Maximum hops for bronze class
qos_bronze_max_hops = 10

# Congestion thresholds (0.0 - 1.0)
qos_gold_congestion_threshold = 0.3
qos_silver_congestion_threshold = 0.6
```

### Update: `fusion/cli/parameters/network.py`

```python
QOS_PARAMS = {
    "qos_enabled": {"type": bool, "default": False},
    "qos_gold_max_hops": {"type": int, "default": 3},
    "qos_silver_max_hops": {"type": int, "default": 5},
    "qos_bronze_max_hops": {"type": int, "default": 10},
}
```

---

## Pipeline Design Checklist

Use this checklist when creating a new pipeline:

### Protocol
- [ ] Protocol defined in `fusion/interfaces/pipelines.py`
- [ ] Clear docstrings with Args and Returns
- [ ] NetworkState passed per call, not stored
- [ ] Returns appropriate result object

### Implementation
- [ ] Stateless - no instance variables storing request/state data
- [ ] Single responsibility
- [ ] No direct numpy array access outside NetworkState
- [ ] No hardcoded magic numbers
- [ ] Configuration via constructor

### Integration
- [ ] Added to `PipelineSet` dataclass
- [ ] Factory method in `PipelineFactory`
- [ ] Orchestrator integration (< 5 lines)
- [ ] Feature flag in config

### Testing
- [ ] Unit tests for happy path
- [ ] Tests for edge cases (empty input, no valid result)
- [ ] Tests for configuration variations
- [ ] Integration test with orchestrator

### Documentation
- [ ] Docstrings on all public methods
- [ ] Configuration documented in INI template
- [ ] Added to architecture docs if significant

---

## Common Patterns

### Wrapper Pattern

Add behavior to an existing pipeline:

```python
class LoggingSpectrumPipeline:
    """Wrapper that adds logging to spectrum pipeline."""

    def __init__(self, inner: SpectrumPipeline):
        self._inner = inner

    def find_spectrum(self, path, mods, bw, network_state) -> SpectrumResult:
        logger.debug(f"Finding spectrum for path {path}")
        result = self._inner.find_spectrum(path, mods, bw, network_state)
        logger.debug(f"Found: {result.is_free}")
        return result
```

### Metrics Wrapper

Collect performance metrics:

```python
class MetricsRoutingPipeline:
    """Wrapper that collects timing metrics."""

    def __init__(self, inner: RoutingPipeline, stats: StatsCollector):
        self._inner = inner
        self._stats = stats

    def find_routes(self, src, dst, bw, network_state, forced_path=None):
        start = time.perf_counter()
        result = self._inner.find_routes(src, dst, bw, network_state, forced_path)
        elapsed = time.perf_counter() - start
        self._stats.record_routing_time(elapsed)
        return result
```

### Fallback Pattern

Try multiple strategies:

```python
class FallbackRoutingPipeline:
    """Try primary strategy, fall back to secondary."""

    def __init__(self, primary: RoutingPipeline, fallback: RoutingPipeline):
        self._primary = primary
        self._fallback = fallback

    def find_routes(self, src, dst, bw, network_state, forced_path=None):
        result = self._primary.find_routes(src, dst, bw, network_state, forced_path)
        if result.is_empty:
            result = self._fallback.find_routes(src, dst, bw, network_state, forced_path)
        return result
```

---

## Anti-Patterns to Avoid

### 1. Storing NetworkState

```python
# BAD: Storing state
class BadPipeline:
    def __init__(self, network_state: NetworkState):
        self._state = network_state  # NEVER DO THIS

# GOOD: Receive per call
class GoodPipeline:
    def process(self, request: Request, network_state: NetworkState):
        # Use network_state only within this call
        ...
```

### 2. Algorithm Logic in Orchestrator

```python
# BAD: Algorithm in orchestrator
def handle_arrival(self, request, network_state):
    if request.qos_class == "gold":
        paths = self._filter_short_paths(...)  # WRONG PLACE

# GOOD: Delegate to pipeline
def handle_arrival(self, request, network_state):
    if self.qos:
        route_result = self.qos.prioritize(route_result, request, network_state)
```

### 3. Mutable Result Objects

```python
# BAD: Mutable result
@dataclass
class BadResult:
    paths: list[list[str]]

# GOOD: Immutable result
@dataclass(frozen=True)
class GoodResult:
    paths: tuple[tuple[str, ...], ...]
```

---

## Next Steps

- [Architecture: Pipeline Interfaces](../architecture/pipeline_interfaces.md) - Full protocol specifications
- [Architecture: Orchestration](../architecture/orchestration.md) - Orchestrator integration details
- [Adding a New Routing Strategy](./adding_a_new_routing_strategy.md) - Similar tutorial for routing
