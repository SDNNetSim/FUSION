# Phase 1 Testing Strategy

This document defines the testing approach for Phase 1 domain objects.

## Test Organization

### Directory Structure

```
fusion/tests/
├── domain/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   ├── test_config.py        # SimulationConfig tests
│   ├── test_request.py       # Request tests
│   ├── test_lightpath.py     # Lightpath tests
│   └── test_results.py       # Result types tests
└── stats/
    ├── __init__.py
    ├── conftest.py           # Shared fixtures
    └── test_collector.py     # StatsCollector tests
```

### Naming Conventions

Test names follow the pattern: `test_<what>_<when>_<expected>`

```python
# Good
def test_request_status_when_pending_returns_is_arrival_true():
def test_config_from_engine_props_with_missing_keys_uses_defaults():
def test_lightpath_utilization_when_fully_allocated_returns_one():

# Avoid
def test_request():
def test_config_works():
```

---

## Test Categories

### Unit Tests

Pure function/method tests with no external dependencies.

```python
class TestSimulationConfig:
    """Unit tests for SimulationConfig."""

    def test_from_engine_props_creates_valid_config(self, sample_engine_props):
        """from_engine_props creates config with correct values."""
        config = SimulationConfig.from_engine_props(sample_engine_props)

        assert config.network_name == sample_engine_props["network"]
        assert config.cores_per_link == sample_engine_props["cores_per_link"]
        assert config.k_paths == sample_engine_props["k_paths"]

    def test_from_engine_props_with_missing_keys_uses_defaults(self):
        """Missing keys use sensible defaults."""
        config = SimulationConfig.from_engine_props({})

        assert config.cores_per_link == 1
        assert config.k_paths == 3
        assert config.grooming_enabled is False

    def test_frozen_config_raises_on_mutation(self, sample_config):
        """Frozen dataclass prevents mutation."""
        with pytest.raises(FrozenInstanceError):
            sample_config.k_paths = 5
```

### Property Tests

Test computed properties with various inputs.

```python
class TestRequestProperties:
    """Tests for Request computed properties."""

    def test_endpoint_key_returns_sorted_tuple(self):
        """endpoint_key always returns sorted tuple."""
        req1 = Request(source="Z", destination="A", ...)
        req2 = Request(source="A", destination="Z", ...)

        assert req1.endpoint_key == ("A", "Z")
        assert req2.endpoint_key == ("A", "Z")
        assert req1.endpoint_key == req2.endpoint_key

    def test_holding_time_computed_correctly(self):
        """holding_time is depart - arrive."""
        request = Request(arrive_time=1.5, depart_time=3.5, ...)

        assert request.holding_time == 2.0

    def test_is_arrival_when_pending(self):
        """is_arrival returns True when PENDING."""
        request = Request(status=RequestStatus.PENDING, ...)

        assert request.is_arrival is True

    def test_is_arrival_when_routed(self):
        """is_arrival returns False when ROUTED."""
        request = Request(status=RequestStatus.ROUTED, ...)

        assert request.is_arrival is False
```

### Roundtrip Tests

Verify legacy conversion preserves data.

```python
class TestLegacyConversion:
    """Tests for legacy dict conversion."""

    def test_config_roundtrip_preserves_data(self, sample_engine_props):
        """from_engine_props -> to_engine_props preserves data."""
        config = SimulationConfig.from_engine_props(sample_engine_props)
        result = config.to_engine_props()

        for key in sample_engine_props:
            assert result[key] == sample_engine_props[key], f"Mismatch for {key}"

    def test_request_roundtrip_preserves_data(self, sample_request_dict):
        """from_legacy_dict -> to_legacy_dict preserves data."""
        time_key = (42, 1.5)
        request = Request.from_legacy_dict(time_key, sample_request_dict)
        result = request.to_legacy_dict()

        assert result["source"] == sample_request_dict["source"]
        assert result["destination"] == sample_request_dict["destination"]
        assert result["bandwidth"] == sample_request_dict["bandwidth"]

    def test_lightpath_roundtrip_preserves_data(self, sample_lightpath_dict):
        """from_legacy_dict -> to_legacy_dict preserves data."""
        lp = Lightpath.from_legacy_dict(1, sample_lightpath_dict)
        result = lp.to_legacy_dict()

        for key in sample_lightpath_dict:
            assert result[key] == sample_lightpath_dict[key], f"Mismatch for {key}"
```

### Edge Case Tests

Test boundary conditions and unusual inputs.

```python
class TestEdgeCases:
    """Edge case tests."""

    def test_lightpath_with_zero_remaining_bandwidth(self):
        """Lightpath with no remaining capacity."""
        lp = Lightpath(
            total_bandwidth_gbps=100,
            remaining_bandwidth_gbps=0,
            ...
        )

        assert lp.utilization == 1.0

    def test_lightpath_with_empty_request_allocations(self):
        """Lightpath with no requests allocated."""
        lp = Lightpath(
            total_bandwidth_gbps=100,
            remaining_bandwidth_gbps=100,
            request_allocations={},
            ...
        )

        assert lp.utilization == 0.0

    def test_route_result_empty_paths(self):
        """RouteResult with no paths."""
        result = RouteResult(paths=[], modulations=[], weights_km=[])

        assert result.is_empty
        assert result.num_paths == 0

    def test_request_with_same_source_destination(self):
        """Request with same source and destination is invalid."""
        with pytest.raises(ValueError):
            Request(source="A", destination="A", ...)
```

### State Transition Tests

Test valid and invalid state changes.

```python
class TestStateTransitions:
    """Tests for state transition rules."""

    def test_pending_to_routed_valid(self):
        """PENDING -> ROUTED is valid."""
        request = Request(status=RequestStatus.PENDING, ...)
        request.status = RequestStatus.ROUTED
        request.lightpath_ids = [1]

        assert request.is_successful

    def test_pending_to_blocked_valid(self):
        """PENDING -> BLOCKED is valid."""
        request = Request(status=RequestStatus.PENDING, ...)
        request.status = RequestStatus.BLOCKED
        request.block_reason = "no_spectrum"

        assert request.is_blocked

    def test_routed_to_released_valid(self):
        """ROUTED -> RELEASED is valid."""
        request = Request(status=RequestStatus.ROUTED, lightpath_ids=[1], ...)
        request.status = RequestStatus.RELEASED

        assert request.status == RequestStatus.RELEASED
```

---

## Fixtures

### conftest.py for domain tests

```python
# fusion/tests/domain/conftest.py

import pytest
from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request, RequestStatus
from fusion.domain.lightpath import Lightpath


@pytest.fixture
def sample_engine_props() -> dict:
    """Sample engine_props dictionary."""
    return {
        "network": "USbackbone60",
        "cores_per_link": 7,
        "band_list": ["c", "l"],
        "c_band": 320,
        "l_band": 320,
        "s_band": 0,
        "guard_slots": 1,
        "num_requests": 1000,
        "erlang": 100.0,
        "holding_time": 1.0,
        "route_method": "k_shortest_path",
        "k_paths": 3,
        "allocation_method": "first_fit",
        "is_grooming_enabled": False,
        "max_segments": 1,
        "snr_type": None,
        "snr_recheck": False,
        "can_partially_serve": False,
        "modulation_formats_dict": {},
        "mod_per_bw": {},
        "req_snr": {},
    }


@pytest.fixture
def sample_config(sample_engine_props) -> SimulationConfig:
    """Sample SimulationConfig."""
    return SimulationConfig.from_engine_props(sample_engine_props)


@pytest.fixture
def sample_request_dict() -> dict:
    """Sample request dictionary."""
    return {
        "source": "A",
        "destination": "B",
        "bandwidth": 100,
        "arrive": 1.5,
        "depart": 3.5,
        "mod_format": None,
        "path": None,
    }


@pytest.fixture
def sample_request() -> Request:
    """Sample Request object."""
    return Request(
        request_id=42,
        source="A",
        destination="B",
        bandwidth_gbps=100,
        arrive_time=1.5,
        depart_time=3.5,
    )


@pytest.fixture
def sample_lightpath_dict() -> dict:
    """Sample lightpath dictionary."""
    return {
        "path": ["A", "C", "B"],
        "core": 0,
        "band": "c",
        "start_slot": 10,
        "end_slot": 18,
        "mod_format": "QPSK",
        "lightpath_bandwidth": 100,
        "remaining_bandwidth": 100,
        "snr_cost": 18.5,
        "xt_cost": None,
        "path_weight": 500.0,
        "is_degraded": False,
        "requests_dict": {},
    }


@pytest.fixture
def sample_lightpath() -> Lightpath:
    """Sample Lightpath object."""
    return Lightpath(
        lightpath_id=1,
        path=["A", "C", "B"],
        start_slot=10,
        end_slot=18,
        core=0,
        band="c",
        modulation="QPSK",
        total_bandwidth_gbps=100,
        remaining_bandwidth_gbps=100,
        path_weight_km=500.0,
    )
```

---

## Coverage Requirements

| Module | Target | Reason |
|--------|--------|--------|
| `config.py` | 95% | Core configuration, must be thorough |
| `request.py` | 95% | Core domain object, critical paths |
| `lightpath.py` | 95% | Core domain object, capacity tracking |
| `results.py` | 90% | Mostly data classes, some computed properties |
| `collector.py` | 90% | Statistics tracking, less complex logic |

### Measuring Coverage

```bash
# Run with coverage
pytest fusion/tests/domain/ fusion/tests/stats/ \
    --cov=fusion/domain --cov=fusion/stats \
    --cov-report=term-missing \
    --cov-report=html

# Check coverage meets thresholds
pytest ... --cov-fail-under=90
```

---

## Test Data Builders

For complex objects, use builder pattern:

```python
class RequestBuilder:
    """Builder for Request test objects."""

    def __init__(self):
        self._request_id = 1
        self._source = "A"
        self._destination = "B"
        self._bandwidth_gbps = 100
        self._arrive_time = 0.0
        self._depart_time = 1.0
        self._status = RequestStatus.PENDING

    def with_id(self, id: int) -> "RequestBuilder":
        self._request_id = id
        return self

    def with_endpoints(self, src: str, dst: str) -> "RequestBuilder":
        self._source = src
        self._destination = dst
        return self

    def with_bandwidth(self, bw: int) -> "RequestBuilder":
        self._bandwidth_gbps = bw
        return self

    def with_status(self, status: RequestStatus) -> "RequestBuilder":
        self._status = status
        return self

    def build(self) -> Request:
        return Request(
            request_id=self._request_id,
            source=self._source,
            destination=self._destination,
            bandwidth_gbps=self._bandwidth_gbps,
            arrive_time=self._arrive_time,
            depart_time=self._depart_time,
            status=self._status,
        )


# Usage in tests
def test_blocked_request():
    request = (
        RequestBuilder()
        .with_id(42)
        .with_status(RequestStatus.BLOCKED)
        .build()
    )
    assert request.is_blocked
```

---

## Running Tests

### Full Test Suite

```bash
# All Phase 1 tests
pytest fusion/tests/domain/ fusion/tests/stats/ -v

# With coverage
pytest fusion/tests/domain/ fusion/tests/stats/ -v --cov=fusion/domain --cov=fusion/stats
```

### Selective Tests

```bash
# Only config tests
pytest fusion/tests/domain/test_config.py -v

# Only tests matching pattern
pytest fusion/tests/domain/ -k "roundtrip" -v

# Only failed tests from last run
pytest fusion/tests/domain/ --lf
```

### CI Integration

```yaml
# In .github/workflows/quality.yml
- name: Run Phase 1 Tests
  run: |
    pytest fusion/tests/domain/ fusion/tests/stats/ \
      --cov=fusion/domain --cov=fusion/stats \
      --cov-fail-under=90 \
      --junitxml=junit.xml
```

---

## Regression Testing

Phase 1 is additive, so existing tests must continue to pass:

```bash
# Verify no regressions
pytest tests/ -v --ignore=fusion/tests/

# Verify comparison still works
python tests/run_comparison.py --config configs/test_config.ini
```
