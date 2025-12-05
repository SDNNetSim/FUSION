# Phase 5 Testing Plan

## Overview

Phase 5 introduces the ControlPolicy abstraction, heuristic policies, ML policies, and the protection pipeline. Testing focuses on:

1. **Protocol compliance**: All policies implement ControlPolicy correctly
2. **Policy correctness**: Each policy selects expected paths
3. **Protection functionality**: Disjoint paths, dual allocation, switchover
4. **Integration**: Policies work correctly with orchestrator
5. **Regression**: Phase 4 functionality preserved

## Test Structure

```
fusion/tests/
├── interfaces/
│   └── test_control_policy.py        # Protocol tests
├── policies/
│   ├── test_heuristic_policy.py      # Heuristic policy tests
│   ├── test_ml_policy.py             # ML policy tests
│   └── test_policy_factory.py        # Factory tests
├── pipelines/
│   └── test_protection_pipeline.py   # Protection pipeline tests
└── core/
    └── test_orchestrator.py          # Integration tests (extended)

tests/integration/
├── test_policy_simulation.py         # End-to-end policy tests
└── test_protection_simulation.py     # End-to-end protection tests
```

---

## Unit Tests

### 1. ControlPolicy Protocol Tests

**File**: `fusion/tests/interfaces/test_control_policy.py`

```python
import pytest
from typing import runtime_checkable
from fusion.interfaces.control_policy import ControlPolicy
from fusion.policies.heuristic_policy import FirstFeasiblePolicy


class TestControlPolicyProtocol:
    """Test ControlPolicy protocol definition."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        assert hasattr(ControlPolicy, '__protocol_attrs__')

    def test_heuristic_implements_protocol(self):
        """Heuristic policies should implement protocol."""
        policy = FirstFeasiblePolicy()
        assert isinstance(policy, ControlPolicy)

    def test_protocol_requires_select_action(self):
        """Protocol requires select_action method."""
        class InvalidPolicy:
            def update(self, request, action, reward):
                pass

        policy = InvalidPolicy()
        assert not isinstance(policy, ControlPolicy)

    def test_custom_policy_implements_protocol(self):
        """Custom class with correct methods should implement protocol."""
        class CustomPolicy:
            def select_action(self, request, options, network_state):
                return 0

            def update(self, request, action, reward):
                pass

        policy = CustomPolicy()
        assert isinstance(policy, ControlPolicy)
```

### 2. Heuristic Policy Tests

**File**: `fusion/tests/policies/test_heuristic_policy.py`

```python
import pytest
from fusion.policies.heuristic_policy import (
    FirstFeasiblePolicy,
    ShortestFeasiblePolicy,
    LeastCongestedPolicy,
    RandomFeasiblePolicy,
    LoadBalancedPolicy,
)
from fusion.modules.rl.env_adapter import PathOption


@pytest.fixture
def sample_options():
    """Create sample path options for testing."""
    return [
        PathOption(
            path_index=0,
            path=["A", "B", "C"],
            weight_km=100.0,
            is_feasible=False,  # Infeasible
            modulation="QPSK",
            slots_needed=10,
            congestion=0.8,
        ),
        PathOption(
            path_index=1,
            path=["A", "D", "C"],
            weight_km=200.0,
            is_feasible=True,
            modulation="QPSK",
            slots_needed=10,
            congestion=0.2,  # Lowest congestion
        ),
        PathOption(
            path_index=2,
            path=["A", "E", "C"],
            weight_km=150.0,  # Shortest feasible
            is_feasible=True,
            modulation="QPSK",
            slots_needed=10,
            congestion=0.5,
        ),
    ]


@pytest.fixture
def mock_request():
    """Create mock request."""
    from fusion.domain.request import Request, RequestStatus
    return Request(
        request_id=1,
        source="A",
        destination="C",
        bandwidth_gbps=100,
        holding_time=1.0,
        arrival_time=0.0,
        status=RequestStatus.PENDING,
    )


@pytest.fixture
def mock_network_state(mocker):
    """Create mock network state."""
    return mocker.MagicMock()


class TestFirstFeasiblePolicy:
    """Tests for FirstFeasiblePolicy."""

    def test_selects_first_feasible(self, sample_options, mock_request, mock_network_state):
        policy = FirstFeasiblePolicy()
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        assert action == 1  # First feasible (index 0 is infeasible)

    def test_returns_negative_when_all_infeasible(self, mock_request, mock_network_state):
        options = [
            PathOption(path_index=0, is_feasible=False, path=[], weight_km=0, modulation=None, slots_needed=None, congestion=0),
            PathOption(path_index=1, is_feasible=False, path=[], weight_km=0, modulation=None, slots_needed=None, congestion=0),
        ]
        policy = FirstFeasiblePolicy()
        action = policy.select_action(mock_request, options, mock_network_state)
        assert action == -1

    def test_returns_negative_for_empty_options(self, mock_request, mock_network_state):
        policy = FirstFeasiblePolicy()
        action = policy.select_action(mock_request, [], mock_network_state)
        assert action == -1

    def test_update_is_noop(self, mock_request):
        policy = FirstFeasiblePolicy()
        # Should not raise
        policy.update(mock_request, 0, 1.0)


class TestShortestFeasiblePolicy:
    """Tests for ShortestFeasiblePolicy."""

    def test_selects_shortest_feasible(self, sample_options, mock_request, mock_network_state):
        policy = ShortestFeasiblePolicy()
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        assert action == 2  # 150km < 200km

    def test_ignores_infeasible_even_if_shorter(self, mock_request, mock_network_state):
        options = [
            PathOption(path_index=0, weight_km=50.0, is_feasible=False, path=[], modulation=None, slots_needed=None, congestion=0),
            PathOption(path_index=1, weight_km=100.0, is_feasible=True, path=[], modulation=None, slots_needed=None, congestion=0),
        ]
        policy = ShortestFeasiblePolicy()
        action = policy.select_action(mock_request, options, mock_network_state)
        assert action == 1


class TestLeastCongestedPolicy:
    """Tests for LeastCongestedPolicy."""

    def test_selects_least_congested(self, sample_options, mock_request, mock_network_state):
        policy = LeastCongestedPolicy()
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        assert action == 1  # 0.2 congestion

    def test_ignores_infeasible_even_if_less_congested(self, mock_request, mock_network_state):
        options = [
            PathOption(path_index=0, congestion=0.1, is_feasible=False, path=[], weight_km=0, modulation=None, slots_needed=None),
            PathOption(path_index=1, congestion=0.5, is_feasible=True, path=[], weight_km=0, modulation=None, slots_needed=None),
        ]
        policy = LeastCongestedPolicy()
        action = policy.select_action(mock_request, options, mock_network_state)
        assert action == 1


class TestRandomFeasiblePolicy:
    """Tests for RandomFeasiblePolicy."""

    def test_selects_feasible_option(self, sample_options, mock_request, mock_network_state):
        policy = RandomFeasiblePolicy(seed=42)
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        assert action in [1, 2]  # Only feasible indices
        assert sample_options[action].is_feasible

    def test_reproducible_with_seed(self, sample_options, mock_request, mock_network_state):
        policy1 = RandomFeasiblePolicy(seed=42)
        policy2 = RandomFeasiblePolicy(seed=42)

        actions1 = [policy1.select_action(mock_request, sample_options, mock_network_state) for _ in range(10)]
        actions2 = [policy2.select_action(mock_request, sample_options, mock_network_state) for _ in range(10)]

        assert actions1 == actions2


class TestLoadBalancedPolicy:
    """Tests for LoadBalancedPolicy."""

    def test_alpha_zero_equals_least_congested(self, sample_options, mock_request, mock_network_state):
        policy = LoadBalancedPolicy(alpha=0.0)
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        # Should match LeastCongestedPolicy
        expected = LeastCongestedPolicy().select_action(mock_request, sample_options, mock_network_state)
        assert action == expected

    def test_alpha_one_equals_shortest(self, sample_options, mock_request, mock_network_state):
        policy = LoadBalancedPolicy(alpha=1.0)
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        # Should match ShortestFeasiblePolicy
        expected = ShortestFeasiblePolicy().select_action(mock_request, sample_options, mock_network_state)
        assert action == expected
```

### 3. ML Policy Tests

**File**: `fusion/tests/policies/test_ml_policy.py`

```python
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestMLControlPolicy:
    """Tests for MLControlPolicy."""

    @pytest.fixture
    def mock_pytorch_model(self):
        """Create mock PyTorch model."""
        model = MagicMock()
        model.eval = MagicMock()
        # Return logits favoring action 1
        model.return_value = MagicMock()
        model.return_value.argmax.return_value = MagicMock()
        model.return_value.argmax.return_value.item.return_value = 1
        return model

    @pytest.fixture
    def sample_options(self):
        from fusion.modules.rl.env_adapter import PathOption
        return [
            PathOption(path_index=0, is_feasible=True, path=["A", "B"], weight_km=100, modulation="QPSK", slots_needed=10, congestion=0.5),
            PathOption(path_index=1, is_feasible=True, path=["A", "C"], weight_km=150, modulation="QPSK", slots_needed=10, congestion=0.3),
        ]

    def test_load_pytorch_model(self, tmp_path, mock_pytorch_model):
        """Test PyTorch model loading."""
        import torch
        model_path = tmp_path / "model.pt"
        torch.save(mock_pytorch_model, model_path)

        from fusion.policies.ml_policy import MLControlPolicy
        policy = MLControlPolicy(str(model_path), model_type="pytorch")
        assert policy.model is not None

    def test_select_action_applies_mask(self, sample_options, mock_request, mocker):
        """Test that action masking is applied."""
        from fusion.policies.ml_policy import MLControlPolicy

        # Make option 1 infeasible
        sample_options[1] = sample_options[1]._replace(is_feasible=False)

        # Mock model to prefer action 1 (infeasible)
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()

        with patch.object(MLControlPolicy, '_load_model', return_value=mock_model):
            with patch.object(MLControlPolicy, '_predict_pytorch', return_value=1):
                policy = MLControlPolicy("dummy.pt", model_type="pytorch")
                # Policy should fallback since action 1 is infeasible
                action = policy.select_action(mock_request, sample_options, MagicMock())
                # Should return 0 (only feasible option) via fallback
                assert action == 0

    def test_fallback_on_error(self, sample_options, mock_request, mocker):
        """Test fallback policy on model error."""
        from fusion.policies.ml_policy import MLControlPolicy
        from fusion.policies.heuristic_policy import FirstFeasiblePolicy

        with patch.object(MLControlPolicy, '_load_model', return_value=MagicMock()):
            with patch.object(MLControlPolicy, '_predict', side_effect=Exception("Model error")):
                policy = MLControlPolicy("dummy.pt", fallback_policy=FirstFeasiblePolicy())
                action = policy.select_action(mock_request, sample_options, MagicMock())
                # Should use fallback
                assert action == 0  # First feasible
```

### 4. Protection Pipeline Tests

**File**: `fusion/tests/pipelines/test_protection_pipeline.py`

```python
import pytest
import networkx as nx
from fusion.pipelines.protection_pipeline import ProtectionPipeline


@pytest.fixture
def diamond_topology():
    """Create diamond topology with multiple disjoint paths."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=100)
    G.add_edge("A", "C", weight=100)
    G.add_edge("B", "D", weight=100)
    G.add_edge("C", "D", weight=100)
    return G


@pytest.fixture
def linear_topology():
    """Create linear topology (no disjoint paths)."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=100)
    G.add_edge("B", "C", weight=100)
    return G


@pytest.fixture
def mock_network_state(diamond_topology, mocker):
    """Create mock network state."""
    state = mocker.MagicMock()
    state.topology = diamond_topology
    return state


@pytest.fixture
def mock_config(mocker):
    """Create mock config."""
    config = mocker.MagicMock()
    config.protection_disjointness = "link"
    config.protection_switchover_ms = 50
    config.band_list = ["c"]
    config.cores_per_link = 1
    return config


class TestProtectionPipelineRouting:
    """Tests for protection route finding."""

    def test_finds_disjoint_paths_in_diamond(self, mock_config, mock_network_state):
        pipeline = ProtectionPipeline(mock_config)
        result = pipeline.find_protected_routes("A", "D", 100, mock_network_state)

        assert len(result.paths) > 0
        assert len(result.backup_paths) == len(result.paths)

    def test_no_protection_in_linear_topology(self, mock_config, linear_topology, mocker):
        state = mocker.MagicMock()
        state.topology = linear_topology

        pipeline = ProtectionPipeline(mock_config)
        result = pipeline.find_protected_routes("A", "C", 100, state)

        assert len(result.paths) == 0

    def test_paths_are_link_disjoint(self, mock_config, mock_network_state):
        mock_config.protection_disjointness = "link"
        pipeline = ProtectionPipeline(mock_config)
        result = pipeline.find_protected_routes("A", "D", 100, mock_network_state)

        if result.paths:
            primary = result.paths[0]
            backup = result.backup_paths[0]

            primary_links = set()
            for i in range(len(primary) - 1):
                primary_links.add((primary[i], primary[i+1]))
                primary_links.add((primary[i+1], primary[i]))

            backup_links = set()
            for i in range(len(backup) - 1):
                backup_links.add((backup[i], backup[i+1]))
                backup_links.add((backup[i+1], backup[i]))

            assert primary_links.isdisjoint(backup_links)


class TestProtectionPipelineFailure:
    """Tests for failure handling."""

    def test_switchover_on_primary_failure(self, mock_config, mocker):
        from fusion.domain.lightpath import Lightpath

        state = mocker.MagicMock()
        lp = Lightpath(
            lightpath_id=1,
            path=["A", "B", "D"],
            backup_path=["A", "C", "D"],
            is_protected=True,
            active_path="primary",
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            total_bandwidth_gbps=100,
            remaining_bandwidth_gbps=100,
            path_weight_km=200,
        )
        state.get_lightpaths_on_link.return_value = [lp]

        pipeline = ProtectionPipeline(mock_config)
        actions = pipeline.handle_failure(("A", "B"), state)

        assert len(actions) == 1
        assert actions[0]["action"] == "switchover"
        assert lp.active_path == "backup"

    def test_no_switchover_on_unaffected_link(self, mock_config, mocker):
        from fusion.domain.lightpath import Lightpath

        state = mocker.MagicMock()
        lp = Lightpath(
            lightpath_id=1,
            path=["A", "B", "D"],
            backup_path=["A", "C", "D"],
            is_protected=True,
            active_path="primary",
            start_slot=0,
            end_slot=10,
            core=0,
            band="c",
            modulation="QPSK",
            total_bandwidth_gbps=100,
            remaining_bandwidth_gbps=100,
            path_weight_km=200,
        )
        state.get_lightpaths_on_link.return_value = []  # No affected lightpaths

        pipeline = ProtectionPipeline(mock_config)
        actions = pipeline.handle_failure(("X", "Y"), state)

        assert len(actions) == 0
```

---

## Integration Tests

### Policy Simulation Integration

**File**: `tests/integration/test_policy_simulation.py`

```python
import pytest
from fusion.domain.config import SimulationConfig
from fusion.core.pipeline_factory import PipelineFactory
from fusion.core.orchestrator import SDNOrchestrator
from fusion.policies.heuristic_policy import (
    FirstFeasiblePolicy,
    ShortestFeasiblePolicy,
    LeastCongestedPolicy,
)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return SimulationConfig(
        network_name="test",
        cores_per_link=1,
        band_list=("c",),
        band_slots={"c": 320},
        guard_slots=1,
        num_requests=100,
        erlang=100,
        holding_time=1.0,
        route_method="k_shortest_path",
        k_paths=3,
        allocation_method="first_fit",
        grooming_enabled=False,
        slicing_enabled=False,
        max_slices=1,
        snr_enabled=False,
        snr_type=None,
        snr_recheck=False,
        can_partially_serve=False,
        modulation_formats={},
        mod_per_bw={},
        snr_thresholds={},
        policy_type="first_feasible",
    )


class TestPolicySimulation:
    """Integration tests for policy-driven simulation."""

    @pytest.mark.parametrize("policy_class", [
        FirstFeasiblePolicy,
        ShortestFeasiblePolicy,
        LeastCongestedPolicy,
    ])
    def test_simulation_runs_with_policy(self, test_config, policy_class):
        """Test that simulation completes with each policy type."""
        from fusion.domain.network_state import NetworkState
        import networkx as nx

        # Create simple test topology
        topology = nx.Graph()
        topology.add_edge("A", "B", weight=100)
        topology.add_edge("B", "C", weight=100)
        topology.add_edge("A", "C", weight=150)

        network_state = NetworkState(topology, test_config)
        pipelines = PipelineFactory.create_pipelines(test_config)
        policy = policy_class()
        orchestrator = SDNOrchestrator(test_config, pipelines, policy=policy)

        # Process a few requests
        from fusion.domain.request import Request, RequestStatus
        for i in range(10):
            request = Request(
                request_id=i,
                source="A",
                destination="C",
                bandwidth_gbps=50,
                holding_time=1.0,
                arrival_time=float(i),
                status=RequestStatus.PENDING,
            )
            result = orchestrator.handle_arrival_with_policy(request, network_state)
            # Some may block, but should not error
            assert result is not None
```

### Protection Simulation Integration

**File**: `tests/integration/test_protection_simulation.py`

```python
import pytest


class TestProtectionSimulation:
    """Integration tests for protection functionality."""

    def test_protection_allocates_on_both_paths(self, protection_config):
        """Test that protection allocates spectrum on both paths."""
        # ... implementation
        pass

    def test_protection_switchover_on_failure(self, protection_config):
        """Test that failure triggers switchover."""
        # ... implementation
        pass

    def test_no_protection_when_insufficient_paths(self, linear_topology_config):
        """Test blocking when disjoint paths unavailable."""
        # ... implementation
        pass
```

---

## Test Commands

```bash
# Run all Phase 5 tests
pytest fusion/tests/interfaces/test_control_policy.py -v
pytest fusion/tests/policies/ -v
pytest fusion/tests/pipelines/test_protection_pipeline.py -v
pytest tests/integration/test_policy_simulation.py -v
pytest tests/integration/test_protection_simulation.py -v

# Run with coverage
pytest fusion/tests/policies/ --cov=fusion/policies --cov-report=html

# Run specific test class
pytest fusion/tests/policies/test_heuristic_policy.py::TestFirstFeasiblePolicy -v

# Run comparison test
python tests/run_comparison.py --policy first_feasible
```

---

## Coverage Targets

| Module | Target Coverage |
|--------|-----------------|
| `fusion/interfaces/control_policy.py` | 100% |
| `fusion/policies/heuristic_policy.py` | 95% |
| `fusion/policies/ml_policy.py` | 85% |
| `fusion/policies/policy_factory.py` | 90% |
| `fusion/pipelines/protection_pipeline.py` | 90% |

---

## Regression Testing

Ensure Phase 4 functionality is preserved:

```bash
# RL environment tests
pytest fusion/tests/modules/rl/gymnasium_envs/test_unified_sim_env.py -v

# Adapter tests
pytest fusion/tests/modules/rl/test_env_adapter.py -v

# Full comparison
python tests/run_comparison.py --all
```
