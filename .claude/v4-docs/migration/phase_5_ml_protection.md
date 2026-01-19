# Phase 5: ML Control + Protection Integration

## Overview

Phase 5 introduces the `ControlPolicy` abstraction and integrates 1+1 protection into the pipeline architecture. This phase unifies RL, ML, and heuristic decision-making under a common interface while adding network survivability capabilities.

## Prerequisites

Before starting Phase 5:

- [ ] Phase 4 complete (RL integration verified)
- [ ] UnifiedSimEnv functional
- [ ] RLSimulationAdapter tested
- [ ] `run_comparison.py` passes with RL path

## Objectives

1. **ControlPolicy Protocol**: Define common interface for all policy types
2. **Heuristic Policies**: Implement baseline policies (first-fit, shortest, etc.)
3. **ML Policy Support**: Enable pre-trained model deployment
4. **Protection Pipeline**: Integrate 1+1 dedicated protection
5. **Policy Factory**: Create policies from configuration

## Micro-Phases

### P5.1: ControlPolicy Protocol

**Goal**: Define the control policy abstraction.

**Files Created**:
```
fusion/interfaces/control_policy.py
fusion/policies/__init__.py
fusion/policies/heuristic_policy.py
```

**Implementation**:

```python
# fusion/interfaces/control_policy.py

from typing import Protocol, runtime_checkable
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.modules.rl.env_adapter import PathOption


@runtime_checkable
class ControlPolicy(Protocol):
    """Protocol for control policies."""

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select action (path index) for the request."""
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """Optional: Update policy based on experience."""
        ...
```

**Verification**:
```bash
pytest fusion/tests/interfaces/test_control_policy.py -v
mypy fusion/interfaces/control_policy.py
```

### P5.2: Heuristic Policies

**Goal**: Implement baseline heuristic policies.

**Files Created**:
```
fusion/policies/heuristic_policy.py
fusion/policies/policy_factory.py
```

**Policies Implemented**:
- `FirstFeasiblePolicy`
- `ShortestFeasiblePolicy`
- `LeastCongestedPolicy`
- `RandomFeasiblePolicy`
- `LoadBalancedPolicy`

**Implementation**:

```python
# fusion/policies/heuristic_policy.py

from abc import ABC, abstractmethod
from fusion.interfaces.control_policy import ControlPolicy


class HeuristicPolicy(ControlPolicy, ABC):
    """Base class for heuristic policies."""

    @abstractmethod
    def select_action(self, request, options, network_state) -> int:
        ...

    def update(self, request, action, reward) -> None:
        pass  # Heuristics don't learn


class FirstFeasiblePolicy(HeuristicPolicy):
    def select_action(self, request, options, network_state) -> int:
        for opt in options:
            if opt.is_feasible:
                return opt.path_index
        return -1


class ShortestFeasiblePolicy(HeuristicPolicy):
    def select_action(self, request, options, network_state) -> int:
        feasible = [o for o in options if o.is_feasible]
        if not feasible:
            return -1
        return min(feasible, key=lambda o: o.weight_km).path_index


class LeastCongestedPolicy(HeuristicPolicy):
    def select_action(self, request, options, network_state) -> int:
        feasible = [o for o in options if o.is_feasible]
        if not feasible:
            return -1
        return min(feasible, key=lambda o: o.congestion).path_index
```

**Verification**:
```bash
pytest fusion/tests/policies/test_heuristic_policy.py -v
```

### P5.3: ML Policy Support

**Goal**: Enable deployment of pre-trained ML models.

**Files Created**:
```
fusion/policies/ml_policy.py
```

**Implementation**:

```python
# fusion/policies/ml_policy.py

import torch
import numpy as np
from fusion.interfaces.control_policy import ControlPolicy


class MLControlPolicy(ControlPolicy):
    """ML-based control policy using pre-trained model."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def select_action(self, request, options, network_state) -> int:
        features = self._build_features(request, options, network_state)

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            logits = self.model(features_tensor)

            # Apply action mask
            mask = torch.tensor([opt.is_feasible for opt in options])
            logits[:, ~mask] = float('-inf')

            action = logits.argmax(dim=1).item()

        return action if options[action].is_feasible else -1

    def _build_features(self, request, options, network_state) -> np.ndarray:
        features = []
        for opt in options:
            features.extend([
                opt.weight_km / 10000.0,
                opt.congestion,
                1.0 if opt.is_feasible else 0.0,
                (opt.slots_needed or 0) / 100.0,
            ])
        return np.array(features, dtype=np.float32)

    def update(self, request, action, reward) -> None:
        pass  # Pre-trained, no online updates
```

**Verification**:
```bash
pytest fusion/tests/policies/test_ml_policy.py -v
```

### P5.4: Protection Pipeline

**Goal**: Integrate 1+1 dedicated path protection.

**Files Created**:
```
fusion/pipelines/protection_pipeline.py
```

**Files Modified**:
```
fusion/domain/lightpath.py          # Add protection fields
fusion/domain/network_state.py      # Add create_protected_lightpath
fusion/domain/results.py            # Add protection result fields
fusion/core/pipeline_factory.py     # Add protection pipeline creation
```

**Lightpath Extension**:

```python
# In fusion/domain/lightpath.py

@dataclass
class Lightpath:
    # ... existing fields ...

    # Protection fields
    backup_path: Optional[list[str]] = None
    is_protected: bool = False
    active_path: str = "primary"  # "primary" or "backup"

    @property
    def current_path(self) -> list[str]:
        if self.is_protected and self.active_path == "backup":
            return self.backup_path
        return self.path
```

**RouteResult Extension**:

```python
# In fusion/domain/results.py

@dataclass(frozen=True)
class RouteResult:
    # ... existing fields ...

    # Protection fields
    backup_paths: Optional[list[list[str]]] = None
    backup_weights_km: Optional[list[float]] = None
    backup_modulations: Optional[list[list[str | None]]] = None
```

**ProtectionPipeline**:

```python
# fusion/pipelines/protection_pipeline.py

import networkx as nx
from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.results import RouteResult


class ProtectionPipeline:
    """Pipeline for 1+1 dedicated path protection."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.disjointness = config.protection_disjointness
        self.switchover_time_ms = config.protection_switchover_ms

    def find_protected_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> RouteResult:
        topology = network_state.topology

        try:
            if self.disjointness == "link":
                disjoint_paths = list(nx.edge_disjoint_paths(
                    topology, source, destination
                ))
            else:
                disjoint_paths = list(nx.node_disjoint_paths(
                    topology, source, destination
                ))
        except nx.NetworkXNoPath:
            return RouteResult(paths=[], modulations=[], weights_km=[])

        if len(disjoint_paths) < 2:
            return RouteResult(paths=[], modulations=[], weights_km=[])

        # Pair up disjoint paths
        paths, backup_paths = [], []
        modulations, backup_modulations = [], []
        weights, backup_weights = [], []

        for i in range(0, len(disjoint_paths) - 1, 2):
            primary = list(disjoint_paths[i])
            backup = list(disjoint_paths[i + 1])

            primary_mods = self._get_valid_modulations(primary, bandwidth_gbps, network_state)
            backup_mods = self._get_valid_modulations(backup, bandwidth_gbps, network_state)

            if not any(primary_mods) or not any(backup_mods):
                continue

            paths.append(primary)
            backup_paths.append(backup)
            modulations.append(primary_mods)
            backup_modulations.append(backup_mods)
            weights.append(self._compute_path_weight(primary, network_state))
            backup_weights.append(self._compute_path_weight(backup, network_state))

        return RouteResult(
            paths=paths,
            modulations=modulations,
            weights_km=weights,
            backup_paths=backup_paths,
            backup_modulations=backup_modulations,
            backup_weights_km=backup_weights,
            strategy_name="1plus1_protection",
        )

    def handle_failure(
        self,
        failed_link: tuple[str, str],
        network_state: NetworkState,
    ) -> list[dict]:
        affected = network_state.get_lightpaths_on_link(failed_link)
        switchovers = []

        for lp in affected:
            if lp.is_protected and lp.active_path == "primary":
                if self._path_contains_link(lp.path, failed_link):
                    lp.active_path = "backup"
                    switchovers.append({
                        "lightpath_id": lp.lightpath_id,
                        "action": "switchover",
                        "latency_ms": self.switchover_time_ms,
                    })

        return switchovers
```

**Verification**:
```bash
pytest fusion/tests/pipelines/test_protection_pipeline.py -v
pytest fusion/tests/domain/test_lightpath.py::test_protection_fields -v
```

### P5.5: Orchestrator Integration

**Goal**: Connect policies to orchestrator.

**Files Modified**:
```
fusion/core/orchestrator.py
fusion/core/pipeline_factory.py
```

**SDNOrchestrator Extension**:

```python
# In fusion/core/orchestrator.py

class SDNOrchestrator:
    def __init__(
        self,
        config: SimulationConfig,
        pipelines: PipelineSet,
        policy: Optional[ControlPolicy] = None,
    ):
        self.config = config
        self.routing = pipelines.routing
        self.spectrum = pipelines.spectrum
        # ... other pipelines

        # Policy defaults to FirstFeasiblePolicy
        self.policy = policy or FirstFeasiblePolicy()
        self._adapter = RLSimulationAdapter(config, self)

    def handle_arrival_with_policy(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> AllocationResult:
        """Handle arrival using configured policy."""
        options = self._adapter.get_path_options(
            request.source,
            request.destination,
            request.bandwidth_gbps,
            network_state,
        )

        if not any(opt.is_feasible for opt in options):
            return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)

        action = self.policy.select_action(request, options, network_state)

        if action < 0 or not options[action].is_feasible:
            return AllocationResult(success=False, block_reason=BlockReason.INVALID_ACTION)

        result, reward = self._adapter.apply_action(action, request, options, network_state)
        self.policy.update(request, action, reward)

        return result
```

**PipelineFactory Extension**:

```python
# In fusion/core/pipeline_factory.py

from fusion.policies.heuristic_policy import FirstFeasiblePolicy
from fusion.policies.ml_policy import MLControlPolicy


class PipelineFactory:
    @staticmethod
    def create_policy(config: SimulationConfig) -> ControlPolicy:
        policy_type = config.policy_type

        if policy_type == "first_feasible":
            return FirstFeasiblePolicy()
        elif policy_type == "shortest_feasible":
            return ShortestFeasiblePolicy()
        elif policy_type == "least_congested":
            return LeastCongestedPolicy()
        elif policy_type == "ml":
            return MLControlPolicy(config.policy_model_path, config.device)
        elif policy_type == "rl":
            model = load_rl_model(config.policy_model_path)
            return RLPolicy(model)
        else:
            return FirstFeasiblePolicy()

    @staticmethod
    def create_routing(config: SimulationConfig) -> RoutingPipeline:
        if config.route_method == "1plus1_protection":
            return ProtectionPipeline(config)
        elif config.route_method == "k_shortest_path":
            return KSPRoutingPipeline(config)
        # ... other strategies
```

**Verification**:
```bash
pytest fusion/tests/core/test_orchestrator.py::test_policy_integration -v
pytest fusion/tests/core/test_pipeline_factory.py::test_policy_creation -v
```

### P5.6: Configuration Extension

**Goal**: Add Phase 5 configuration options.

**Files Modified**:
```
fusion/configs/templates/default.ini
fusion/cli/parameters/policy.py     # New file
```

**Configuration Additions**:

```ini
[policy]
type = first_feasible           ; first_feasible, shortest_feasible, least_congested, ml, rl
model_path =                    ; Path to model for ml/rl policies
device = cpu                    ; cpu, cuda

[protection]
enabled = false
disjointness = link             ; link, node
switchover_time_ms = 50

[heuristic]
load_balance_alpha = 0.5        ; For load_balanced policy
random_seed = 42                ; For random_feasible policy
```

## Files Created Summary

| File | Purpose |
|------|---------|
| `fusion/interfaces/control_policy.py` | ControlPolicy protocol |
| `fusion/policies/__init__.py` | Policies module |
| `fusion/policies/heuristic_policy.py` | Heuristic policies |
| `fusion/policies/ml_policy.py` | ML policy |
| `fusion/policies/policy_factory.py` | Policy factory |
| `fusion/pipelines/protection_pipeline.py` | Protection pipeline |
| `fusion/cli/parameters/policy.py` | CLI parameters |

## Files Modified Summary

| File | Changes |
|------|---------|
| `fusion/domain/lightpath.py` | Add protection fields |
| `fusion/domain/network_state.py` | Add `create_protected_lightpath`, `get_lightpaths_on_link` |
| `fusion/domain/results.py` | Add backup path fields to RouteResult |
| `fusion/core/orchestrator.py` | Add policy integration |
| `fusion/core/pipeline_factory.py` | Add policy and protection creation |
| `fusion/configs/templates/default.ini` | Add policy and protection sections |

## Verification Commands

```bash
# Run all Phase 5 tests
pytest fusion/tests/policies/ -v
pytest fusion/tests/pipelines/test_protection_pipeline.py -v
pytest fusion/tests/core/test_orchestrator.py::TestPolicyIntegration -v

# Type checking
mypy fusion/interfaces/control_policy.py
mypy fusion/policies/
mypy fusion/pipelines/protection_pipeline.py

# Lint checking
ruff check fusion/interfaces/control_policy.py
ruff check fusion/policies/
ruff check fusion/pipelines/protection_pipeline.py

# Integration test
pytest tests/integration/test_policy_simulation.py -v

# Comparison test (must pass)
python tests/run_comparison.py --policy first_feasible
```

## Exit Criteria

- [ ] All heuristic policies implemented and tested
- [ ] MLControlPolicy functional with test model
- [ ] Protection pipeline finds disjoint paths correctly
- [ ] Protected spectrum allocation works on both paths
- [ ] Failure switchover mechanism functional
- [ ] Policy factory creates all policy types
- [ ] Orchestrator integrates with policies
- [ ] Configuration options documented
- [ ] `run_comparison.py` passes
- [ ] No regressions in Phase 4 functionality

## Rollback Plan

If Phase 5 causes issues:

1. **Policy issues**: Set `policy_type = first_feasible` (default heuristic)
2. **Protection issues**: Set `protection_enabled = false`
3. **Full rollback**: Revert to Phase 4 codebase (git checkout)

## Next Phase

Phase 6 (Legacy Removal) will:
- Remove legacy compatibility properties from NetworkState
- Remove deprecated SDNController
- Clean up Props classes
- Update documentation
