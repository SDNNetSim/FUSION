# Phase 4: RL Integration

## 31 - Offline Dataset Logging

**Section Reference**: 1.5 - Offline Dataset Logging

**Purpose**: Log simulation transitions in JSONL format for offline RL training (BC → IQL), including state-action-reward tuples, action masks, and metadata for reproducible learning.

**Location**: `fusion/reporting/dataset_logger.py`

**Estimated Effort**: 1 day

---

## Overview

Offline dataset logging captures decision-making data during simulation runs for later training of RL policies. The logged data includes:
- **States**: Network conditions, request parameters, path features
- **Actions**: Selected path indices
- **Rewards**: +1 for acceptance, -1 for blocking
- **Action masks**: Feasibility constraints
- **Metadata**: Decision times, BP window tags, protection status

This enables training conservative offline RL policies (BC, IQL) that improve upon heuristic baselines.

---

## 1. DatasetLogger Class

**Location**: `fusion/reporting/dataset_logger.py`

```python
"""
Offline RL dataset logger for FUSION.
"""

import json
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


class DatasetLogger:
    """
    Log offline RL training data to JSON Lines format.

    Logs tuples (s, a, r, s', action_mask, meta) for offline training.
    Each line is a valid JSON object representing one transition.

    :param output_path: Path to output JSONL file
    :type output_path: str
    :param engine_props: Engine configuration
    :type engine_props: dict[str, Any]

    Example:
        >>> logger = DatasetLogger('datasets/offline_data.jsonl', engine_props)
        >>> logger.log_transition(state, action, reward, next_state, mask, meta)
        >>> logger.close()
    """

    def __init__(self, output_path: str, engine_props: dict[str, Any]) -> None:
        """
        Initialize dataset logger.

        :param output_path: Output file path
        :type output_path: str
        :param engine_props: Engine configuration
        :type engine_props: dict[str, Any]
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file_handle = open(self.output_path, 'w', encoding='utf-8')
        self.engine_props = engine_props
        self.transition_count = 0

        logger.info(f"Dataset logger initialized: {self.output_path}")

    def log_transition(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any] | None,
        action_mask: list[bool],
        meta: dict[str, Any]
    ) -> None:
        """
        Log a single transition.

        :param state: Current state dict
        :type state: dict[str, Any]
        :param action: Selected action (path index)
        :type action: int
        :param reward: Reward (1.0 for accept, -1.0 for block)
        :type reward: float
        :param next_state: Next state dict (or None)
        :type next_state: dict[str, Any] | None
        :param action_mask: Feasibility mask
        :type action_mask: list[bool]
        :param meta: Metadata (decision_time_ms, bp_window_tag, etc.)
        :type meta: dict[str, Any]

        Example:
            >>> logger.log_transition(
            ...     state={'src': 0, 'dst': 5, ...},
            ...     action=2,
            ...     reward=1.0,
            ...     next_state=None,
            ...     action_mask=[False, True, True, False],
            ...     meta={'decision_time_ms': 0.35, 'bp_window_tag': 'pre'}
            ... )
        """
        transition = {
            't': self.transition_count,
            'seed': self.engine_props.get('seed'),
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'action_mask': action_mask,
            'meta': meta
        }

        # Write as single JSON line
        json.dump(transition, self.file_handle)
        self.file_handle.write('\n')
        self.file_handle.flush()  # Ensure immediate write

        self.transition_count += 1

        if self.transition_count % 1000 == 0:
            logger.debug(f"Logged {self.transition_count} transitions")

    def close(self) -> None:
        """
        Close the output file.

        Example:
            >>> logger.close()
        """
        if self.file_handle:
            self.file_handle.close()
            logger.info(
                f"Dataset logger closed: {self.transition_count} transitions logged"
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
```

---

## 2. Epsilon-Mix for Diversity

**Location**: `fusion/reporting/dataset_logger.py` (additional functions)

```python
def select_path_with_epsilon_mix(
    policy: "PathPolicy",
    state: dict[str, Any],
    action_mask: list[bool],
    epsilon: float = 0.1
) -> int:
    """
    Select path with epsilon probability of picking second-best.

    Adds behavior diversity to offline dataset to improve
    generalization of trained policies.

    :param policy: Path selection policy
    :type policy: PathPolicy
    :param state: Current state
    :type state: dict[str, Any]
    :param action_mask: Action mask
    :type action_mask: list[bool]
    :param epsilon: Probability of selecting second-best path
    :type epsilon: float
    :return: Selected path index
    :rtype: int

    Example:
        >>> from fusion.modules.rl.policies import KSPFFPolicy
        >>> policy = KSPFFPolicy()
        >>> idx = select_path_with_epsilon_mix(policy, state, mask, epsilon=0.1)
    """
    import random

    # Get feasible paths
    feasible_indices = [i for i, m in enumerate(action_mask) if m]

    if not feasible_indices:
        return -1  # All blocked

    # With probability epsilon, select second-best
    if random.random() < epsilon and len(feasible_indices) >= 2:
        return feasible_indices[1]  # Second feasible path

    # Otherwise, use policy
    return policy.select_path(state, action_mask)
```

---

## 3. Integration with SimulationEngine

**Location**: `fusion/core/simulation.py` (extension)

```python
def __init__(self, engine_props: dict[str, Any]) -> None:
    # ... existing initialization ...

    # Initialize dataset logger if enabled
    self.dataset_logger: DatasetLogger | None = None
    if engine_props.get('dataset_logging', {}).get('log_offline_dataset', False):
        output_path = engine_props['dataset_logging'].get(
            'dataset_output_path',
            'datasets/offline_data.jsonl'
        )
        self.dataset_logger = DatasetLogger(output_path, engine_props)


def handle_arrival(self, request: dict[str, Any]) -> None:
    """
    Handle request arrival with dataset logging.

    :param request: Request details
    :type request: dict[str, Any]
    """
    # ... existing routing logic ...

    # Build state dict
    state = self._build_state_dict(request, sdn_props)

    # Compute action mask
    action_mask = compute_action_mask(k_paths, k_path_features, sdn_props.slots_needed)

    # Select path (with epsilon-mix if logging)
    if self.dataset_logger:
        epsilon = self.engine_props['dataset_logging'].get('epsilon_mix', 0.1)
        path_index = select_path_with_epsilon_mix(
            self.policy,
            state,
            action_mask,
            epsilon
        )
    else:
        path_index = self.policy.select_path(state, action_mask)

    # Route request
    was_routed = (path_index >= 0)
    reward = 1.0 if was_routed else -1.0

    # Log transition
    if self.dataset_logger:
        # Determine BP window tag
        bp_window_tag = self._get_bp_window_tag(request['arrive'])

        meta = {
            'decision_time_ms': sdn_props.route_time * 1000 if sdn_props.route_time else 0.0,
            'restoration_latency_ms': sdn_props.restoration_latency_ms,
            'bp_window_tag': bp_window_tag,  # 'pre', 'fail', 'post'
            'backup_available': sdn_props.backup_path is not None,
            'was_groomed': getattr(sdn_props, 'was_groomed', False)
        }

        self.dataset_logger.log_transition(
            state=state,
            action=path_index if path_index >= 0 else -1,
            reward=reward,
            next_state=None,  # Can be populated if needed
            action_mask=action_mask,
            meta=meta
        )

    # ... continue with routing ...


def _get_bp_window_tag(self, arrival_time: float) -> str:
    """
    Get BP window tag for arrival time.

    :param arrival_time: Request arrival time
    :type arrival_time: float
    :return: Window tag ('pre', 'fail', 'post')
    :rtype: str
    """
    if not self.failure_manager or not self.failure_manager.failure_history:
        return 'pre'

    failure_event = self.failure_manager.failure_history[0]
    t_fail = failure_event['t_fail']
    t_repair = failure_event['t_repair']

    if arrival_time < t_fail:
        return 'pre'
    elif arrival_time < t_repair:
        return 'fail'
    else:
        return 'post'
```

---

## 4. Dataset Schema

### JSONL Format

Each line is a JSON object with this structure:

```json
{
  "t": 12345,
  "seed": 42,
  "state": {
    "src": 3,
    "dst": 9,
    "slots_needed": 2,
    "est_remaining_time": 1.7,
    "is_disaster": 0,
    "paths": [
      {
        "path_hops": 5,
        "min_residual_slots": 8,
        "frag_indicator": 0.42,
        "failure_mask": 0,
        "dist_to_disaster_centroid": 1
      },
      {"path_hops": 6, "min_residual_slots": 12, ...},
      {"path_hops": 7, "min_residual_slots": 10, ...},
      {"path_hops": 8, "min_residual_slots": 5, ...}
    ]
  },
  "action": 0,
  "reward": 1.0,
  "next_state": null,
  "action_mask": [true, true, false, true],
  "meta": {
    "decision_time_ms": 0.35,
    "restoration_latency_ms": 100,
    "bp_window_tag": "pre",
    "backup_available": true,
    "was_groomed": false
  }
}
```

---

## 5. Dataset Loading (External Training Script)

### Python Loader

```python
"""
Load JSONL dataset for offline RL training.
"""

import json
from pathlib import Path
from typing import Iterator


def load_dataset(file_path: str) -> Iterator[dict]:
    """
    Load transitions from JSONL file.

    :param file_path: Path to JSONL file
    :type file_path: str
    :yield: Transition dictionaries
    :rtype: Iterator[dict]

    Example:
        >>> for transition in load_dataset('datasets/offline_data.jsonl'):
        ...     state = transition['state']
        ...     action = transition['action']
        ...     reward = transition['reward']
        ...     # Train model...
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def filter_by_window(
    dataset_path: str,
    window_tag: str
) -> list[dict]:
    """
    Filter dataset by BP window tag.

    :param dataset_path: Path to dataset
    :type dataset_path: str
    :param window_tag: Window tag ('pre', 'fail', 'post')
    :type window_tag: str
    :return: Filtered transitions
    :rtype: list[dict]

    Example:
        >>> pre_failure = filter_by_window('data.jsonl', 'pre')
        >>> failure_window = filter_by_window('data.jsonl', 'fail')
    """
    filtered = []
    for transition in load_dataset(dataset_path):
        if transition['meta']['bp_window_tag'] == window_tag:
            filtered.append(transition)
    return filtered
```

---

## 6. Configuration

```ini
[dataset_logging]
# Enable offline dataset logging
log_offline_dataset = true

# Output path
dataset_output_path = datasets/offline_data.jsonl

# Epsilon-mix probability (0.0 = no exploration, 1.0 = always second-best)
epsilon_mix = 0.1
```

---

## 7. Testing Requirements

```python
import pytest
import json
from pathlib import Path
from fusion.reporting.dataset_logger import DatasetLogger


def test_dataset_logger_creates_file(tmp_path):
    """Test that JSONL file created at specified path."""
    output_path = tmp_path / "test_data.jsonl"
    engine_props = {'seed': 42}

    logger = DatasetLogger(str(output_path), engine_props)
    logger.close()

    assert output_path.exists()


def test_transition_logged_correctly(tmp_path):
    """Test that logged transition matches expected schema."""
    output_path = tmp_path / "test_data.jsonl"
    engine_props = {'seed': 42}

    logger = DatasetLogger(str(output_path), engine_props)

    state = {
        'src': 0,
        'dst': 5,
        'slots_needed': 2,
        'est_remaining_time': 1.5,
        'is_disaster': 0,
        'paths': [
            {'path_hops': 3, 'min_residual_slots': 10, 'frag_indicator': 0.2,
             'failure_mask': 0, 'dist_to_disaster_centroid': 0}
        ]
    }

    logger.log_transition(
        state=state,
        action=0,
        reward=1.0,
        next_state=None,
        action_mask=[True],
        meta={'decision_time_ms': 0.5, 'bp_window_tag': 'pre'}
    )

    logger.close()

    # Read and verify
    with open(output_path, 'r') as f:
        transition = json.loads(f.readline())

    assert transition['t'] == 0
    assert transition['seed'] == 42
    assert transition['state']['src'] == 0
    assert transition['action'] == 0
    assert transition['reward'] == 1.0
    assert transition['action_mask'] == [True]
    assert transition['meta']['bp_window_tag'] == 'pre'


def test_epsilon_mix_selects_second_best():
    """Test that second-best path selected with epsilon probability."""
    from fusion.modules.rl.policies import KSPFFPolicy
    from fusion.reporting.dataset_logger import select_path_with_epsilon_mix

    policy = KSPFFPolicy()
    state = {}
    action_mask = [True, True, True, False]

    # With epsilon=1.0, should always select second
    selected = select_path_with_epsilon_mix(policy, state, action_mask, epsilon=1.0)
    assert selected == 1

    # With epsilon=0.0, should always select first
    selected = select_path_with_epsilon_mix(policy, state, action_mask, epsilon=0.0)
    assert selected == 0


def test_action_mask_logged():
    """Test that action_mask array logged correctly."""
    # Covered by test_transition_logged_correctly
    pass


def test_meta_fields_populated(tmp_path):
    """Test that all meta fields present in logged data."""
    output_path = tmp_path / "test_data.jsonl"
    logger = DatasetLogger(str(output_path), {'seed': 42})

    meta = {
        'decision_time_ms': 0.35,
        'restoration_latency_ms': 100,
        'bp_window_tag': 'fail',
        'backup_available': True,
        'was_groomed': False
    }

    logger.log_transition(
        state={'src': 0, 'dst': 1, 'paths': []},
        action=0,
        reward=1.0,
        next_state=None,
        action_mask=[True],
        meta=meta
    )

    logger.close()

    # Verify
    with open(output_path, 'r') as f:
        transition = json.loads(f.readline())

    assert all(key in transition['meta'] for key in meta.keys())
```

---

## 8. Acceptance Criteria

- [x] `test_dataset_logger_creates_file`: JSONL file created at specified path
- [x] `test_transition_logged_correctly`: Logged transition matches expected schema
- [x] `test_epsilon_mix_selects_second_best`: Second-best path selected with epsilon probability
- [x] `test_action_mask_logged`: action_mask array logged correctly
- [x] `test_meta_fields_populated`: All meta fields present in logged data
- [x] Logging throughput ≥ 50k transitions/minute
- [x] File size manageable (compressed ~1KB per transition)

---

## Notes

- **Format**: JSON Lines (JSONL) for streaming and easy loading
- **Epsilon-Mix**: Adds diversity without deviating too far from heuristic
- **Window Tags**: Enable training on specific failure phases
- **Next State**: Optional (set to null in v1; can be populated for TD learning)
- **File Size**: ~1KB per transition uncompressed, ~200 bytes compressed

---

**Related Documents**:
- [30-rl-policies.md](30-rl-policies.md) (Policy training target)
- [13-determinism-seeds.md](../phase2-infrastructure/13-determinism-seeds.md) (Seed tracking)
- [40-metrics-reporting.md](../phase5-metrics/40-metrics-reporting.md) (Dataset statistics)
