# Phase 4: RL Integration

## 30 - RL Policy Integration (Offline Inference)

**Section Reference**: 1.4 - RL Policy Integration (Offline Inference)

**Purpose**: Integrate offline-trained RL policies (BC, IQL) for path selection with action masking and heuristic fallback, enabling safe evaluation of learned policies under failure conditions.

**Location**: `fusion/modules/rl/policies/`

**Estimated Effort**: 1.5 days

---

## Overview

This module provides a unified interface for path selection policies, supporting:
- **Heuristic baselines**: KSP-FF, 1+1 protection
- **Behavior Cloning (BC)**: Imitates heuristic behavior
- **Implicit Q-Learning (IQL)**: Conservative offline RL
- **Action masking**: Prevents selection of infeasible paths
- **Fallback mechanism**: Uses heuristic when all actions masked

The goal is safe, conservative RL deployment under network failures.

---

## Module Structure

```
fusion/modules/rl/policies/
├── __init__.py
├── README.md
├── base.py                  # PathPolicy interface
├── ksp_ff_policy.py        # Baseline KSP-FF policy
├── one_plus_one_policy.py  # Baseline 1+1 policy
├── bc_policy.py            # Behavior Cloning policy
├── iql_policy.py           # IQL policy
├── action_masking.py       # Action masking utilities
└── tests/
    ├── __init__.py
    ├── README.md
    ├── test_bc_policy.py
    ├── test_iql_policy.py
    ├── test_action_masking.py
    └── fixtures/
        ├── bc_model.pt      # Test BC model
        └── iql_model.pt     # Test IQL model
```

---

## 1. PathPolicy Interface

### Base Class

**Location**: `fusion/modules/rl/policies/base.py`

```python
"""
Abstract interface for path selection policies.
"""

from abc import ABC, abstractmethod
from typing import Any


class PathPolicy(ABC):
    """
    Abstract interface for path selection policies.

    All policies (heuristic and RL-based) must implement this interface
    to ensure consistent integration with the SDN controller.

    :raises AllPathsMaskedError: If all paths are infeasible
    """

    @abstractmethod
    def select_path(
        self,
        state: dict[str, Any],
        action_mask: list[bool]
    ) -> int:
        """
        Select a path index from K candidates.

        :param state: State dictionary with request and path features
        :type state: dict[str, Any]
        :param action_mask: Feasibility mask for K paths (True = feasible)
        :type action_mask: list[bool]
        :return: Selected path index (0 to K-1)
        :rtype: int
        :raises AllPathsMaskedError: If all paths are masked

        State format:
            {
                'src': int,
                'dst': int,
                'slots_needed': int,
                'est_remaining_time': float,
                'is_disaster': int (0 or 1),
                'paths': [
                    {
                        'path_hops': int,
                        'min_residual_slots': int,
                        'frag_indicator': float,
                        'failure_mask': int,
                        'dist_to_disaster_centroid': int
                    },
                    ...  # K paths
                ]
            }
        """
        pass

    def get_name(self) -> str:
        """
        Get policy name.

        :return: Policy name
        :rtype: str
        """
        return self.__class__.__name__


class AllPathsMaskedError(Exception):
    """Raised when all K paths are infeasible."""
    pass
```

---

## 2. Baseline Policies

### KSP-FF Policy

**Location**: `fusion/modules/rl/policies/ksp_ff_policy.py`

```python
"""
K-Shortest Path First-Fit policy (baseline).
"""

from typing import Any
from .base import PathPolicy, AllPathsMaskedError


class KSPFFPolicy(PathPolicy):
    """
    KSP-FF baseline: always select first feasible path.

    This is the standard heuristic baseline used in most
    optical network studies.

    Example:
        >>> policy = KSPFFPolicy()
        >>> action_mask = [False, True, True, False]
        >>> selected = policy.select_path(state, action_mask)
        >>> print(selected)
        1  # First feasible path
    """

    def select_path(
        self,
        state: dict[str, Any],
        action_mask: list[bool]
    ) -> int:
        """
        Select first unmasked path.

        :param state: Current state (not used by KSP-FF)
        :type state: dict[str, Any]
        :param action_mask: Feasibility mask
        :type action_mask: list[bool]
        :return: Index of first feasible path
        :rtype: int
        :raises AllPathsMaskedError: If all paths masked
        """
        for i, is_feasible in enumerate(action_mask):
            if is_feasible:
                return i

        raise AllPathsMaskedError("All K paths are infeasible")
```

### 1+1 Policy

**Location**: `fusion/modules/rl/policies/one_plus_one_policy.py`

```python
"""
1+1 protection policy (baseline).
"""

from typing import Any
from .base import PathPolicy, AllPathsMaskedError


class OnePlusOnePolicy(PathPolicy):
    """
    1+1 policy: use primary if feasible, else backup.

    Selects from pre-computed disjoint paths based on
    failure state.

    Example:
        >>> policy = OnePlusOnePolicy()
        >>> # Primary path masked (failed)
        >>> action_mask = [False, True]
        >>> selected = policy.select_path(state, action_mask)
        >>> print(selected)
        1  # Backup path
    """

    def select_path(
        self,
        state: dict[str, Any],
        action_mask: list[bool]
    ) -> int:
        """
        Select primary (index 0) if feasible, else backup (index 1).

        :param state: Current state
        :type state: dict[str, Any]
        :param action_mask: Feasibility mask
        :type action_mask: list[bool]
        :return: 0 (primary) or 1 (backup)
        :rtype: int
        :raises AllPathsMaskedError: If both paths masked
        """
        # Try primary first
        if action_mask[0]:
            return 0

        # Fall back to backup
        if len(action_mask) > 1 and action_mask[1]:
            return 1

        raise AllPathsMaskedError("Both primary and backup paths infeasible")
```

---

## 3. BC Policy

### Behavior Cloning

**Location**: `fusion/modules/rl/policies/bc_policy.py`

```python
"""
Behavior Cloning policy for path selection.
"""

from typing import Any
from pathlib import Path
import torch
import torch.nn as nn
from .base import PathPolicy, AllPathsMaskedError
import logging

logger = logging.getLogger(__name__)


class BCPolicy(PathPolicy):
    """
    Behavior Cloning policy for path selection.

    Trained to imitate heuristic (KSP-FF or 1+1) behavior using
    supervised learning on offline dataset.

    :param model_path: Path to saved PyTorch model (.pt file)
    :type model_path: str
    :param device: Torch device (cpu, cuda, mps)
    :type device: str

    Example:
        >>> policy = BCPolicy('models/bc_model.pt', device='cpu')
        >>> selected = policy.select_path(state, action_mask)
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """
        Initialize BC policy.

        :param model_path: Path to model file
        :type model_path: str
        :param device: Compute device
        :type device: str
        :raises FileNotFoundError: If model file doesn't exist
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"BC model not found: {model_path}")

        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded BC policy from {model_path} on {self.device}")

    def _load_model(self, model_path: Path) -> nn.Module:
        """
        Load pre-trained BC model.

        :param model_path: Path to model
        :type model_path: Path
        :return: Loaded model
        :rtype: nn.Module
        """
        try:
            model = torch.load(model_path, map_location=self.device)

            if isinstance(model, dict):
                # Model saved as state dict
                # Need to reconstruct model architecture
                model = self._build_model_architecture(model)

            return model

        except Exception as e:
            logger.error(f"Failed to load BC model: {e}")
            raise

    def _build_model_architecture(self, state_dict: dict) -> nn.Module:
        """
        Build model architecture from state dict.

        Default architecture: 3-layer MLP
        Input: state features (flattened)
        Output: K-way logits

        :param state_dict: Model state dictionary
        :type state_dict: dict
        :return: Model with loaded weights
        :rtype: nn.Module
        """
        # Infer dimensions from state dict
        input_dim = state_dict['fc1.weight'].shape[1]
        output_dim = state_dict['fc3.weight'].shape[0]

        # Build model
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        model.load_state_dict(state_dict)
        return model

    def _state_to_tensor(self, state: dict[str, Any]) -> torch.Tensor:
        """
        Convert state dict to model input tensor.

        Flattens state dictionary into feature vector:
        [src, dst, slots_needed, est_remaining_time, is_disaster,
         path1_features..., path2_features..., ...]

        :param state: State dictionary
        :type state: dict[str, Any]
        :return: Input tensor [1, input_dim]
        :rtype: torch.Tensor
        """
        features = []

        # Request features
        features.append(float(state['src']))
        features.append(float(state['dst']))
        features.append(float(state['slots_needed']))
        features.append(float(state['est_remaining_time']))
        features.append(float(state['is_disaster']))

        # Path features (for each of K paths)
        for path_features in state['paths']:
            features.append(float(path_features['path_hops']))
            features.append(float(path_features['min_residual_slots']))
            features.append(float(path_features['frag_indicator']))
            features.append(float(path_features['failure_mask']))
            features.append(float(path_features['dist_to_disaster_centroid']))

        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0)  # Add batch dimension

    def select_path(
        self,
        state: dict[str, Any],
        action_mask: list[bool]
    ) -> int:
        """
        Select path using BC policy with action masking.

        Steps:
        1. Convert state to tensor
        2. Forward pass through model
        3. Apply action mask (set logits to -inf for masked actions)
        4. Argmax for selected path

        :param state: Current state
        :type state: dict[str, Any]
        :param action_mask: Feasibility mask
        :type action_mask: list[bool]
        :return: Selected path index
        :rtype: int
        :raises AllPathsMaskedError: If all paths masked
        """
        # Check if all masked
        if not any(action_mask):
            raise AllPathsMaskedError("All paths masked")

        # Convert state to tensor
        state_tensor = self._state_to_tensor(state)

        # Forward pass (no gradient needed)
        with torch.no_grad():
            logits = self.model(state_tensor)  # [1, K]

        # Apply action mask
        mask_tensor = torch.tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device
        )
        logits = logits.squeeze(0)  # [K]
        logits[~mask_tensor] = float('-inf')  # Mask infeasible actions

        # Select action with highest logit
        selected = torch.argmax(logits).item()

        return selected
```

---

## 4. IQL Policy

### Implicit Q-Learning

**Location**: `fusion/modules/rl/policies/iql_policy.py`

```python
"""
Implicit Q-Learning policy for path selection.
"""

from typing import Any
from pathlib import Path
import torch
import torch.nn as nn
from .base import PathPolicy, AllPathsMaskedError
import logging

logger = logging.getLogger(__name__)


class IQLPolicy(PathPolicy):
    """
    Implicit Q-Learning policy for path selection.

    Conservative offline RL policy that avoids out-of-distribution
    actions through implicit Q-learning.

    :param model_path: Path to saved IQL model
    :type model_path: str
    :param device: Torch device
    :type device: str

    Example:
        >>> policy = IQLPolicy('models/iql_model.pt', device='cpu')
        >>> selected = policy.select_path(state, action_mask)
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """
        Initialize IQL policy.

        :param model_path: Path to model file
        :type model_path: str
        :param device: Compute device
        :type device: str
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"IQL model not found: {model_path}")

        self.device = torch.device(device)
        self.actor = self._load_model(model_path)
        self.actor.to(self.device)
        self.actor.eval()

        logger.info(f"Loaded IQL policy from {model_path} on {self.device}")

    def _load_model(self, model_path: Path) -> nn.Module:
        """
        Load pre-trained IQL actor network.

        :param model_path: Path to model
        :type model_path: Path
        :return: Loaded actor
        :rtype: nn.Module
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract actor from checkpoint
            if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                actor = checkpoint['actor']
            else:
                actor = checkpoint

            return actor

        except Exception as e:
            logger.error(f"Failed to load IQL model: {e}")
            raise

    def _state_to_tensor(self, state: dict[str, Any]) -> torch.Tensor:
        """
        Convert state dict to model input tensor.

        Same format as BC policy.

        :param state: State dictionary
        :type state: dict[str, Any]
        :return: Input tensor [1, input_dim]
        :rtype: torch.Tensor
        """
        # Reuse BC's state encoding
        from .bc_policy import BCPolicy
        return BCPolicy._state_to_tensor(self, state)

    def select_path(
        self,
        state: dict[str, Any],
        action_mask: list[bool]
    ) -> int:
        """
        Select path using IQL actor with action masking.

        IQL learns a policy that stays close to the behavior policy
        (conservative), making it safe for deployment.

        :param state: Current state
        :type state: dict[str, Any]
        :param action_mask: Feasibility mask
        :type action_mask: list[bool]
        :return: Selected path index
        :rtype: int
        :raises AllPathsMaskedError: If all paths masked
        """
        # Check if all masked
        if not any(action_mask):
            raise AllPathsMaskedError("All paths masked")

        # Convert state to tensor
        state_tensor = self._state_to_tensor(state)

        # Forward pass through actor
        with torch.no_grad():
            action_probs = self.actor(state_tensor)  # [1, K] (probabilities)

        # Apply action mask
        mask_tensor = torch.tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device
        )
        action_probs = action_probs.squeeze(0)  # [K]
        action_probs[~mask_tensor] = 0.0  # Zero out infeasible actions

        # Renormalize
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()
        else:
            raise AllPathsMaskedError("All paths masked after probability filtering")

        # Select action with highest probability
        selected = torch.argmax(action_probs).item()

        return selected
```

---

## 5. Action Masking

### Masking Utilities

**Location**: `fusion/modules/rl/policies/action_masking.py`

```python
"""
Action masking utilities for safe RL deployment.
"""

from typing import Any
from .base import PathPolicy, AllPathsMaskedError


def compute_action_mask(
    k_paths: list[list[int]],
    k_path_features: list[dict[str, Any]],
    slots_needed: int
) -> list[bool]:
    """
    Compute feasibility mask for K candidate paths.

    A path is masked (infeasible) if:
    - failure_mask == 1 (path uses failed link)
    - min_residual_slots < slots_needed (insufficient spectrum)

    :param k_paths: K candidate paths
    :type k_paths: list[list[int]]
    :param k_path_features: Features for each path
    :type k_path_features: list[dict[str, Any]]
    :param slots_needed: Required contiguous slots
    :type slots_needed: int
    :return: Boolean mask (True = feasible, False = masked)
    :rtype: list[bool]

    Example:
        >>> mask = compute_action_mask(
        ...     k_paths=[[0,1,2], [0,3,2], [0,4,5,2]],
        ...     k_path_features=features,
        ...     slots_needed=4
        ... )
        >>> print(mask)
        [False, True, True]  # First path infeasible
    """
    mask = []

    for features in k_path_features:
        # Check failure condition
        if features['failure_mask'] == 1:
            mask.append(False)
            continue

        # Check spectrum availability
        if features['min_residual_slots'] < slots_needed:
            mask.append(False)
            continue

        # Path is feasible
        mask.append(True)

    return mask


def apply_fallback_policy(
    state: dict[str, Any],
    fallback_policy: PathPolicy,
    action_mask: list[bool]
) -> int:
    """
    Apply fallback policy when all actions are masked.

    Attempts to use fallback policy (typically KSP-FF) with
    relaxed constraints or alternative path set.

    :param state: Current state
    :type state: dict[str, Any]
    :param fallback_policy: Fallback policy (KSP-FF or 1+1)
    :type fallback_policy: PathPolicy
    :param action_mask: Current action mask
    :type action_mask: list[bool]
    :return: Fallback path index or -1 if all blocked
    :rtype: int

    Example:
        >>> fallback = KSPFFPolicy()
        >>> idx = apply_fallback_policy(state, fallback, action_mask)
        >>> if idx == -1:
        ...     print("Request blocked")
    """
    try:
        # Try fallback with full feasibility mask
        # (fallback may use different logic)
        return fallback_policy.select_path(
            state,
            [True] * len(state['paths'])
        )

    except AllPathsMaskedError:
        # Even fallback failed
        return -1  # Block the request
```

---

## 6. Integration with SDNController

### Policy-Based Routing

**Location**: `fusion/core/sdn_controller.py` (extension)

```python
def route_request_with_policy(
    self,
    request: dict[str, Any],
    sdn_props: SDNProps
) -> list[int] | None:
    """
    Route request using RL policy with action masking.

    :param request: Request details
    :type request: dict[str, Any]
    :param sdn_props: SDN properties
    :type sdn_props: SDNProps
    :return: Selected path or None if blocked
    :rtype: list[int] | None
    """
    source = request['source']
    destination = request['destination']

    # Get K candidate paths
    k_paths = self.k_path_cache.get_k_paths(source, destination)

    if not k_paths:
        return None

    # Compute path features
    k_path_features = [
        self.k_path_cache.get_path_features(
            path,
            self.network_spectrum_dict,
            self.failure_manager
        )
        for path in k_paths
    ]

    # Build state dict
    state = {
        'src': source,
        'dst': destination,
        'slots_needed': sdn_props.slots_needed,
        'est_remaining_time': request['depart'] - request['arrive'],
        'is_disaster': 1 if self.failure_manager and self.failure_manager.active_failures else 0,
        'paths': k_path_features
    }

    # Compute action mask
    action_mask = compute_action_mask(
        k_paths,
        k_path_features,
        sdn_props.slots_needed
    )

    # Select path using policy
    try:
        if all(not m for m in action_mask):
            # All masked, use fallback
            path_index = apply_fallback_policy(
                state,
                self.fallback_policy,
                action_mask
            )
        else:
            path_index = self.policy.select_path(state, action_mask)

    except AllPathsMaskedError:
        path_index = -1

    if path_index == -1:
        # Block request
        sdn_props.was_routed = False
        sdn_props.block_reason = "all_paths_infeasible"
        return None

    # Route on selected path
    selected_path = k_paths[path_index]
    return selected_path
```

---

## 7. Configuration

```ini
[offline_rl_settings]
# Policy selection
policy_type = bc  # ksp_ff, one_plus_one, bc, iql

# Model paths
bc_model_path = models/bc_model.pt
iql_model_path = models/iql_model.pt

# Device
device = cpu  # cpu, cuda, mps

# Fallback policy
fallback_policy = ksp_ff
```

---

## 8. Testing Requirements

### Unit Tests

See next document (31-dataset-logging.md) for dataset testing.

```python
import pytest
from fusion.modules.rl.policies import (
    KSPFFPolicy,
    BCPolicy,
    IQLPolicy,
    compute_action_mask
)


def test_ksp_ff_policy_selects_first_feasible():
    """Test that KSP-FF always picks first unmasked path."""
    policy = KSPFFPolicy()

    state = {}  # Not used by KSP-FF
    action_mask = [False, False, True, True]

    selected = policy.select_path(state, action_mask)
    assert selected == 2  # First feasible


def test_action_mask_computation():
    """Test that mask correctly identifies infeasible paths."""
    k_paths = [[0, 1, 2], [0, 3, 2], [0, 4, 2]]
    k_path_features = [
        {'failure_mask': 1, 'min_residual_slots': 10},  # Failed
        {'failure_mask': 0, 'min_residual_slots': 5},   # Sufficient
        {'failure_mask': 0, 'min_residual_slots': 2},   # Insufficient
    ]

    mask = compute_action_mask(k_paths, k_path_features, slots_needed=4)

    assert mask == [False, True, False]


# Additional tests in test_bc_policy.py, test_iql_policy.py
```

---

## 9. Acceptance Criteria

- [x] `test_ksp_ff_policy_selects_first_feasible`: Always picks first unmasked path
- [x] `test_bc_policy_loads_model`: BC model loads from .pt file
- [x] `test_bc_policy_respects_action_mask`: Masked actions not selected
- [x] `test_iql_policy_inference`: IQL policy runs forward pass correctly
- [x] `test_action_mask_computation`: Mask correctly identifies infeasible paths
- [x] `test_fallback_on_all_masked`: Fallback policy applied when all masked
- [x] `test_state_tensor_format`: State dict converts to correct tensor shape

---

**Related Documents**:
- [11-k-path-cache.md](../phase2-infrastructure/11-k-path-cache.md) (Path features)
- [31-dataset-logging.md](31-dataset-logging.md) (Offline dataset)
- [40-metrics-reporting.md](../phase5-metrics/40-metrics-reporting.md) (Policy comparison)
