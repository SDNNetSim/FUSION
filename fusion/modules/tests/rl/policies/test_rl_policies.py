"""
Tests for RL policies (BC and IQL).
"""

from typing import Any

import pytest
import torch.nn as nn

from fusion.modules.rl.policies import (
    BCPolicy,
    IQLPolicy,
)


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create sample state dictionary."""
    return {
        "src": 0,
        "dst": 5,
        "slots_needed": 2,
        "est_remaining_time": 1.5,
        "is_disaster": 0,
        "paths": [
            {
                "path_hops": 3,
                "min_residual_slots": 10,
                "frag_indicator": 0.2,
                "failure_mask": 0,
                "dist_to_disaster_centroid": 0,
            },
            {
                "path_hops": 4,
                "min_residual_slots": 8,
                "frag_indicator": 0.3,
                "failure_mask": 0,
                "dist_to_disaster_centroid": 1,
            },
            {
                "path_hops": 5,
                "min_residual_slots": 6,
                "frag_indicator": 0.4,
                "failure_mask": 1,
                "dist_to_disaster_centroid": 2,
            },
        ],
    }


def _get_bc_state_dict() -> dict:
    """Create BC model state dict without saving to disk."""
    input_dim = 5 + (5 * 3)
    output_dim = 3

    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )

    # Map Sequential indices to fc1/fc2/fc3 naming that BCPolicy expects
    state_dict = model.state_dict()
    return {
        "fc1.weight": state_dict["0.weight"],
        "fc1.bias": state_dict["0.bias"],
        "fc2.weight": state_dict["2.weight"],
        "fc2.bias": state_dict["2.bias"],
        "fc3.weight": state_dict["4.weight"],
        "fc3.bias": state_dict["4.bias"],
    }


def _get_iql_state_dict() -> dict:
    """Create IQL model state dict without saving to disk."""
    input_dim = 5 + (5 * 3)
    output_dim = 3

    actor = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
        nn.Softmax(dim=-1),
    )

    return {"actor": actor.state_dict()}


@pytest.fixture
def bc_model(monkeypatch: Any, tmp_path: Any) -> str:
    """Mock BC model loading to avoid torch.save pickling issues."""
    state_dict = _get_bc_state_dict()

    # Create an actual dummy file so Path.exists() returns True
    model_path = tmp_path / "bc_model.pt"
    model_path.touch()

    def mock_load_model(self: Any, model_path: Any) -> nn.Module:
        """Mock _load_model to return a model built from in-memory state dict."""
        # Build the model architecture
        input_dim = state_dict["fc1.weight"].shape[1]
        output_dim = state_dict["fc3.weight"].shape[0]

        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

        # Remap the state dict keys from fc1/fc2/fc3 to Sequential indices
        remapped_state_dict = {
            "0.weight": state_dict["fc1.weight"],
            "0.bias": state_dict["fc1.bias"],
            "2.weight": state_dict["fc2.weight"],
            "2.bias": state_dict["fc2.bias"],
            "4.weight": state_dict["fc3.weight"],
            "4.bias": state_dict["fc3.bias"],
        }

        model.load_state_dict(remapped_state_dict)
        return model

    # Patch the _load_model method on BCPolicy
    monkeypatch.setattr(
        "fusion.modules.rl.policies.bc_policy.BCPolicy._load_model", mock_load_model
    )
    return str(model_path)


@pytest.fixture
def iql_model(monkeypatch: Any, tmp_path: Any) -> str:
    """Mock IQL model loading to avoid torch.save pickling issues."""
    checkpoint = _get_iql_state_dict()

    # Create an actual dummy file so Path.exists() returns True
    model_path = tmp_path / "iql_model.pt"
    model_path.touch()

    def mock_load_model(self: Any, model_path: Any) -> nn.Module:
        """Mock _load_model to return actor from in-memory checkpoint."""
        actor_state = checkpoint["actor"]
        result: nn.Module = self._build_actor_from_state_dict(actor_state)
        return result

    # Patch the _load_model method on IQLPolicy
    monkeypatch.setattr(
        "fusion.modules.rl.policies.iql_policy.IQLPolicy._load_model", mock_load_model
    )
    return str(model_path)


class TestBCPolicy:
    """Test cases for BC policy."""

    def test_bc_policy_loads_model(self, bc_model: str) -> None:
        """Test that BC model loads from .pt file."""
        policy = BCPolicy(bc_model, device="cpu")
        assert policy.model is not None
        assert policy.device.type == "cpu"

    def test_bc_policy_raises_on_missing_file(self, monkeypatch: Any) -> None:
        """Test that FileNotFoundError raised for missing model."""
        # Don't mock torch.load for this test - let it fail naturally
        with pytest.raises(FileNotFoundError):
            BCPolicy("nonexistent_model.pt", device="cpu")

    def test_bc_policy_respects_action_mask(
        self, bc_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that masked actions are not selected."""
        policy = BCPolicy(bc_model, device="cpu")

        # Only middle path is feasible
        action_mask = [False, True, False]

        selected = policy.select_path(sample_state, action_mask)

        # Should select the only feasible path
        assert selected == 1

    def test_bc_policy_state_tensor_format(
        self, bc_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that state dict converts to correct tensor shape."""
        policy = BCPolicy(bc_model, device="cpu")

        tensor = policy._state_to_tensor(sample_state)

        # Expected: [1, input_dim] where input_dim = 5 + (5 * 3)
        assert tensor.shape == (1, 20)

    def test_bc_policy_returns_negative_one_when_all_masked(
        self, bc_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that -1 returned when all paths masked."""
        policy = BCPolicy(bc_model, device="cpu")

        action_mask = [False, False, False]

        selected = policy.select_path(sample_state, action_mask)
        assert selected == -1

    def test_bc_policy_forward_pass(
        self, bc_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that forward pass completes without errors."""
        policy = BCPolicy(bc_model, device="cpu")

        action_mask = [True, True, True]

        # Should not raise
        selected = policy.select_path(sample_state, action_mask)

        assert 0 <= selected < 3

    def test_bc_policy_get_name(self, bc_model: str) -> None:
        """Test that policy name is correct."""
        policy = BCPolicy(bc_model, device="cpu")
        assert policy.get_name() == "BCPolicy"


class TestIQLPolicy:
    """Test cases for IQL policy."""

    def test_iql_policy_loads_model(self, iql_model: str) -> None:
        """Test that IQL model loads from checkpoint."""
        policy = IQLPolicy(iql_model, device="cpu")
        assert policy.actor is not None
        assert policy.device.type == "cpu"

    def test_iql_policy_raises_on_missing_file(self, monkeypatch: Any) -> None:
        """Test that FileNotFoundError raised for missing model."""
        # Don't mock torch.load for this test - let it fail naturally
        with pytest.raises(FileNotFoundError):
            IQLPolicy("nonexistent_model.pt", device="cpu")

    def test_iql_policy_respects_action_mask(
        self, iql_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that masked actions are not selected."""
        policy = IQLPolicy(iql_model, device="cpu")

        # Only first path is feasible
        action_mask = [True, False, False]

        selected = policy.select_path(sample_state, action_mask)

        # Should select the only feasible path
        assert selected == 0

    def test_iql_policy_state_tensor_format(
        self, iql_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that state dict converts to correct tensor shape."""
        policy = IQLPolicy(iql_model, device="cpu")

        tensor = policy._state_to_tensor(sample_state)

        # Expected: [1, input_dim] where input_dim = 5 + (5 * 3)
        assert tensor.shape == (1, 20)

    def test_iql_policy_returns_negative_one_when_all_masked(
        self, iql_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that -1 returned when all paths masked."""
        policy = IQLPolicy(iql_model, device="cpu")

        action_mask = [False, False, False]

        selected = policy.select_path(sample_state, action_mask)
        assert selected == -1

    def test_iql_policy_forward_pass(
        self, iql_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that forward pass completes without errors."""
        policy = IQLPolicy(iql_model, device="cpu")

        action_mask = [True, True, True]

        # Should not raise
        selected = policy.select_path(sample_state, action_mask)

        assert 0 <= selected < 3

    def test_iql_policy_probability_renormalization(
        self, iql_model: str, sample_state: dict[str, Any]
    ) -> None:
        """Test that probabilities renormalize after masking."""
        policy = IQLPolicy(iql_model, device="cpu")

        # Mask one path
        action_mask = [True, True, False]

        # Should still work and select from remaining paths
        selected = policy.select_path(sample_state, action_mask)

        assert selected in [0, 1]

    def test_iql_policy_get_name(self, iql_model: str) -> None:
        """Test that policy name is correct."""
        policy = IQLPolicy(iql_model, device="cpu")
        assert policy.get_name() == "IQLPolicy"
