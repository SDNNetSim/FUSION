"""
Tests for RL policies (BC and IQL).
"""

import pytest
import torch
import torch.nn as nn

from fusion.modules.rl.policies import (
    AllPathsMaskedError,
    BCPolicy,
    IQLPolicy,
)


@pytest.fixture
def sample_state():
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


@pytest.fixture
def bc_model(tmp_path):
    """Create a simple BC model for testing."""
    # Calculate input dimension: 5 request features + 5 path features * 3 paths
    input_dim = 5 + (5 * 3)
    output_dim = 3  # 3 paths

    # Create simple model
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )

    # Save model
    model_path = tmp_path / "bc_model.pt"
    torch.save(model, str(model_path))

    return str(model_path)


@pytest.fixture
def iql_model(tmp_path):
    """Create a simple IQL actor model for testing."""
    # Same architecture as BC
    input_dim = 5 + (5 * 3)
    output_dim = 3

    # IQL actor outputs probabilities
    actor = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
        nn.Softmax(dim=-1),  # Output probabilities
    )

    # Save as checkpoint
    model_path = tmp_path / "iql_model.pt"
    torch.save({"actor": actor}, str(model_path))

    return str(model_path)


class TestBCPolicy:
    """Test cases for BC policy."""

    def test_bc_policy_loads_model(self, bc_model):
        """Test that BC model loads from .pt file."""
        policy = BCPolicy(bc_model, device="cpu")
        assert policy.model is not None
        assert policy.device.type == "cpu"

    def test_bc_policy_raises_on_missing_file(self):
        """Test that FileNotFoundError raised for missing model."""
        with pytest.raises(FileNotFoundError):
            BCPolicy("nonexistent_model.pt", device="cpu")

    def test_bc_policy_respects_action_mask(self, bc_model, sample_state):
        """Test that masked actions are not selected."""
        policy = BCPolicy(bc_model, device="cpu")

        # Only middle path is feasible
        action_mask = [False, True, False]

        selected = policy.select_path(sample_state, action_mask)

        # Should select the only feasible path
        assert selected == 1

    def test_bc_policy_state_tensor_format(self, bc_model, sample_state):
        """Test that state dict converts to correct tensor shape."""
        policy = BCPolicy(bc_model, device="cpu")

        tensor = policy._state_to_tensor(sample_state)

        # Expected: [1, input_dim] where input_dim = 5 + (5 * 3)
        assert tensor.shape == (1, 20)

    def test_bc_policy_raises_when_all_masked(self, bc_model, sample_state):
        """Test that exception raised when all paths masked."""
        policy = BCPolicy(bc_model, device="cpu")

        action_mask = [False, False, False]

        with pytest.raises(AllPathsMaskedError):
            policy.select_path(sample_state, action_mask)

    def test_bc_policy_forward_pass(self, bc_model, sample_state):
        """Test that forward pass completes without errors."""
        policy = BCPolicy(bc_model, device="cpu")

        action_mask = [True, True, True]

        # Should not raise
        selected = policy.select_path(sample_state, action_mask)

        assert 0 <= selected < 3

    def test_bc_policy_get_name(self, bc_model):
        """Test that policy name is correct."""
        policy = BCPolicy(bc_model, device="cpu")
        assert policy.get_name() == "BCPolicy"


class TestIQLPolicy:
    """Test cases for IQL policy."""

    def test_iql_policy_loads_model(self, iql_model):
        """Test that IQL model loads from checkpoint."""
        policy = IQLPolicy(iql_model, device="cpu")
        assert policy.actor is not None
        assert policy.device.type == "cpu"

    def test_iql_policy_raises_on_missing_file(self):
        """Test that FileNotFoundError raised for missing model."""
        with pytest.raises(FileNotFoundError):
            IQLPolicy("nonexistent_model.pt", device="cpu")

    def test_iql_policy_respects_action_mask(self, iql_model, sample_state):
        """Test that masked actions are not selected."""
        policy = IQLPolicy(iql_model, device="cpu")

        # Only first path is feasible
        action_mask = [True, False, False]

        selected = policy.select_path(sample_state, action_mask)

        # Should select the only feasible path
        assert selected == 0

    def test_iql_policy_state_tensor_format(self, iql_model, sample_state):
        """Test that state dict converts to correct tensor shape."""
        policy = IQLPolicy(iql_model, device="cpu")

        tensor = policy._state_to_tensor(sample_state)

        # Expected: [1, input_dim] where input_dim = 5 + (5 * 3)
        assert tensor.shape == (1, 20)

    def test_iql_policy_raises_when_all_masked(self, iql_model, sample_state):
        """Test that exception raised when all paths masked."""
        policy = IQLPolicy(iql_model, device="cpu")

        action_mask = [False, False, False]

        with pytest.raises(AllPathsMaskedError):
            policy.select_path(sample_state, action_mask)

    def test_iql_policy_forward_pass(self, iql_model, sample_state):
        """Test that forward pass completes without errors."""
        policy = IQLPolicy(iql_model, device="cpu")

        action_mask = [True, True, True]

        # Should not raise
        selected = policy.select_path(sample_state, action_mask)

        assert 0 <= selected < 3

    def test_iql_policy_probability_renormalization(self, iql_model, sample_state):
        """Test that probabilities renormalize after masking."""
        policy = IQLPolicy(iql_model, device="cpu")

        # Mask one path
        action_mask = [True, True, False]

        # Should still work and select from remaining paths
        selected = policy.select_path(sample_state, action_mask)

        assert selected in [0, 1]

    def test_iql_policy_get_name(self, iql_model):
        """Test that policy name is correct."""
        policy = IQLPolicy(iql_model, device="cpu")
        assert policy.get_name() == "IQLPolicy"
