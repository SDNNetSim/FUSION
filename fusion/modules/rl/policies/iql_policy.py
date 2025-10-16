"""
Implicit Q-Learning policy for path selection.

This module implements an IQL (Implicit Q-Learning) policy, a conservative
offline RL algorithm that avoids out-of-distribution actions.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .base import AllPathsMaskedError, PathPolicy

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
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"IQL model not found: {model_path}")

        self.device = torch.device(device)
        self.actor = self._load_model(model_path_obj)
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
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )  # nosec B614 - Loading trusted model files from local filesystem

            # Extract actor from checkpoint
            if isinstance(checkpoint, dict) and "actor" in checkpoint:
                actor = checkpoint["actor"]
                # If actor is a state_dict, reconstruct the model
                if isinstance(actor, dict):
                    actor = self._build_actor_from_state_dict(actor)
            else:
                actor = checkpoint

            return actor

        except Exception as e:
            logger.error(f"Failed to load IQL model: {e}")
            raise

    def _build_actor_from_state_dict(self, state_dict: dict) -> nn.Module:
        """
        Build actor architecture from state dict.

        Infers dimensions from the state dict and creates a simple
        3-layer MLP with Softmax output.

        :param state_dict: Actor state dictionary
        :type state_dict: dict
        :return: Actor model with loaded weights
        :rtype: nn.Module
        """
        # Infer dimensions from state dict
        # Assumes Sequential with Linear layers at indices 0, 2
        input_dim = state_dict["0.weight"].shape[1]
        hidden_dim = state_dict["0.weight"].shape[0]
        output_dim = state_dict["2.weight"].shape[0]

        # Build actor architecture
        actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )

        actor.load_state_dict(state_dict)
        return actor

    def _state_to_tensor(self, state: dict[str, Any]) -> torch.Tensor:
        """
        Convert state dict to model input tensor.

        Same format as BC policy.

        :param state: State dictionary
        :type state: dict[str, Any]
        :return: Input tensor [1, input_dim]
        :rtype: torch.Tensor
        """
        features = []

        # Request features
        features.append(float(state["src"]))
        features.append(float(state["dst"]))
        features.append(float(state["slots_needed"]))
        features.append(float(state["est_remaining_time"]))
        features.append(float(state["is_disaster"]))

        # Path features (for each of K paths)
        for path_features in state["paths"]:
            features.append(float(path_features["path_hops"]))
            features.append(float(path_features["min_residual_slots"]))
            features.append(float(path_features["frag_indicator"]))
            features.append(float(path_features["failure_mask"]))
            features.append(float(path_features["dist_to_disaster_centroid"]))

        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0)  # Add batch dimension

    def select_path(self, state: dict[str, Any], action_mask: list[bool]) -> int:
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
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        action_probs = action_probs.squeeze(0)  # [K]
        action_probs[~mask_tensor] = 0.0  # Zero out infeasible actions

        # Renormalize
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()
        else:
            raise AllPathsMaskedError("All paths masked after probability filtering")

        # Select action with highest probability
        selected: int = int(torch.argmax(action_probs).item())

        return selected
