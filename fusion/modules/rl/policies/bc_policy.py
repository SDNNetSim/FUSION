"""
Behavior Cloning policy for path selection.

This module implements a BC (Behavior Cloning) policy that imitates
heuristic behavior using supervised learning on offline datasets.
"""

import logging
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from .base import PathPolicy

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
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"BC model not found: {model_path}")

        self.device = torch.device(device)
        self.model = self._load_model(model_path_obj)
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
            model = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )  # nosec B614 - Loading trusted model files from local filesystem

            if isinstance(model, dict):
                # Model saved as state dict
                # Need to reconstruct model architecture
                model = self._build_model_architecture(model)

            # Cast to nn.Module (torch.load returns Any)
            return cast(nn.Module, model)

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
        input_dim = state_dict["fc1.weight"].shape[1]
        output_dim = state_dict["fc3.weight"].shape[0]

        # Build model
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
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
        :return: Selected path index, or -1 if all paths masked
        :rtype: int
        """
        # Check if all masked
        if not any(action_mask):
            return -1  # All paths masked - request should be blocked

        # Convert state to tensor
        state_tensor = self._state_to_tensor(state)

        # Forward pass (no gradient needed)
        with torch.no_grad():
            logits = self.model(state_tensor)  # [1, K]

        # Apply action mask
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        logits = logits.squeeze(0)  # [K]
        logits[~mask_tensor] = float("-inf")  # Mask infeasible actions

        # Select action with highest logit
        selected: int = int(torch.argmax(logits).item())

        return selected
