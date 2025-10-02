"""
Pointer-based policy implementation for reinforcement learning.

This module implements a pointer network-based policy that uses attention mechanisms
for path selection in network routing scenarios.
"""

# Standard library imports
import math
from typing import Any

# Third-party imports
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

# Module constants
DEFAULT_ATTENTION_HEADS = 3
QUERY_KEY_VALUE_MULTIPLIER = 3

__all__ = [
    "PointerHead",
    "PointerPolicy",
]


class PointerHead(torch.nn.Module):
    """
    Pointer head implementation for attention-based path selection.

    Uses query-key-value attention mechanism to compute logits for path selection
    in reinforcement learning policies. The attention mechanism allows the model
    to focus on the most relevant paths for decision making.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initialize the pointer head with specified dimension.

        :param dimension: Input feature dimension for linear transformations
        :type dimension: int
        :raises ValueError: If dimension is not positive
        """
        super().__init__()
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")

        self.query_key_value = torch.nn.Linear(
            dimension, dimension * QUERY_KEY_VALUE_MULTIPLIER
        )
        self.dimension = dimension

    def forward(self, path_features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-based logits for path selection.

        Performs multi-head attention computation to generate path selection logits.
        The attention mechanism allows the model to weigh different paths based on
        their relevance for the current decision.

        :param path_features: Input tensor with shape (batch, 3, dimension)
        :type path_features: torch.Tensor
        :return: Path selection logits with shape (batch, 3)
        :rtype: torch.Tensor
        :raises ValueError: If input tensor has incorrect shape

        Example:
            >>> head = PointerHead(64)
            >>> features = torch.randn(32, 3, 64)
            >>> logits = head.forward(features)
            >>> print(logits.shape)
            torch.Size([32, 3])
        """
        # Input validation
        if path_features.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {path_features.dim()}D tensor")
        if path_features.size(1) != DEFAULT_ATTENTION_HEADS:
            raise ValueError(
                f"Expected {DEFAULT_ATTENTION_HEADS} paths, got {path_features.size(1)}"
            )
        if path_features.size(-1) != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {path_features.size(-1)}"
            )

        # Compute query, key, value transformations
        query_key_value = self.query_key_value(path_features)  # (batch, 3, 3*dimension)
        query, key, value = query_key_value.chunk(QUERY_KEY_VALUE_MULTIPLIER, dim=-1)

        # Compute attention scores
        scores = torch.einsum("bid,bjd->bij", query, key) / math.sqrt(query.size(-1))
        attention = torch.softmax(scores, dim=-1)  # (batch, 3, 3)

        # Apply attention to values
        output = torch.einsum("bij,bjd->bid", attention, value)  # (batch, 3, dimension)

        # Generate path logits
        logits = output.sum(dim=-1)  # (batch, 3)
        return logits


class PointerPolicy(ActorCriticPolicy):
    """
    Pointer-based policy for reinforcement learning with attention mechanisms.

    Integrates the PointerHead attention mechanism with Stable Baselines3's
    ActorCriticPolicy framework. This policy is designed for scenarios where
    the agent needs to select from a discrete set of paths or options.

    The policy uses attention-based path selection rather than traditional
    multi-layer perceptrons for the policy network.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the pointer policy.

        :param args: Positional arguments passed to ActorCriticPolicy
        :param kwargs: Keyword arguments passed to ActorCriticPolicy
        """
        super().__init__(*args, **kwargs)
        # mlp_extractor will be initialized in _build_mlp_extractor()

    def _build_mlp_extractor(self) -> None:
        """
        Build the MLP extractor using PointerHead for policy network.

        Overrides the default MLP extractor to use the attention-based PointerHead
        for the policy network while maintaining a standard linear layer for the
        value network.

        :raises AttributeError: If features_extractor is not properly initialized
        """
        if not hasattr(self, "features_extractor") or self.features_extractor is None:
            raise AttributeError(
                "Features extractor must be initialized before building MLP extractor"
            )

        # Initialize MLP extractor with dummy networks
        self.mlp_extractor = MlpExtractor(
            feature_dim=self.features_extractor.features_dim,
            net_arch=[],  # Empty architecture since we'll override
            activation_fn=torch.nn.ReLU,
        )

        # Use PointerHead for policy network (attention-based path selection)
        self.mlp_extractor.policy_net = PointerHead(  # type: ignore[assignment]
            self.features_extractor.features_dim
        )

        # Use standard linear layer for value network
        self.mlp_extractor.value_net = torch.nn.Linear(  # type: ignore[assignment]
            self.features_extractor.features_dim, 1
        )
