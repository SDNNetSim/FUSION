"""
Base feature extractor class for graph neural networks.

This module provides common functionality for GNN-based feature extractors
to reduce code duplication across different implementations.
"""

from abc import abstractmethod

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from fusion.modules.rl.feat_extrs.constants import EDGE_EMBEDDING_SCALE_FACTOR


class BaseGraphFeatureExtractor(BaseFeaturesExtractor):
    """
    Base class for graph neural network feature extractors.

    This class provides common functionality for processing graph observations
    and handling batch dimensions consistently across different GNN architectures.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int):
        """
        Initialize the base graph feature extractor.

        :param observation_space: The observation space containing graph data
        :type observation_space: spaces.Dict
        :param features_dim: The dimension of the output feature vector
        :type features_dim: int
        """
        super().__init__(observation_space, features_dim)

    def _process_batch_dimensions(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        path_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
        """
        Process and normalize batch dimensions for consistent handling.

        :param node_features: Node feature tensor
            [batch_size, num_nodes, features] or [num_nodes, features]
        :type node_features: torch.Tensor
        :param edge_index: Edge connectivity tensor
            [batch_size, 2, num_edges] or [2, num_edges]
        :type edge_index: torch.Tensor
        :param path_masks: Path selection masks
            [batch_size, num_paths, num_edges] or [num_paths, num_edges]
        :type path_masks: torch.Tensor
        :return: Tuple of (node_features, edge_index, path_masks, batch_size)
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]
        """
        # Add batch dimension if not present
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
            path_masks = path_masks.unsqueeze(0)
            batch_size = None  # Indicates single sample
        else:
            batch_size = node_features.size(0)

        return node_features, edge_index, path_masks, batch_size

    def _compute_edge_embeddings(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge embeddings from node embeddings.

        :param node_embeddings: Node embedding tensor [num_nodes, embedding_dim]
        :type node_embeddings: torch.Tensor
        :param edge_index: Edge connectivity tensor [2, num_edges]
        :type edge_index: torch.Tensor
        :return: Edge embeddings [num_edges, embedding_dim]
        :rtype: torch.Tensor
        """
        source_idx, destination_idx = edge_index
        edge_embeddings = (node_embeddings[source_idx] + node_embeddings[destination_idx]) * EDGE_EMBEDDING_SCALE_FACTOR
        return edge_embeddings

    def _compute_path_embeddings(self, edge_embeddings: torch.Tensor, path_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute path embeddings from edge embeddings using path masks.

        :param edge_embeddings: Edge embedding tensor [num_edges, embedding_dim]
        :type edge_embeddings: torch.Tensor
        :param path_masks: Path selection masks [num_paths, num_edges]
        :type path_masks: torch.Tensor
        :return: Path embeddings [num_paths, embedding_dim]
        :rtype: torch.Tensor
        """
        return path_masks @ edge_embeddings

    @abstractmethod
    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for processing observations.

        Must be implemented by subclasses to define the forward pass.

        :param observation: Input observation data
        :return: Processed feature tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
