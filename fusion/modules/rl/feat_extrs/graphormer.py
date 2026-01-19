"""
Graph Transformer feature extractor for reinforcement learning.

This module implements a Transformer-based graph neural network feature extractor
that uses multi-head attention mechanisms to process graph-structured observations.
"""

import torch
from gymnasium import spaces
from torch_geometric.nn import TransformerConv

from fusion.modules.rl.feat_extrs.base_feature_extractor import (
    BaseGraphFeatureExtractor,
)
from fusion.modules.rl.feat_extrs.constants import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    EDGE_EMBEDDING_SCALE_FACTOR,
)

# TODO: (v6.X) Integrate Graphormer hyperparameters with Optuna search space


class GraphTransformerExtractor(BaseGraphFeatureExtractor):
    """
    Custom Graph Transformer feature extractor integrated with StableBaselines3.

    This feature extractor uses Transformer convolution layers with multi-head
    attention to process graph observations and extract meaningful features
    for reinforcement learning.
    """

    def __init__(
        self,
        obs_space: spaces.Dict,  # Note: kept for backward compatibility
        emb_dim: int = DEFAULT_EMBEDDING_DIMENSION,  # Kept for compat
        heads: int = DEFAULT_NUM_HEADS,  # Kept for compat
        layers: int = DEFAULT_NUM_LAYERS,
    ):
        """
        Initialize the Graph Transformer feature extractor.

        :param obs_space: Observation space containing graph components:
            - 'x': Node features [num_nodes, feature_dim]
            - 'edge_index': Edge connectivity [2, num_edges]
            - 'path_masks': Path selection masks [num_paths, num_edges]
        :type obs_space: spaces.Dict
        :param emb_dim: Total embedding dimension (must be divisible by heads)
        :type emb_dim: int
        :param heads: Number of attention heads
        :type heads: int
        :param layers: Number of transformer convolution layers
        :type layers: int
        :raises ValueError: If emb_dim is not divisible by heads
        """
        # Calculate dimensions
        path_masks_shape = obs_space["path_masks"].shape
        assert path_masks_shape is not None and len(path_masks_shape) >= 2
        num_paths = path_masks_shape[0]

        x_shape = obs_space["x"].shape
        assert x_shape is not None and len(x_shape) >= 2
        input_dimension = x_shape[1]

        if emb_dim % heads != 0:
            raise ValueError(f"Embedding dimension ({emb_dim}) must be divisible by number of heads ({heads})")

        output_per_head = emb_dim // heads
        convolution_output_dimension = heads * output_per_head

        # Initialize base class with total feature dimension
        features_dimension = emb_dim * num_paths
        super().__init__(obs_space, features_dimension)

        # Create transformer convolution layers
        self.convolution_layers = torch.nn.ModuleList(
            [
                TransformerConv(
                    in_channels=(input_dimension if layer_idx == 0 else convolution_output_dimension),
                    out_channels=output_per_head,
                    heads=heads,
                    concat=True,  # Concatenate attention head outputs
                )
                for layer_idx in range(layers)
            ]
        )

        # Readout layer to transform concatenated head outputs to final embedding
        self.readout_layer = torch.nn.Linear(convolution_output_dimension, emb_dim)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert graph observation to fixed-size feature vector using transformer layers.

        :param observation: Dictionary containing:
            - 'x': Node features
                [batch_size, num_nodes, features] or [num_nodes, features]
            - 'edge_index': Edge indices [batch_size, 2, num_edges] or [2, num_edges]
            - 'path_masks': Path masks
                [batch_size, num_paths, num_edges] or [num_paths, num_edges]
        :type observation: dict[str, torch.Tensor]
        :return: Feature vector [batch_size, feature_dim]
        :rtype: torch.Tensor
        """
        # Extract components from observation
        node_features_list = observation["x"]
        edge_index_list = observation["edge_index"].long()
        path_masks_list = observation["path_masks"]

        # Handle batch dimensions for three-dimensional inputs
        if node_features_list.dim() == 3:
            batch_size = node_features_list.size(0)

            if batch_size > 1:
                # Process multiple samples in batch
                batch_outputs: list[torch.Tensor] = []

                for batch_idx in range(batch_size):
                    # Extract sample from batch
                    node_features_batch = node_features_list[batch_idx]
                    edge_index_batch = edge_index_list[batch_idx] if edge_index_list.dim() == 3 else edge_index_list
                    path_masks_batch = path_masks_list[batch_idx] if path_masks_list.dim() == 3 else path_masks_list

                    # Process through transformer layers
                    node_embeddings_batch = node_features_batch
                    for convolution_layer in self.convolution_layers:
                        node_embeddings_batch = convolution_layer(node_embeddings_batch, edge_index_batch).relu()

                    # Compute edge embeddings
                    source_idx, destination_idx = edge_index_batch
                    edge_embeddings_batch = (
                        node_embeddings_batch[source_idx] + node_embeddings_batch[destination_idx]
                    ) * EDGE_EMBEDDING_SCALE_FACTOR

                    # Aggregate path embeddings
                    path_embeddings_batch = path_masks_batch @ edge_embeddings_batch

                    # Apply readout and flatten
                    path_vectors_batch = self.readout_layer(path_embeddings_batch).flatten()
                    batch_outputs.append(path_vectors_batch)

                return torch.stack(batch_outputs, dim=0)

            # Handle single sample with batch dimension
            node_features_list = node_features_list.squeeze(0)
            edge_index_list = edge_index_list.squeeze(0) if edge_index_list.dim() == 3 else edge_index_list
            path_masks_list = path_masks_list.squeeze(0) if path_masks_list.dim() == 3 else path_masks_list

        # Process single sample (no batch) or after squeezing batch=1
        node_embeddings = node_features_list
        for convolution_layer in self.convolution_layers:
            node_embeddings = convolution_layer(node_embeddings, edge_index_list).relu()

        # Compute edge embeddings
        source_idx, destination_idx = edge_index_list
        edge_embeddings = (node_embeddings[source_idx] + node_embeddings[destination_idx]) * EDGE_EMBEDDING_SCALE_FACTOR

        # Aggregate path embeddings
        path_embeddings: torch.Tensor = path_masks_list @ edge_embeddings
        path_vectors: torch.Tensor = self.readout_layer(path_embeddings)
        flattened_features = path_vectors.flatten()

        # Add batch dimension for consistency
        return flattened_features.unsqueeze(0)
