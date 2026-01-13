"""
Cached Path GNN feature extractor for efficient inference.

This module provides a cached version of the Path GNN feature extractor
that pre-computes embeddings for static graphs to improve inference speed.
"""

import torch
from gymnasium import spaces
from torch_geometric.nn import GATv2Conv, GraphConv, SAGEConv

from fusion.modules.rl.feat_extrs.base_feature_extractor import (
    BaseGraphFeatureExtractor,
)
from fusion.modules.rl.feat_extrs.constants import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_GNN_TYPE,
    DEFAULT_NUM_LAYERS,
    EDGE_EMBEDDING_SCALE_FACTOR,
)


class PathGNNEncoder(torch.nn.Module):
    """
    GNN encoder for pre-computing graph embeddings.

    This encoder processes static graph structures to generate embeddings
    that can be cached and reused during inference.
    """

    def __init__(
        self,
        obs_space: spaces.Dict,  # Note: kept for backward compatibility
        emb_dim: int = DEFAULT_EMBEDDING_DIMENSION,  # Kept for compat
        gnn_type: str = DEFAULT_GNN_TYPE,
        layers: int = DEFAULT_NUM_LAYERS,
    ):
        """
        Initialize the Path GNN encoder.

        :param obs_space: Observation space containing graph structure
        :type obs_space: spaces.Dict
        :param emb_dim: Embedding dimension for each layer
        :type emb_dim: int
        :param gnn_type: Type of graph convolution ('gat', 'sage', 'graphconv')
        :type gnn_type: str
        :param layers: Number of convolution layers
        :type layers: int
        :raises ValueError: If gnn_type is not recognized
        """
        super().__init__()

        # Map convolution types
        convolution_mapping = {
            "gat": GATv2Conv,
            "sage": SAGEConv,
            "graphconv": GraphConv,
        }

        if gnn_type not in convolution_mapping:
            raise ValueError(
                f"Unknown GNN type: {gnn_type}. "
                f"Valid types: {list(convolution_mapping.keys())}"
            )

        convolution_class = convolution_mapping[gnn_type]
        # Assert shape exists and has expected dimensions for type safety
        x_shape = obs_space["x"].shape
        assert x_shape is not None and len(x_shape) >= 2
        input_dimension = x_shape[1]

        # Build convolution layers
        self.convolution_layers = torch.nn.ModuleList(
            [
                convolution_class(
                    input_dimension if layer_idx == 0 else emb_dim, emb_dim
                )
                for layer_idx in range(layers)
            ]
        )

        # Readout layer
        self.readout_layer = torch.nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        path_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward propagation through the encoder.

        :param node_features: Node feature tensor [num_nodes, feature_dim]
        :type node_features: torch.Tensor
        :param edge_index: Edge connectivity [2, num_edges]
        :type edge_index: torch.Tensor
        :param path_masks: Path selection masks [num_paths, num_edges]
        :type path_masks: torch.Tensor
        :return: Flattened path embeddings
        :rtype: torch.Tensor
        """
        # Process through convolution layers
        node_embeddings = node_features
        for convolution_layer in self.convolution_layers:
            node_embeddings = convolution_layer(node_embeddings, edge_index).relu()

        # Compute edge embeddings
        source_indices, destination_indices = edge_index
        edge_embeddings = (
            node_embeddings[source_indices] + node_embeddings[destination_indices]
        ) * EDGE_EMBEDDING_SCALE_FACTOR

        # Aggregate path embeddings
        path_embeddings = path_masks @ edge_embeddings

        # Apply readout and flatten
        readout_result: torch.Tensor = self.readout_layer(path_embeddings)
        return readout_result.flatten()


class CachedPathGNN(BaseGraphFeatureExtractor):
    """
    Cached GNN feature extractor for efficient inference.

    This extractor uses pre-computed embeddings instead of processing
    the graph at each forward pass, significantly improving inference speed
    for static graph structures.
    """

    def __init__(
        self,
        obs_space: spaces.Dict,  # Note: kept for backward compatibility
        cached_embedding: torch.Tensor,
    ):
        """
        Initialize the cached Path GNN feature extractor.

        :param obs_space: Observation space (used for compatibility)
        :type obs_space: spaces.Dict
        :param cached_embedding: Pre-computed embedding tensor
        :type cached_embedding: torch.Tensor
        """
        super().__init__(obs_space, features_dim=cached_embedding.numel())

        # Register cached embedding as a buffer (not updated during training)
        self.register_buffer("cached_embedding", cached_embedding)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return the cached embedding, handling batch dimensions appropriately.

        :param observation: Graph observation (only used to determine batch size)
        :type observation: dict[str, torch.Tensor]
        :return: Cached embeddings repeated for the batch
        :rtype: torch.Tensor
        """
        # Determine batch size from input
        node_features = observation["x"]
        if node_features.dim() == 3:
            batch_size = node_features.shape[0]
        else:
            batch_size = 1

        # Return cached embedding repeated for each sample in batch
        # Access the cached embedding buffer and ensure it's a tensor
        cached_emb = self.cached_embedding
        assert isinstance(cached_emb, torch.Tensor)
        return cached_emb.unsqueeze(0).repeat(batch_size, 1)
