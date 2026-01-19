"""
Path-based Graph Neural Network feature extractor.

This module implements a GNN feature extractor that processes graph observations
through multiple convolution layers to extract path-based features for
reinforcement learning agents.
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
)

# TODO: (v6.X) Integrate PathGNN hyperparameters with Optuna search space


class PathGNN(BaseGraphFeatureExtractor):
    """
    Custom PathGNN feature extraction algorithm integrated with StableBaselines3.

    This feature extractor processes graph-structured observations through
    multiple graph convolution layers to produce fixed-size feature vectors
    for downstream RL algorithms.
    """

    def __init__(
        self,
        obs_space: spaces.Dict,  # Note: kept for backward compatibility
        emb_dim: int = DEFAULT_EMBEDDING_DIMENSION,  # Kept for compat
        gnn_type: str = DEFAULT_GNN_TYPE,
        layers: int = DEFAULT_NUM_LAYERS,
    ):
        """
        Initialize the Path GNN feature extractor.

        :param obs_space: Observation space containing graph components:
            - 'x': Node features [num_nodes, feature_dim]
            - 'edge_index': Edge connectivity [2, num_edges]
            - 'path_masks': Path selection masks [num_paths, num_edges]
        :type obs_space: spaces.Dict
        :param emb_dim: Embedding dimension for each convolution layer
        :type emb_dim: int
        :param gnn_type: Type of graph convolution ('gat', 'sage', 'graphconv')
        :type gnn_type: str
        :param layers: Number of graph convolution layers
        :type layers: int
        :raises ValueError: If gnn_type is not recognized
        """
        # Calculate output dimension based on number of paths and embedding dimension
        path_masks_shape = obs_space["path_masks"].shape
        assert path_masks_shape is not None and len(path_masks_shape) >= 2
        num_paths = path_masks_shape[0]
        features_dimension = emb_dim * num_paths
        super().__init__(obs_space, features_dimension)

        # Map string names to convolution classes
        convolution_type_mapping = {
            "gat": GATv2Conv,
            "sage": SAGEConv,
            "graphconv": GraphConv,
        }

        if gnn_type not in convolution_type_mapping:
            raise ValueError(f"Unknown GNN type: {gnn_type}. Valid types: {list(convolution_type_mapping.keys())}")

        selected_convolution_class = convolution_type_mapping[gnn_type]
        x_shape = obs_space["x"].shape
        assert x_shape is not None and len(x_shape) >= 2
        input_dimension = x_shape[1]

        # Create convolution layers
        self.convolution_layers = torch.nn.ModuleList(
            [selected_convolution_class(input_dimension if layer_idx == 0 else emb_dim, emb_dim) for layer_idx in range(layers)]
        )

        # Readout layer for final transformation
        self.readout_layer = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert graph observation into fixed-size feature vector.

        Processes the graph through convolution layers, computes edge embeddings,
        aggregates them according to path masks, and produces a flattened
        feature vector for each sample in the batch.

        :param observation: Dictionary containing:
            - 'x': Node features
                [batch_size, num_nodes, features] or [num_nodes, features]
            - 'edge_index': Edge indices [batch_size, 2, num_edges] or [2, num_edges]
            - 'path_masks': Path masks
                [batch_size, num_paths, num_edges] or [num_paths, num_edges]
        :type observation: dict[str, torch.Tensor]
        :return: Feature vector [batch_size, feature_dim] or [1, feature_dim]
        :rtype: torch.Tensor
        """
        # Extract components from observation
        node_features_list = observation["x"]
        edge_index = observation["edge_index"].long()
        path_masks_list = observation["path_masks"]

        # Process batch dimensions
        node_features_list, edge_index, path_masks_list, _ = self._process_batch_dimensions(node_features_list, edge_index, path_masks_list)

        # Process each sample in the batch
        batch_outputs: list[torch.Tensor] = []
        actual_batch_size = node_features_list.size(0)

        for batch_idx in range(actual_batch_size):
            # Extract batch sample
            node_features_batch = node_features_list[batch_idx]
            edge_index_batch = edge_index[batch_idx]
            path_masks_batch = path_masks_list[batch_idx]

            # Process through convolution layers
            node_embeddings = node_features_batch
            for convolution_layer in self.convolution_layers:
                node_embeddings = convolution_layer(node_embeddings, edge_index_batch).relu()

            # Compute edge embeddings
            edge_embeddings = self._compute_edge_embeddings(node_embeddings, edge_index_batch)

            # Aggregate edge embeddings according to path masks
            path_embeddings = self._compute_path_embeddings(edge_embeddings, path_masks_batch)

            # Apply readout layer and flatten
            path_vectors = self.readout_layer(path_embeddings).flatten()
            batch_outputs.append(path_vectors)

        return torch.stack(batch_outputs, dim=0)
