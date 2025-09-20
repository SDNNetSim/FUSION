"""
Feature extraction modules for reinforcement learning.

This package provides graph neural network-based feature extractors
for processing graph-structured observations in RL environments.
Available extractors include standard GNN, cached GNN, and graph transformer
architectures.
"""

from fusion.modules.rl.feat_extrs.base_feature_extractor import (
    BaseGraphFeatureExtractor,
)

# Import constants
from fusion.modules.rl.feat_extrs.constants import (
    CACHE_DIR,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_GNN_TYPE,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    EDGE_EMBEDDING_SCALE_FACTOR,
)
from fusion.modules.rl.feat_extrs.graphormer import GraphTransformerExtractor

# Import feature extractors
from fusion.modules.rl.feat_extrs.path_gnn import PathGNN
from fusion.modules.rl.feat_extrs.path_gnn_cached import CachedPathGNN, PathGNNEncoder

__all__ = [
    # Feature extractors
    "PathGNN",
    "CachedPathGNN",
    "PathGNNEncoder",
    "GraphTransformerExtractor",
    "BaseGraphFeatureExtractor",
    # Constants
    "CACHE_DIR",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_GNN_TYPE",
    "DEFAULT_NUM_HEADS",
    "EDGE_EMBEDDING_SCALE_FACTOR",
]
