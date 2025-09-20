"""
Constants for the feature extraction module.

This module defines directory paths and other constants used by
the feature extraction components.
"""

from pathlib import Path

# Directory for caching GNN embeddings
CACHE_DIR: Path = Path(__file__).resolve().parents[3] / "gnn_cached"
CACHE_DIR.mkdir(exist_ok=True)

# Default values for feature extractors
DEFAULT_EMBEDDING_DIMENSION: int = 64
DEFAULT_NUM_LAYERS: int = 2
DEFAULT_GNN_TYPE: str = "gat"
DEFAULT_NUM_HEADS: int = 4

# Scale factor for edge embedding computation
EDGE_EMBEDDING_SCALE_FACTOR: float = 0.5
