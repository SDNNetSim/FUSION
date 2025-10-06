# Feature Extractors Module

This module provides graph neural network (GNN) based feature extractors for processing graph-structured observations in reinforcement learning environments.

## Overview

The feature extractors transform variable-sized graph observations into fixed-size feature vectors suitable for standard RL algorithms. All extractors are compatible with StableBaselines3.

## Available Feature Extractors

### PathGNN
Standard GNN feature extractor that processes graphs through multiple convolution layers.

**Usage:**
```python
from fusion.modules.rl.feat_extrs import PathGNN

feature_extractor = PathGNN(
    obs_space=env.observation_space,
    emb_dim=64,  # Embedding dimension
    gnn_type="gat",  # Options: 'gat', 'sage', 'graphconv'
    layers=2  # Number of convolution layers
)
```

### CachedPathGNN
Optimized feature extractor that uses pre-computed embeddings for static graphs, significantly improving inference speed.

**Usage:**
```python
from fusion.modules.rl.feat_extrs import PathGNNEncoder, CachedPathGNN

# First, create and cache the embedding
encoder = PathGNNEncoder(obs_space, emb_dim=64, gnn_type="gat", layers=2)
embedding = encoder(x, edge_index, path_masks)

# Then use the cached version
feature_extractor = CachedPathGNN(
    obs_space=env.observation_space,
    cached_embedding=embedding
)
```

### GraphTransformerExtractor
Transformer-based GNN that uses multi-head attention for graph processing.

**Usage:**
```python
from fusion.modules.rl.feat_extrs import GraphTransformerExtractor

feature_extractor = GraphTransformerExtractor(
    obs_space=env.observation_space,
    emb_dim=64,  # Must be divisible by heads
    heads=4,  # Number of attention heads
    layers=2  # Number of transformer layers
)
```

## Input Format

All feature extractors expect observations as dictionaries containing:
- `x`: Node features tensor `[batch_size, num_nodes, features]` or `[num_nodes, features]`
- `edge_index`: Edge connectivity `[batch_size, 2, num_edges]` or `[2, num_edges]`
- `path_masks`: Path selection masks `[batch_size, num_paths, num_edges]` or `[num_paths, num_edges]`

## Architecture Details

### Base Architecture
All extractors inherit from `BaseGraphFeatureExtractor` which provides:
- Consistent batch dimension handling
- Edge embedding computation
- Path embedding aggregation

### Processing Pipeline
1. **Node Processing**: Apply graph convolutions to node features
2. **Edge Embeddings**: Compute edge representations from connected nodes
3. **Path Aggregation**: Use path masks to aggregate edge embeddings
4. **Feature Output**: Flatten to fixed-size feature vector

## Constants and Configuration

Key constants are defined in `constants.py`:
- `DEFAULT_EMBEDDING_DIMENSION`: 64
- `DEFAULT_NUM_LAYERS`: 2
- `DEFAULT_GNN_TYPE`: "gat"
- `DEFAULT_NUM_HEADS`: 4 (for GraphTransformer)
- `EDGE_EMBEDDING_SCALE_FACTOR`: 0.5

## Caching

The module supports caching GNN embeddings for static graphs:
- Cache directory: `FUSION/gnn_cached/`
- Use `PathGNNEncoder` to generate embeddings
- Use `CachedPathGNN` for efficient inference

## Integration with StableBaselines3

All extractors are compatible with SB3's policy networks:

```python
from stable_baselines3 import PPO

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": PathGNN,
        "features_extractor_kwargs": {
            "emb_dim": 64,
            "gnn_type": "gat",
            "layers": 2
        }
    }
)
```

## Performance Considerations

- **PathGNN**: Standard performance, recomputes embeddings each forward pass
- **CachedPathGNN**: Fastest for static graphs, uses pre-computed embeddings
- **GraphTransformerExtractor**: Higher computational cost due to attention mechanism

## TODO Items

- Add Optuna hyperparameter optimization support
- Implement additional GNN architectures (GIN, PNA)
- Add support for edge features
