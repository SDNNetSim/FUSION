.. _rl-feat-extrs:

=========================
Feature Extractors Module
=========================

.. warning::

   **Status: BETA**

   This module is currently in **BETA** and is actively being developed.
   The API and functionality may change in future releases (v6.X).

.. admonition:: At a Glance
   :class: tip

   :Purpose: GNN-based feature extraction for RL observations
   :Location: ``fusion/modules/rl/feat_extrs/``
   :Key Files: ``path_gnn.py``, ``graphormer.py``, ``base_feature_extractor.py``
   :Prerequisites: PyTorch, PyTorch Geometric, Stable-Baselines3

Overview
========

The feature extractors module transforms variable-sized graph observations into
fixed-size feature vectors for reinforcement learning agents. This is essential
because standard RL algorithms (PPO, DQN, etc.) require fixed-dimensional inputs,
but optical network state is naturally represented as a graph with varying topology.

**Why Use Feature Extractors?**

- **Graph-to-Vector Transformation**: Converts network topology and state into
  dense vector representations suitable for neural network policies
- **Topology-Aware Learning**: GNN architectures capture structural relationships
  between nodes and edges in the network
- **SB3 Compatibility**: All extractors integrate seamlessly with Stable-Baselines3
  policy networks

**Current Extractors:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Extractor
     - Architecture
     - Use Case
   * - ``PathGNN``
     - GNN (GAT/SAGE/GraphConv)
     - Standard path-based feature extraction
   * - ``CachedPathGNN``
     - Pre-computed embeddings
     - Static graphs, fast inference
   * - ``GraphTransformerExtractor``
     - Transformer with attention
     - Experimental, attention-based processing

Processing Pipeline
===================

All feature extractors follow a common pipeline:

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   | Graph Observation|---->| GNN Convolutions |---->| Node Embeddings  |
   | (x, edge_index,  |     | (GAT/SAGE/Conv)  |     | [N, emb_dim]     |
   |  path_masks)     |     +------------------+     +------------------+
   +------------------+                                      |
                                                             v
   +------------------+     +------------------+     +------------------+
   | Feature Vector   |<----| Path Aggregation |<----| Edge Embeddings  |
   | [batch, feat_dim]|     | (mask @ edges)   |     | [E, emb_dim]     |
   +------------------+     +------------------+     +------------------+

**Pipeline Steps:**

1. **Node Processing**: Graph convolution layers process node features
2. **Edge Embeddings**: Computed from source and destination node embeddings
3. **Path Aggregation**: Path masks select and aggregate edge embeddings
4. **Flattening**: Output flattened to fixed-size vector for RL policy

Input/Output Format
===================

Input Observation
-----------------

Feature extractors expect a dictionary observation with:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Key
     - Shape
     - Description
   * - ``x``
     - ``[batch, N, F]`` or ``[N, F]``
     - Node features (N nodes, F features per node)
   * - ``edge_index``
     - ``[batch, 2, E]`` or ``[2, E]``
     - Edge connectivity (source/dest indices)
   * - ``path_masks``
     - ``[batch, K, E]`` or ``[K, E]``
     - Binary masks selecting edges for K paths

Output
------

All extractors output a tensor of shape ``[batch_size, features_dim]`` where
``features_dim = emb_dim * num_paths``.

Quick Start
===========

Using PathGNN with Stable-Baselines3
------------------------------------

.. code-block:: python

   from stable_baselines3 import PPO
   from fusion.modules.rl.feat_extrs import PathGNN

   # Create environment (provides observation space)
   env = make_your_env()

   # Configure PPO with PathGNN feature extractor
   model = PPO(
       "MultiInputPolicy",
       env,
       policy_kwargs={
           "features_extractor_class": PathGNN,
           "features_extractor_kwargs": {
               "emb_dim": 64,
               "gnn_type": "gat",   # Options: 'gat', 'sage', 'graphconv'
               "layers": 2,
           }
       }
   )

   model.learn(total_timesteps=10000)

Direct Usage
------------

.. code-block:: python

   from fusion.modules.rl.feat_extrs import PathGNN

   # Create extractor
   extractor = PathGNN(
       obs_space=env.observation_space,
       emb_dim=64,
       gnn_type="gat",
       layers=2,
   )

   # Process observation
   observation = env.reset()[0]
   features = extractor(observation)  # [1, features_dim]

Available GNN Types
===================

The ``PathGNN`` extractor supports multiple GNN convolution types via the
``gnn_type`` parameter:

GAT (Graph Attention Network)
-----------------------------

:Config Value: ``gat``
:PyG Class: ``GATv2Conv``
:Description: Uses attention mechanisms to weight neighbor contributions

GAT learns attention weights for each edge, allowing the model to focus on
more important connections. This is often effective for heterogeneous graphs
where edge importance varies.

.. code-block:: ini

   [rl_settings]
   feature_extractor = path_gnn
   gnn_type = gat

SAGE (GraphSAGE)
----------------

:Config Value: ``sage``
:PyG Class: ``SAGEConv``
:Description: Samples and aggregates features from neighbors

GraphSAGE uses a sampling-based approach that scales well to larger graphs.
It aggregates neighbor features using mean, max, or LSTM aggregators.

.. code-block:: ini

   [rl_settings]
   feature_extractor = path_gnn
   gnn_type = sage

GraphConv
---------

:Config Value: ``graphconv``
:PyG Class: ``GraphConv``
:Description: Standard graph convolution with sum aggregation

A simpler architecture that sums neighbor features with learnable weights.
Often faster than attention-based methods with comparable performance.

.. code-block:: ini

   [rl_settings]
   feature_extractor = path_gnn
   gnn_type = graph_conv

GraphTransformer (Experimental)
-------------------------------

The ``GraphTransformerExtractor`` uses Transformer-style multi-head attention
for graph processing. This is an experimental approach exploring attention
mechanisms for network state representation.

.. code-block:: python

   from fusion.modules.rl.feat_extrs import GraphTransformerExtractor

   extractor = GraphTransformerExtractor(
       obs_space=env.observation_space,
       emb_dim=64,   # Must be divisible by heads
       heads=4,      # Number of attention heads
       layers=2,
   )

.. note::

   The GraphTransformer extractor has higher computational cost due to the
   attention mechanism. It is currently experimental and may be refined in
   future versions.

Configuration Reference
=======================

INI File Settings
-----------------

Configure feature extractors in the ``[rl_settings]`` section:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``feature_extractor``
     - ``path_gnn``
     - Extractor type (``path_gnn``, ``mlp``)
   * - ``gnn_type``
     - ``gat``
     - GNN architecture (``gat``, ``sage``, ``graphconv``)
   * - ``emb_dim``
     - ``64``
     - Embedding dimension for GNN layers
   * - ``layers``
     - ``2``
     - Number of GNN convolution layers
   * - ``heads``
     - ``4``
     - Attention heads (GraphTransformer only)

Example Configuration
---------------------

.. code-block:: ini

   [rl_settings]
   # Algorithm selection
   path_algorithm = ppo

   # Feature extractor configuration
   feature_extractor = path_gnn
   gnn_type = gat
   emb_dim = 64
   layers = 2

   # Training parameters
   is_training = True
   device = cuda

Module Constants
----------------

Default values defined in ``constants.py``:

.. code-block:: python

   DEFAULT_EMBEDDING_DIMENSION = 64
   DEFAULT_NUM_LAYERS = 2
   DEFAULT_GNN_TYPE = "gat"
   DEFAULT_NUM_HEADS = 4
   EDGE_EMBEDDING_SCALE_FACTOR = 0.5

Architecture Details
====================

BaseGraphFeatureExtractor
-------------------------

All extractors inherit from ``BaseGraphFeatureExtractor``, which extends
SB3's ``BaseFeaturesExtractor`` with graph-specific utilities:

.. code-block:: python

   from fusion.modules.rl.feat_extrs import BaseGraphFeatureExtractor

   class CustomExtractor(BaseGraphFeatureExtractor):
       def __init__(self, obs_space, features_dim):
           super().__init__(obs_space, features_dim)
           # Custom initialization

       def forward(self, observation):
           # Use inherited utilities
           x, edge_index, masks, batch_size = self._process_batch_dimensions(
               observation["x"],
               observation["edge_index"],
               observation["path_masks"],
           )

           # Process graph...
           node_emb = self.process_nodes(x, edge_index)

           # Compute edge and path embeddings
           edge_emb = self._compute_edge_embeddings(node_emb, edge_index)
           path_emb = self._compute_path_embeddings(edge_emb, masks)

           return path_emb.flatten().unsqueeze(0)

**Inherited Methods:**

- ``_process_batch_dimensions()``: Normalizes batch dimensions
- ``_compute_edge_embeddings()``: Creates edge representations from nodes
- ``_compute_path_embeddings()``: Aggregates edges according to path masks

Caching for Static Graphs
-------------------------

For static network topologies, use ``CachedPathGNN`` to avoid redundant
computation:

.. code-block:: python

   from fusion.modules.rl.feat_extrs import PathGNNEncoder, CachedPathGNN

   # Pre-compute embeddings once
   encoder = PathGNNEncoder(obs_space, emb_dim=64, gnn_type="gat", layers=2)
   cached_embedding = encoder(x, edge_index, path_masks)

   # Use cached version for fast inference
   extractor = CachedPathGNN(
       obs_space=obs_space,
       cached_embedding=cached_embedding,
   )

Performance Considerations
==========================

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Extractor
     - Speed
     - When to Use
   * - ``PathGNN``
     - Standard
     - Dynamic graphs, general use
   * - ``CachedPathGNN``
     - Fastest
     - Static topology, inference-heavy workloads
   * - ``GraphTransformerExtractor``
     - Slowest
     - Experimental, attention analysis

Future Development (v6.X)
=========================

The following enhancements are planned:

- **Optuna Integration**: Hyperparameter optimization for GNN architecture
  (embedding dimension, layer count, GNN type selection)
- **Additional Architectures**: GIN (Graph Isomorphism Network), PNA
  (Principal Neighbourhood Aggregation)
- **Edge Features**: Support for edge attributes in convolution layers

See ``TODO.md`` in the module directory for the current development roadmap.

File Reference
==============

.. code-block:: text

   fusion/modules/rl/feat_extrs/
   |-- __init__.py                  # Public exports
   |-- README.md                    # Module documentation
   |-- TODO.md                      # Development roadmap (BETA status)
   |-- constants.py                 # Default values and paths
   |-- base_feature_extractor.py    # BaseGraphFeatureExtractor
   |-- path_gnn.py                  # PathGNN extractor
   |-- path_gnn_cached.py           # CachedPathGNN, PathGNNEncoder
   `-- graphormer.py                # GraphTransformerExtractor

**What to Import:**

.. code-block:: python

   # Feature extractors
   from fusion.modules.rl.feat_extrs import (
       PathGNN,
       CachedPathGNN,
       PathGNNEncoder,
       GraphTransformerExtractor,
       BaseGraphFeatureExtractor,
   )

   # Constants
   from fusion.modules.rl.feat_extrs import (
       DEFAULT_EMBEDDING_DIMENSION,
       DEFAULT_NUM_LAYERS,
       DEFAULT_GNN_TYPE,
       DEFAULT_NUM_HEADS,
   )

Related Documentation
=====================

- :ref:`rl-module` - Parent RL module documentation
- :ref:`rl-algorithms` - RL algorithms that use feature extractors
- :ref:`rl-environments` - Environments providing graph observations

.. seealso::

   - `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/>`_ - GNN library
   - `Stable-Baselines3 Custom Features <https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html>`_
