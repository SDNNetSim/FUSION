Feature Extractor
==================

Before a model can learn, it needs something meaningful to learn from.
This module contains the feature extractors that process graph-topological
and request-specific information into neural-friendly embeddings.
We provide several extractors — GNNs, Transformer-based models, and even pointer networks —
each tailored to different architectures and training setups.
These are the perceptual systems of the agent — the eyes through which it views the network.

.. toctree::

    graphormer
    path_gnn
    path_gnn_cached
