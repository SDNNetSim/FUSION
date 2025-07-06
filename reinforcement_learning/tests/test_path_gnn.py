"""Unit tests for reinforcement_learning.feat_extrs.path_gnn."""

from types import SimpleNamespace
from unittest import TestCase, mock

import torch
from reinforcement_learning.feat_extrs import path_gnn

# ---------------------------- helpers ---------------------------------
EMB_DIM = 4
FEAT_DIM = 3
N_NODES = 3
N_EDGES = 2
N_PATHS = 2


class _DummyConv(torch.nn.Module):
    """Lightweight replacement for GNN conv layers."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, _edge_index):  # pylint: disable=arguments-differ
        """
        Forward propagation.
        """
        return self.lin(x)


def _make_obs(batch=1):
    """Return fake observation dict."""
    x = torch.randn(batch, N_NODES, FEAT_DIM) if batch > 1 else \
        torch.randn(N_NODES, FEAT_DIM)
    ei = torch.tensor([[0, 1], [1, 2]])  # shape (2, 2)
    ei = ei.repeat(batch, 1, 1) if batch > 1 else ei
    masks = torch.ones(batch, N_PATHS, N_EDGES) if batch > 1 else \
        torch.ones(N_PATHS, N_EDGES)
    return {"x": x, "edge_index": ei, "path_masks": masks}


def _make_space():
    """Return minimal obs_space stub with shape attributes."""
    return {
        "x": SimpleNamespace(shape=(N_NODES, FEAT_DIM)),
        "edge_index": SimpleNamespace(shape=(2, N_EDGES)),
        "path_masks": SimpleNamespace(shape=(N_PATHS, N_EDGES)),
    }


# ------------------------------ tests ---------------------------------
class TestPathGNN(TestCase):
    """Unit tests for PathGNN."""

    @mock.patch("reinforcement_learning.feat_extrs.path_gnn.GraphConv",
                new=_DummyConv)
    @mock.patch("reinforcement_learning.feat_extrs.path_gnn.SAGEConv",
                new=_DummyConv)
    @mock.patch("reinforcement_learning.feat_extrs.path_gnn.GATv2Conv",
                new=_DummyConv)
    def test_init_builds_correct_number_of_layers(self):
        """__init__ stores requested count of GNN layers."""
        model = path_gnn.PathGNN(
            obs_space=_make_space(),
            emb_dim=EMB_DIM,
            gnn_type="sage",
            layers=3,
        )
        self.assertEqual(len(model.convs), 3)
        self.assertIsInstance(model.convs[0], _DummyConv)

    @mock.patch("reinforcement_learning.feat_extrs.path_gnn.GATv2Conv",
                new=_DummyConv)
    def test_forward_single_graph_output_shape(self):
        """forward returns (1, paths*emb_dim) for single graph."""
        model = path_gnn.PathGNN(
            obs_space=_make_space(),
            emb_dim=EMB_DIM,
            gnn_type="gat",
            layers=1,
        )
        out = model(_make_obs(batch=1))
        self.assertEqual(out.shape, (1, N_PATHS * EMB_DIM))

    @mock.patch("reinforcement_learning.feat_extrs.path_gnn.GraphConv",
                new=_DummyConv)
    def test_forward_batch_graph_output_shape(self):
        """forward stacks outputs for batched graphs."""
        model = path_gnn.PathGNN(
            obs_space=_make_space(),
            emb_dim=EMB_DIM,
            gnn_type="graphconv",
            layers=2,
        )
        out = model(_make_obs(batch=3))
        self.assertEqual(out.shape, (3, N_PATHS * EMB_DIM))
