"""Unit tests for reinforcement_learning.feat_extrs.graphormer."""

from types import SimpleNamespace
from unittest import TestCase, mock

import torch
from reinforcement_learning.feat_extrs import graphormer

# --------------------------- constants --------------------------------
EMB_DIM = 6
HEADS = 2  # EMB_DIM must be divisible by HEADS
LAYERS = 3
N_NODES = 4
N_EDGES = 3
N_PATHS = 2


# ---------------------------- stubs -----------------------------------
class _DummyConv(torch.nn.Module):
    """Lightweight stand-in for TransformerConv."""

    def __init__(self, in_channels, out_channels, heads, concat=True):
        super().__init__()
        self.heads = heads
        self.lin = torch.nn.Linear(in_channels,
                                   out_channels * heads,
                                   bias=False)

    def forward(self, x, _edge_index):  # pylint: disable=arguments-differ
        """
        Forward propagation.
        """
        return self.lin(x)


def _obs_space():
    """Return minimal Gym-style obs_space stub."""
    return {
        "x": SimpleNamespace(shape=(N_NODES, EMB_DIM)),  # (N, F)
        "edge_index": SimpleNamespace(shape=(2, N_EDGES)),
        "path_masks": SimpleNamespace(shape=(N_PATHS, N_EDGES)),
    }


def _make_obs(batch=1):
    """Create a dummy observation dict."""
    if batch == 1:
        x = torch.randn(N_NODES, EMB_DIM)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]])  # (2, E)
        masks = torch.ones(N_PATHS, N_EDGES)
    else:
        x = torch.randn(batch, N_NODES, EMB_DIM)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]]).repeat(batch, 1, 1)
        masks = torch.ones(batch, N_PATHS, N_EDGES)
    return {"x": x, "edge_index": ei, "path_masks": masks}


# ------------------------------ tests ---------------------------------
class TestGraphTransformerExtractor(TestCase):
    """GraphTransformerExtractor core behaviour."""

    @mock.patch("reinforcement_learning.feat_extrs.graphormer.TransformerConv",
                new=_DummyConv)
    def test_init_builds_correct_layer_dims(self):
        """__init__ creates L layers with expected input dims."""
        obs_space = _obs_space()
        model = graphormer.GraphTransformerExtractor(
            obs_space=obs_space,
            emb_dim=EMB_DIM,
            heads=HEADS,
            layers=LAYERS,
        )
        self.assertEqual(len(model.convs), LAYERS)
        first_in = model.convs[0].lin.in_features
        second_in = model.convs[1].lin.in_features
        self.assertEqual(first_in, EMB_DIM)  # in_dim
        self.assertEqual(second_in, EMB_DIM)  # heads*out_per_head

    @mock.patch("reinforcement_learning.feat_extrs.graphormer.TransformerConv",
                new=_DummyConv)
    def test_forward_single_graph_output_shape(self):
        """forward returns (1, paths*emb_dim) for single graph."""
        model = graphormer.GraphTransformerExtractor(
            obs_space=_obs_space(),
            emb_dim=EMB_DIM,
            heads=HEADS,
            layers=1,
        )
        out = model(_make_obs(batch=1))
        self.assertEqual(out.shape, (1, N_PATHS * EMB_DIM))

    @mock.patch("reinforcement_learning.feat_extrs.graphormer.TransformerConv",
                new=_DummyConv)
    def test_forward_batch_output_shape(self):
        """forward stacks outputs for batched graphs."""
        model = graphormer.GraphTransformerExtractor(
            obs_space=_obs_space(),
            emb_dim=EMB_DIM,
            heads=HEADS,
            layers=2,
        )
        out = model(_make_obs(batch=4))
        self.assertEqual(out.shape, (4, N_PATHS * EMB_DIM))
