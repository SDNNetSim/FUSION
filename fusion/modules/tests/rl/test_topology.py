"""Unit tests for fusion.modules.rl.utils.topology module."""

from unittest import mock

import networkx as nx

from fusion.modules.rl.utils import topology as topo


# ------------------------------------------------------------------ #
# helpers                                                             #
# ------------------------------------------------------------------ #
def _simple_graph() -> nx.Graph:
    """Return an undirected graph with three nodes, two edges."""
    g = nx.Graph()
    g.add_edge(2, 0)  # intentionally out-of-order to test id2idx sort
    g.add_edge(0, 1)
    return g


# ------------------------------------------------------------------ #
class TestConvertNetworkxTopo:
    """convert_networkx_topo output shapes and directed handling."""

    @mock.patch(
        "fusion.modules.rl.utils.topology.nx.edge_betweenness_centrality",
        return_value={(0, 1): 0.2, (2, 0): 0.3},
    )
    def test_directed_edges_are_duplicated(self, _mock_bet: mock.MagicMock) -> None:
        """With as_directed=True edges appear twice (u→v and v→u)."""
        ei, ea, nf, idx = topo.convert_networkx_topo(_simple_graph(), as_directed=True)

        assert ei.shape[0] == 2  # (2, E)
        assert ei.shape[1] == 4  # 2 original *2 directions
        assert ea.shape == (4, 1)  # one attr per edge
        assert nf.shape == (3, 1)  # 3 nodes, 1 feat each
        assert idx == {0: 0, 1: 1, 2: 2}  # sorted mapping

    @mock.patch(
        "fusion.modules.rl.utils.topology.nx.edge_betweenness_centrality",
        return_value={(0, 1): 0.2, (2, 0): 0.3},
    )
    def test_undirected_edges_not_duplicated(self, _mock_bet: mock.MagicMock) -> None:
        """With as_directed=False each edge added once."""
        ei, _, _, _ = topo.convert_networkx_topo(_simple_graph(), as_directed=False)
        assert ei.shape[1] == 2


# ------------------------------------------------------------------ #
class TestLoadTopologyFromGraph:
    """load_topology_from_graph delegates to convert_networkx_topo."""

    @mock.patch(
        "fusion.modules.rl.utils.topology.convert_networkx_topo",
        return_value=("ei", "ea", "nf", "idx"),
    )
    def test_load_calls_convert_with_kwargs(self, mock_conv: mock.MagicMock) -> None:
        """load_topology_from_graph simply forwards to convert helper."""
        g = _simple_graph()
        out = topo.load_topology_from_graph(g, as_directed=False)

        mock_conv.assert_called_once_with(g, as_directed=False)
        assert out == ("ei", "ea", "nf", "idx")
