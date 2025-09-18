# pylint: disable=protected-access

"""Unit tests for reinforcement_learning.algorithms.q_learning."""

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from fusion.modules.rl.algorithms import q_learning as ql
from fusion.modules.rl.errors import AlgorithmNotFoundError


# -------------------------- helpers -----------------------------------
def _mk_engine(**overrides):
    base = dict(
        epsilon_start=0.5,
        path_levels=1,
        cores_per_link=2,
        cong_cutoff=0.7,
        gamma=0.9,
        save_step=1,
        max_iters=5,
        num_requests=1,
        network="net",
        date="d",
        sim_start="t0",
        erlang=30,
        path_algorithm="q_learning",
        topology=mock.MagicMock(),  # unused due to patching
    )
    base.update(overrides)
    return base


def _mk_rl(num_nodes=2, k_paths=2):
    return SimpleNamespace(
        num_nodes=num_nodes,
        k_paths=k_paths,
        source=0,
        destination=1,
        chosen_path_index=0,
        paths_list=None,
        cores_list=None,
    )


def _new_agent():
    """Return QLearning with heavy ops patched away."""
    with mock.patch.object(ql.QLearning, "_populate_q_tables"):
        return ql.QLearning(_mk_rl(), _mk_engine())


# ------------------------- _create_*_matrix ---------------------------
class TestMatrixCreation(TestCase):
    """Matrix creation shape and dtype."""

    def test_create_routes_matrix_shape_dtype(self):
        """Routes matrix has expected shape and dtype."""
        agent = _new_agent()
        mat = agent._create_routes_matrix()
        self.assertEqual(mat.shape, (2, 2, 2, 1))
        self.assertEqual(mat.dtype.names, ("path", "q_value"))

    def test_create_cores_matrix_shape_dtype(self):
        """Cores matrix has expected shape and dtype."""
        agent = _new_agent()
        mat = agent._create_cores_matrix()
        self.assertEqual(mat.shape, (2, 2, 2, 2, 1))
        self.assertEqual(mat.dtype.names, ("path", "core_action", "q_value"))


# --------------------------- get_max_curr_q ---------------------------
class TestGetMaxCurrQ(TestCase):
    """Selecting max current Q-value."""

    def setUp(self):
        self.agent = _new_agent()
        # Dummy paths / cores
        self.agent.rl_props.paths_list = ["p0", "p1"]
        self.agent.rl_props.cores_list = [0, 1]
        # Fill route matrix values
        routes = self.agent.props.routes_matrix
        routes[0, 1, 0, 0] = ("p0", 0.1)
        routes[0, 1, 1, 0] = ("p1", 0.5)
        # Fill cores matrix values
        cores = self.agent.props.cores_matrix
        cores[0, 1, 0, 0, 0] = ("p0", 0, 0.2)
        cores[0, 1, 0, 1, 0] = ("p0", 1, 0.8)

    def test_max_curr_q_routes(self):
        """Returns path with highest Q in routes matrix."""
        cong_list = [(0, None, 0), (1, None, 0)]
        idx, obj = self.agent.get_max_curr_q(cong_list, "routes_matrix")
        self.assertEqual(idx, 1)
        self.assertEqual(obj, "p1")

    def test_max_curr_q_cores(self):
        """Returns core with highest Q in cores matrix."""
        cong_list = [(0, None, 0), (1, None, 0)]
        self.agent.rl_props.chosen_path_index = 0
        idx, obj = self.agent.get_max_curr_q(cong_list, "cores_matrix")
        self.assertEqual(idx, 1)
        self.assertEqual(obj, 1)


# --------------------------- get_max_future_q -------------------------
class TestGetMaxFutureQ(TestCase):
    """Future-Q computation with congestion helpers."""

    @mock.patch("fusion.modules.rl.algorithms.q_learning.classify_cong",
                return_value=0)
    @mock.patch("fusion.modules.rl.algorithms.q_learning.find_path_cong",
                return_value=(0.4, None))
    def test_max_future_q_path(self, _cong, _classify):
        """Path mode returns correct Q from matrix."""
        agent = _new_agent()
        mat = np.array([(None, 0.33)], dtype=[("path", "O"), ("q_value", "f8")])
        val = agent.get_max_future_q("p", {}, mat, flag="path")
        self.assertAlmostEqual(val, 0.33)


# ----------------------- _convert_q_tables_to_dict --------------------
class TestConvertQTables(TestCase):
    """Conversion of routes matrix to JSON-ready dict."""

    def test_convert_routes_returns_expected(self):
        """Returns average Q per path pair."""
        agent = _new_agent()
        agent.props.routes_matrix[0, 1, 0, 0] = ("p0", 0.4)
        expected = {"(0, 1)": [0.4, 0.0], "(1, 0)": [0.0, 0.0]}
        result = agent._convert_q_tables_to_dict("routes")
        self.assertEqual(result, expected)

    def test_convert_cores_raises_not_implemented(self):
        """Passing 'cores' raises AlgorithmNotFoundError."""
        agent = _new_agent()
        with self.assertRaises(AlgorithmNotFoundError):
            agent._convert_q_tables_to_dict("cores")


# ------------------------------ save_model ----------------------------
class TestSaveModel(TestCase):
    """save_model file outputs."""

    @mock.patch("fusion.modules.rl.algorithms.q_learning.json.dump")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("fusion.modules.rl.algorithms.q_learning.np.save")
    @mock.patch("fusion.modules.rl.algorithms.q_learning.create_dir")
    def test_save_model_writes_files(
            self, mock_dir, mock_npsave, mock_open_fn, mock_dump
    ):
        """save_model calls create_dir, np.save, and json.dump."""
        agent = _new_agent()
        agent.iteration = 0
        agent.rewards_stats_dict = {"average": np.array([1.0])}
        with mock.patch.object(agent, "_convert_q_tables_to_dict",
                               return_value={"k": [1]}):
            agent.save_model(trial=0)

        mock_dir.assert_called_once()
        mock_npsave.assert_called_once()
        mock_open_fn.assert_called_once()
        mock_dump.assert_called_once()
