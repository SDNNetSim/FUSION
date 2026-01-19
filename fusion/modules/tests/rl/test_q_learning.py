# pylint: disable=protected-access

"""Unit tests for fusion.modules.rl.algorithms/q_learning module."""

from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest

from fusion.modules.rl.algorithms import q_learning as ql
from fusion.modules.rl.errors import AlgorithmNotFoundError


# -------------------------- helpers -----------------------------------
def _mk_engine(**overrides: Any) -> dict[str, Any]:
    base = {
        "epsilon_start": 0.5,
        "path_levels": 1,
        "cores_per_link": 2,
        "cong_cutoff": 0.7,
        "gamma": 0.9,
        "save_step": 1,
        "max_iters": 5,
        "num_requests": 1,
        "network": "net",
        "date": "d",
        "sim_start": "t0",
        "erlang": 30,
        "path_algorithm": "q_learning",
        "topology": mock.MagicMock(),  # unused due to patching
    }
    base.update(overrides)
    return base


def _mk_rl(num_nodes: int = 2, k_paths: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        num_nodes=num_nodes,
        k_paths=k_paths,
        source=0,
        destination=1,
        chosen_path_index=0,
        paths_list=None,
        cores_list=None,
    )


def _new_agent() -> ql.QLearning:
    """Return QLearning with heavy ops patched away."""
    with mock.patch.object(ql.QLearning, "_populate_q_tables"):
        return ql.QLearning(_mk_rl(), _mk_engine())


# ------------------------- _create_*_matrix ---------------------------
class TestMatrixCreation:
    """Matrix creation shape and dtype."""

    def test_create_routes_matrix_shape_dtype(self) -> None:
        """Routes matrix has expected shape and dtype."""
        agent = _new_agent()
        mat = agent._create_routes_matrix()
        assert mat.shape == (2, 2, 2, 1)
        assert mat.dtype.names == ("path", "q_value")

    def test_create_cores_matrix_shape_dtype(self) -> None:
        """Cores matrix has expected shape and dtype."""
        agent = _new_agent()
        mat = agent._create_cores_matrix()
        assert mat.shape == (2, 2, 2, 2, 1)
        assert mat.dtype.names == ("path", "core_action", "q_value")


# --------------------------- get_max_curr_q ---------------------------
class TestGetMaxCurrQ:
    """Selecting max current Q-value."""

    def setup_method(self) -> None:
        self.agent = _new_agent()
        # Dummy paths / cores
        self.agent.rl_props.paths_list = ["p0", "p1"]  # type: ignore[attr-defined]
        self.agent.rl_props.cores_list = [0, 1]  # type: ignore[attr-defined]
        # Fill route matrix values
        routes = self.agent.props.routes_matrix
        assert routes is not None
        routes[0, 1, 0, 0] = ("p0", 0.1)
        routes[0, 1, 1, 0] = ("p1", 0.5)
        # Fill cores matrix values
        cores = self.agent.props.cores_matrix
        assert cores is not None
        cores[0, 1, 0, 0, 0] = ("p0", 0, 0.2)
        cores[0, 1, 0, 1, 0] = ("p0", 1, 0.8)

    def test_max_curr_q_routes(self) -> None:
        """Returns path with highest Q in routes matrix."""
        cong_list = [(0, None, 0), (1, None, 0)]
        idx, obj = self.agent.get_max_curr_q(cong_list, "routes_matrix")
        assert idx == 1
        assert obj == "p1"

    def test_max_curr_q_cores(self) -> None:
        """Returns core with highest Q in cores matrix."""
        cong_list = [(0, None, 0), (1, None, 0)]
        self.agent.rl_props.chosen_path_index = 0  # type: ignore[attr-defined]
        idx, obj = self.agent.get_max_curr_q(cong_list, "cores_matrix")
        assert idx == 1
        assert obj == 1


# --------------------------- get_max_future_q -------------------------
class TestGetMaxFutureQ:
    """Future-Q computation with congestion helpers."""

    @mock.patch(
        "fusion.modules.rl.algorithms.q_learning.classify_congestion", return_value=0
    )
    @mock.patch(
        "fusion.modules.rl.algorithms.q_learning.find_path_congestion",
        return_value=(0.4, None),
    )
    def test_max_future_q_path(
        self, _cong: mock.MagicMock, _classify: mock.MagicMock
    ) -> None:
        """Path mode returns correct Q from matrix."""
        agent = _new_agent()
        mat = np.array([(None, 0.33)], dtype=[("path", "O"), ("q_value", "f8")])
        val = agent.get_max_future_q("p", {}, mat, flag="path")
        assert val == pytest.approx(0.33)


# ----------------------- _convert_q_tables_to_dict --------------------
class TestConvertQTables:
    """Conversion of routes matrix to JSON-ready dict."""

    def test_convert_routes_returns_expected(self) -> None:
        """Returns average Q per path pair."""
        agent = _new_agent()
        routes = agent.props.routes_matrix
        assert routes is not None
        routes[0, 1, 0, 0] = ("p0", 0.4)
        expected = {"(0, 1)": [0.4, 0.0], "(1, 0)": [0.0, 0.0]}
        result = agent._convert_q_tables_to_dict("routes")
        assert result == expected

    def test_convert_cores_raises_not_implemented(self) -> None:
        """Passing 'cores' raises AlgorithmNotFoundError."""
        agent = _new_agent()
        with pytest.raises(AlgorithmNotFoundError):
            agent._convert_q_tables_to_dict("cores")


# ------------------------------ save_model ----------------------------
class TestSaveModel:
    """save_model file outputs."""

    @mock.patch("fusion.modules.rl.algorithms.q_learning.json.dump")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("fusion.modules.rl.algorithms.q_learning.np.save")
    @mock.patch("fusion.modules.rl.algorithms.q_learning.create_directory")
    def test_save_model_writes_files(
        self,
        mock_dir: mock.MagicMock,
        mock_npsave: mock.MagicMock,
        mock_open_fn: mock.MagicMock,
        mock_dump: mock.MagicMock,
    ) -> None:
        """save_model calls create_directory, np.save, and json.dump."""
        agent = _new_agent()
        agent.iteration = 0
        agent.rewards_stats_dict = {"average": np.array([1.0]).tolist()}
        with mock.patch.object(
            agent, "_convert_q_tables_to_dict", return_value={"k": [1]}
        ):
            agent.save_model(trial=0)

        mock_dir.assert_called_once()
        mock_npsave.assert_called_once()
        mock_open_fn.assert_called_once()
        mock_dump.assert_called_once()
