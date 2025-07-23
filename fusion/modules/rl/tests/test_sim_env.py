"""Unit tests for reinforcement_learning.utils.sim_env (helpers only)."""

from types import SimpleNamespace
from unittest import TestCase, mock

from reinforcement_learning.utils.sim_env import SimEnvUtils, SimEnvObs


# ------------------------------------------------------------------ #
#  minimal sim_env object builder                                     #
# ------------------------------------------------------------------ #
def _make_sim_env(path_algo="q_learning", is_drl=True):
    """Return a stub with the attributes SimEnv helpers access."""
    rl_props = SimpleNamespace(
        arrival_count=1,
        num_nodes=3,
        arrival_list=[{"source": 0, "destination": 1}],
        chosen_path_list=[[0, 1]],
        path_index=None,
        core_index=None,
        forced_index=None,
        mock_sdn_dict={},
    )

    engine_props = dict(
        num_requests=1,
        is_drl_agent=is_drl,
        topology="topo",
        obs_space="obs_1",
    )

    engine_obj = SimpleNamespace(
        engine_props=engine_props,
        end_iter=mock.MagicMock(),
        net_spec_dict={},
        reqs_dict={0: {"depart": 10}, 1: {"depart": 15}},
        stats_obj=None,
    )

    path_agent = SimpleNamespace(
        end_iter=mock.MagicMock(),
        update=mock.MagicMock(),
        get_route=mock.MagicMock(),
    )

    rl_help = SimpleNamespace(
        handle_releases=mock.MagicMock(),
        update_mock_sdn=mock.MagicMock(return_value={}),
        update_route_props=mock.MagicMock(),
        mock_handle_arrival=mock.MagicMock(return_value=True),
        path_index=None,
        core_num=None,
        rl_props=rl_props,
        engine_obj=engine_obj,
    )

    sim_dict = dict(
        path_algorithm=path_algo,
        core_algorithm="none",
        spectrum_algorithm=None,
        is_training=True,
        request_distribution={"100": 1},
    )

    return SimpleNamespace(
        rl_props=rl_props,
        engine_obj=engine_obj,
        path_agent=path_agent,
        core_agent=None,
        route_obj=SimpleNamespace(
            route_props=SimpleNamespace(
                paths_matrix=[[0, 1]],
                mod_formats_matrix=[["QPSK"]],
            ),
            engine_props={},
            sdn_props={},
            get_route=mock.MagicMock(),
        ),
        rl_help_obj=rl_help,
        sim_dict=sim_dict,
        iteration=0,
        step_helper=None,
        sim_env_helper=SimpleNamespace(get_drl_obs=lambda **_: {}),
    )


# ------------------------------------------------------------------ #
class TestCheckTerminated(TestCase):
    """SimEnvUtils.check_terminated end-of-episode flow."""

    @mock.patch("reinforcement_learning.utils.sim_env.VALID_PATH_ALGORITHMS",
                ["q_learning"])
    @mock.patch("reinforcement_learning.utils.sim_env.os.path.join",
                return_value="/tmp")
    def test_increments_iter_and_calls_end(self, _):
        """Returns True, increments iteration, calls agent end_iter."""
        senv = _make_sim_env()
        helper = SimEnvUtils(senv)

        terminated = helper.check_terminated()

        self.assertTrue(terminated)
        self.assertEqual(senv.iteration, 1)
        senv.path_agent.end_iter.assert_called_once()
        senv.engine_obj.end_iter.assert_called_once()


# ------------------------------------------------------------------ #
class TestScaleReqHolding(TestCase):
    """SimEnvObs._scale_req_holding numeric scaling."""

    def setUp(self):
        self.senv = _make_sim_env()
        self.obs = SimEnvObs(self.senv)

    def test_correct_scaling_value(self):
        """(12-10)/(15-10-1)=0.5 when diff counted discretely."""
        scaled = self.obs._scale_req_holding(holding_time=12)  # pylint:disable=protected-access
        self.assertAlmostEqual(scaled, 0.5)

    def test_equal_min_max_returns_one(self):
        """When all departures equal, value defaults to 1.0."""
        self.senv.engine_obj.reqs_dict = {0: {"depart": 10}, 1: {"depart": 10}}
        scaled = self.obs._scale_req_holding(holding_time=10)  # pylint:disable=protected-access
        self.assertEqual(scaled, 1.0)


# ------------------------------------------------------------------ #
class TestUpdateHelperObj(TestCase):
    """SimEnvObs.update_helper_obj path-index assignment."""

    def test_sets_path_index_for_drl_agent(self):
        """If is_drl_agent True → rl_help_obj.path_index == action."""
        senv = _make_sim_env(is_drl=True)
        obs = SimEnvObs(senv)

        obs.update_helper_obj(action=2, bandwidth="100")  # pylint:disable=invalid-name

        self.assertEqual(senv.rl_help_obj.path_index, 2)
        self.assertIsNone(senv.rl_help_obj.rl_props.forced_index)
