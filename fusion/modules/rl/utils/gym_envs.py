from typing import Any

from stable_baselines3.common.callbacks import CallbackList

from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
from fusion.modules.rl.utils.callbacks import (
    EpisodicRewardCallback,
    LearnRateEntCallback,
)
from fusion.modules.rl.utils.setup import setup_rl_sim


def create_environment(
    config_path: str | None = None
) -> tuple[SimEnv, dict[str, Any], CallbackList]:
    """
    Creates the simulation environment and associated callback for RL.

    :param config_path: Path to configuration file
    :type config_path: str | None
    :return: Tuple containing environment, simulation dictionary, and callback list
    :rtype: tuple[SimEnv, dict[str, Any], CallbackList]
    """
    ep_call_obj = EpisodicRewardCallback(verbose=1)
    param_call_obj = LearnRateEntCallback(verbose=1)
    callback_list = CallbackList([ep_call_obj, param_call_obj])

    flat_dict = setup_rl_sim(config_path=config_path)

    # 🩹 Patch: Ensure SimEnv always receives a dict with sim_dict["s1"]
    wrapped_dict = {"s1": flat_dict} if "s1" not in flat_dict else flat_dict

    env = SimEnv(render_mode=None, custom_callback=ep_call_obj, sim_dict=wrapped_dict)

    return env, wrapped_dict, callback_list
