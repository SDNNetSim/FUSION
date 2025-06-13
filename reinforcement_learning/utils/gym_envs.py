from stable_baselines3.common.callbacks import CallbackList

from reinforcement_learning.gymnasium_envs.general_sim_env import SimEnv

from reinforcement_learning.utils.setup import setup_rl_sim
from reinforcement_learning.utils.callbacks import EpisodicRewardCallback, LearnRateEntCallback


def create_environment(config_path: str = None):
    """
    Creates the simulation environment and associated callback for RL.

    :return: A tuple consisting of the SimEnv object and its sim_dict.
    """
    ep_call_obj = EpisodicRewardCallback(verbose=1)
    param_call_obj = LearnRateEntCallback(verbose=1)

    callback_list = CallbackList([ep_call_obj, param_call_obj])
    # TODO: I don't believe the callback is even used in sim env
    env = SimEnv(render_mode=None, custom_callback=ep_call_obj, sim_dict=setup_rl_sim(config_path=config_path))
    env.sim_dict['callback'] = callback_list
    return env, env.sim_dict, callback_list
