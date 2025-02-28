from rl_scripts.gymnasium_envs.general_sim_env import SimEnv

from rl_scripts.utils.setup_helpers import setup_rl_sim
from rl_scripts.utils.callback_helpers import EpisodicRewardCallback


def create_environment():
    """
    Creates the simulation environment and associated callback for RL.

    :return: A tuple consisting of the SimEnv object and its sim_dict.
    """
    callback = EpisodicRewardCallback(verbose=1)
    env = SimEnv(render_mode=None, custom_callback=callback, sim_dict=setup_rl_sim())
    env.sim_dict['callback'] = callback
    return env, env.sim_dict, callback
