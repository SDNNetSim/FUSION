import numpy as np

from gymnasium import spaces


class A2C:
    """
    Facilitates Advantage Actor-Critic (A2C) for reinforcement learning.

    This class provides functionalities for handling observation space and action
    space specific to the A2C framework for reinforcement learning. It is driven by
    the properties passed during initialization to define the behavior and attributes
    of the RL environment and its engine.
    """

    def __init__(self, rl_props: object, engine_obj: object):
        """
        A2C initialization function.

        :param rl_props: Object containing reinforcement learning-specific properties.
        :param engine_obj: Object containing engine-specific properties for the environment.
        """
        self.rl_props = rl_props
        self.engine_obj = engine_obj

    def get_obs_space(self):
        """
        Gets the observation space for the a2c reinforcement learning framework.
        """
        bw_set = {d["bandwidth"] for d in self.rl_props.arrival_list}
        num_set = {int(s) for s in bw_set}
        max_bw = str(max(num_set))
        mod_per_bw_dict = self.engine_obj.engine_props['mod_per_bw']
        max_slots = mod_per_bw_dict[max_bw]['QPSK']['slots_needed']
        max_paths = self.rl_props.k_paths

        resp_obs = spaces.Dict({
            'source': spaces.MultiBinary(self.rl_props.num_nodes),
            'destination': spaces.MultiBinary(self.rl_props.num_nodes),
            'request_bandwidth': spaces.MultiBinary(len(bw_set)),
            'holding_time': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'slots_needed': spaces.Box(low=-1, high=max_slots, shape=(max_paths,), dtype=np.int32),
            'path_lengths': spaces.Box(low=0, high=10, shape=(max_paths,), dtype=np.float32),
            # 'fragmentation': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # 'path_frag': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        return resp_obs

    def get_action_space(self):
        """
        Gets the action space for the A2C-based environment.

        By default, we use a discrete action space, where the number of actions corresponds
        to valid paths or nodes within the environment.

        :return: An action space object compatible with Gymnasium.
        """
        action_space = spaces.Discrete(self.engine_obj.engine_props['k_paths'])
        return action_space
