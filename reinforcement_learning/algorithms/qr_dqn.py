import numpy as np

from gymnasium import spaces


class QrDQN:
    """
    Facilitates DQN for reinforcement learning.
    """

    def __init__(self, rl_props: object, engine_obj: object):
        self.rl_props = rl_props
        self.engine_obj = engine_obj

    def get_obs_space(self):
        """
        Gets the observation space for the ppo reinforcement learning framework.
        """
        bw_set = {d["bandwidth"] for d in self.rl_props.arrival_list}
        num_set = {int(s) for s in bw_set}
        max_bw = str(max(num_set))
        mod_per_bw_dict = self.engine_obj.engine_props['mod_per_bw']
        max_slots = mod_per_bw_dict[max_bw]['QPSK']['slots_needed']
        max_paths = self.rl_props.k_paths

        obs_combo_dict = {
            "obs_1": ["source", "destination"],
            "obs_2": ["source", "destination", "request_bandwidth"],
            "obs_3": ["source", "destination", "holding_time"],
            "obs_4": ["source", "destination", "request_bandwidth", "holding_time"],
            "obs_5": ["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths"],
            "obs_6": ["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths",
                      "paths_cong"],
        }

        obs_key = self.engine_obj.engine_props.get('obs_space', 'obs_1')
        obs_features = obs_combo_dict.get(obs_key, obs_combo_dict['obs_1'])
        print("Observation space: ", obs_features)

        feature_space_map = {
            "source": lambda: spaces.MultiBinary(self.rl_props.num_nodes),
            "destination": lambda: spaces.MultiBinary(self.rl_props.num_nodes),
            "request_bandwidth": lambda: spaces.MultiBinary(len(bw_set)),
            "holding_time": lambda: spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "slots_needed": lambda: spaces.Box(low=-1, high=max_slots, shape=(max_paths,), dtype=np.int32),
            "path_lengths": lambda: spaces.Box(low=0, high=10, shape=(max_paths,), dtype=np.float32),
            "req_id": lambda: spaces.Discrete(5005),
            "paths_cong": lambda: spaces.Box(low=0, high=1, shape=(max_paths,), dtype=np.float32),
        }

        obs_space_dict = {
            feature: space_fn()
            for feature, space_fn in feature_space_map.items()
            if feature in obs_features
        }

        return spaces.Dict(obs_space_dict)

    def get_action_space(self):
        """
        Get the action space for the environment.
        """
        action_space = spaces.Discrete(self.engine_obj.engine_props['k_paths'])
        return action_space