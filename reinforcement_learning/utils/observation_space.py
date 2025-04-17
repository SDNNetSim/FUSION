import numpy as np

from gymnasium import spaces

from reinforcement_learning.args.observation_args import OBS_DICT


def get_observation_space(rl_props: object, engine_obj: object):
    bw_set = {d["bandwidth"] for d in rl_props.arrival_list}
    num_set = {int(s) for s in bw_set}
    max_bw = str(max(num_set))
    mod_per_bw_dict = engine_obj.engine_props['mod_per_bw']
    max_slots = mod_per_bw_dict[max_bw]['QPSK']['slots_needed']
    max_paths = rl_props.k_paths

    obs_key = engine_obj.engine_props.get('obs_space', 'obs_1')
    obs_features = OBS_DICT.get(obs_key, OBS_DICT['obs_1'])
    print("Observation space: ", obs_features)

    feature_space_map = {
        "source": lambda: spaces.MultiBinary(rl_props.num_nodes),
        "destination": lambda: spaces.MultiBinary(rl_props.num_nodes),
        "request_bandwidth": lambda: spaces.MultiBinary(len(bw_set)),
        "holding_time": lambda: spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "slots_needed": lambda: spaces.Box(low=-1, high=max_slots, shape=(max_paths,), dtype=np.int32),
        "path_lengths": lambda: spaces.Box(low=0, high=10, shape=(max_paths,), dtype=np.float32),
        "available_slots": lambda: spaces.Box(low=0, high=1, shape=(max_paths,), dtype=np.float32),
        "is_feasible": lambda: spaces.MultiBinary(max_paths),
        "paths_cong": lambda: spaces.Box(low=0, high=1, shape=(max_paths,), dtype=np.float32),
    }

    obs_space_dict = {
        feature: space_fn()
        for feature, space_fn in feature_space_map.items()
        if feature in obs_features
    }

    return obs_space_dict
