from typing import Any

import numpy as np
from gymnasium import spaces

from fusion.modules.rl.args.observation_args import OBS_DICT
from fusion.modules.rl.utils.topology import convert_networkx_topo


def get_observation_space(rl_props: Any, engine_props: Any) -> dict[str, spaces.Space]:
    """
    Gets any given observation space for DRL algorithms.

    :param rl_props: RL properties object containing configuration
    :type rl_props: Any
    :param engine_props: Engine properties object containing simulation settings
    :type engine_props: Any
    :return: Dictionary defining the observation space for the environment
    :rtype: dict[str, spaces.Space]
    """
    include_graph = False
    bw_set = {d["bandwidth"] for d in rl_props.arrival_list}
    num_set = {int(s) for s in bw_set}
    max_bw = str(max(num_set))
    mod_per_bw_dict = engine_props.engine_props["mod_per_bw"]
    max_slots = mod_per_bw_dict[max_bw]["QPSK"]["slots_needed"]
    max_paths = rl_props.k_paths

    obs_key = engine_props.engine_props.get("obs_space", "obs_1")
    if "graph" in obs_key:
        include_graph = True
        obs_key = obs_key.replace("_graph", "")

    obs_features = OBS_DICT.get(obs_key, OBS_DICT["obs_1"])
    print("Observation space: ", obs_features)

    feature_space_map = {
        "source": lambda: spaces.MultiBinary(rl_props.num_nodes),
        "destination": lambda: spaces.MultiBinary(rl_props.num_nodes),
        "request_bandwidth": lambda: spaces.MultiBinary(len(bw_set)),
        "holding_time": lambda: spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "slots_needed": lambda: spaces.Box(
            low=-1, high=max_slots, shape=(max_paths,), dtype=np.int32
        ),
        "path_lengths": lambda: spaces.Box(
            low=0, high=10, shape=(max_paths,), dtype=np.float32
        ),
        "available_slots": lambda: spaces.Box(
            low=0, high=1, shape=(max_paths,), dtype=np.float32
        ),
        "is_feasible": lambda: spaces.MultiBinary(max_paths),
        "paths_cong": lambda: spaces.Box(
            low=0, high=1, shape=(max_paths,), dtype=np.float32
        ),
    }

    obs_space_dict = {
        feature: space_fn()
        for feature, space_fn in feature_space_map.items()
        if feature in obs_features
    }

    ei, ea, xf, _ = convert_networkx_topo(
        engine_props.engine_props["topology"], as_directed=True
    )
    num_nodes, num_edges = xf.shape[0], ei.shape[1]
    if include_graph:
        obs_space_dict.update(
            {
                "x": spaces.Box(-np.inf, np.inf, xf.shape, dtype=np.float32),
                "edge_index": spaces.Box(0, num_nodes - 1, ei.shape, dtype=np.int64),
                "edge_attr": spaces.Box(-np.inf, np.inf, ea.shape, dtype=np.float32),
                "path_masks": spaces.Box(
                    0, 1, (max_paths, num_edges), dtype=np.float32
                ),
            }
        )
        print(f"Updated the observation space with graph features: {obs_space_dict}")

    return obs_space_dict


class FragmentationTracker:
    """
    Tracks fragmentation.
    """

    def __init__(self, num_nodes: int, core_count: int, spectral_slots: int) -> None:
        """
        Initializes the fragmentation tracker.

        :param num_nodes: Total number of nodes in the network topology
        :type num_nodes: int
        :param core_count: Number of cores per link
        :type core_count: int
        :param spectral_slots: Number of spectral slots per core
        :type spectral_slots: int
        """
        self.num_nodes = num_nodes
        self.core_count = core_count
        self.spectral_slots = spectral_slots

        # Bitmask: [src, dst, core, slots]
        self.core_masks = np.zeros(
            (num_nodes, num_nodes, core_count, spectral_slots), dtype=np.uint8
        )

        # Dirty mask to avoid unnecessary recomputation
        self.dirty_cores = np.zeros((num_nodes, num_nodes, core_count), dtype=bool)

        # Normalize by max possible transitions per core (not including zero-padding)
        self.norm_factor = (spectral_slots // 3) + 1

    def update(
        self,
        src: int,
        dst: int,
        core_index: int,
        start_slot: int,
        end_slot: int,
        is_allocate: bool = True,
    ) -> None:
        """
        Updates the core mask for a specific link (src→dst), core, and slot range.

        :param src: Source node index
        :type src: int
        :param dst: Destination node index
        :type dst: int
        :param core_index: Core used for allocation
        :type core_index: int
        :param start_slot: First slot index allocated
        :type start_slot: int
        :param end_slot: Last slot index allocated (inclusive)
        :type end_slot: int
        :param is_allocate: If True, mark as allocated (1); if False, free (0)
        :type is_allocate: bool
        """
        slot_indices = np.arange(start_slot, end_slot + 1)
        if is_allocate:
            self.core_masks[src, dst, core_index, slot_indices] = 1
        else:
            self.core_masks[src, dst, core_index, slot_indices] = 0

        self.dirty_cores[src, dst, core_index] = True

    def get_fragmentation(
        self, chosen_path: list[int], core_index: int
    ) -> dict[str, np.ndarray]:
        """
        Computes fragmentation for the specified core and chosen path.

        :param chosen_path: List of node indices representing the path
        :type chosen_path: list[int]
        :param core_index: Core used during allocation
        :type core_index: int
        :return: Dictionary with formatted NumPy arrays for Gym observations
        :rtype: dict[str, np.ndarray]
        """
        # 1. Core-level fragmentation (last link in path)
        core_frag = 0.0
        if len(chosen_path) >= 2:
            last_src, last_dst = chosen_path[-2], chosen_path[-1]
            if self.dirty_cores[last_src, last_dst, core_index]:
                core_mask = self.core_masks[last_src, last_dst, core_index]
                padded = np.pad(core_mask, (1,), "constant")
                transitions = np.abs(np.diff(padded))
                core_frag = np.sum(transitions) / (2 * self.norm_factor)

        # 2. Path-level fragmentation (avg across all links in path, all cores)
        path_transitions = 0
        total_links = len(chosen_path) - 1

        for i in range(total_links):
            src, dst = chosen_path[i], chosen_path[i + 1]
            for c in range(self.core_count):
                padded = np.pad(self.core_masks[src, dst, c], (1,), "constant")
                transitions = np.abs(np.diff(padded))
                path_transitions += np.sum(transitions)

        if total_links > 0:
            path_frag = path_transitions / (
                2 * self.norm_factor * total_links * self.core_count
            )
        else:
            path_frag = 0.0

        self.dirty_cores.fill(False)

        return {
            "fragmentation": np.array([core_frag], dtype=np.float32),
            "path_frag": np.array([path_frag], dtype=np.float32),
        }
