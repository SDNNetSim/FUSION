"""
Visualization and plotting properties.
Moved from cli/args/plot_args.py for better organization.
"""

from __future__ import annotations

import copy
import os


class PlotProps:  # pylint: disable=too-few-public-methods
    """
    Properties used in plotting and visualization operations.
    """

    def __init__(self) -> None:
        # Contains all necessary information for each simulation run
        self.sims_info_dict = None
        # Contains only information related to plotting for each simulation run
        self.plot_dict: dict[str, dict[str, PlotArgs]] | None = None
        # The base output directory when saving graphs
        self.output_dir = os.path.join("..", "..", "data", "output")
        # The base input directory when reading simulation input
        self.input_dir = os.path.join("..", "..", "data", "input")
        # Has the information for one simulation run for each Erlang value
        self.erlang_dict: dict | None = None
        # The number of requests used for each iteration
        self.num_requests = None
        # Number of cores used for each iteration
        self.num_cores = None

        self.color_list = [
            "#024de3",
            "#00b300",
            "orange",
            "#6804cc",
            "#e30220",
        ]  # Colors used for lines
        self.style_list = [
            "solid",
            "dashed",
            "dotted",
            "dashdot",
        ]  # Styles used for lines
        self.marker_list = ["o", "^", "s", "x"]  # Marker styles used for lines
        self.x_tick_list = [50, 100, 200, 300, 400, 500, 600, 700]  # X-tick labels
        self.title_names: str | None = (
            None  # Important names used for titles in plots (one string)
        )

    def __repr__(self) -> str:
        return f"PlotProps({self.__dict__})"


class PlotArgs:
    """
    Arguments and data used in plotting operations.
    """

    def __init__(self) -> None:
        self.erlang_list: list[float] = []  # Numerical Erlang values we are plotting
        self.blocking_list: list[float] = []  # Blocking values to be plotted
        self.lengths_list: list[float] = []  # Average path length values
        self.hops_list: list[float] = []  # Average path hop
        self.occ_slot_matrix: list[
            float
        ] = []  # Occupied slots in the entire network at different snapshots
        self.active_req_matrix: list[
            float
        ] = []  # Number of requests allocated in the network (snapshots)
        self.block_req_matrix: list[
            float
        ] = []  # Running average blocking probabilities (snapshots)
        self.req_num_list: list[
            int
        ] = []  # Active request identification numbers (snapshots)
        self.times_list: list[float] = []  # Simulation start times
        self.modulations_dict: dict[
            str, dict[str, list[float]]
        ] = {}  # Modulation formats used
        self.dist_block_list: list[
            float
        ] = []  # Percentage of blocking due to a reach constraint
        self.cong_block_list: list[
            float
        ] = []  # Percentage of blocking due to a congestion constraint
        self.holding_time = None  # Holding time for the simulation run
        self.cores_per_link = None  # Number of cores per link
        # TODO: (drl_path_agents) Does not support all bands, check on this
        self.c_band = None  # Spectral slots per core for the c-band
        self.learn_rate = (
            None  # For artificial intelligence (AI), learning rate used if any
        )
        self.discount_factor = None  # For AI, discount factor used if any

        self.block_per_iter: list[
            float
        ] = []  # Blocking probability per iteration of one simulation configuration
        self.sum_rewards_list: list[
            float
        ] = []  # For reinforcement learning (RL), sum of rewards per episode
        self.sum_errors_list: list[float] = []  # For RL, sum of errors per episode
        self.epsilon_list: list[
            float
        ] = []  # For RL, decay of epsilon w.r.t. each episode

    def __setitem__(self, key: str, value: object) -> None:
        setattr(self, key, value)

    def __getitem__(self, key: str) -> object:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(f"{key} not found") from exc

    def __delitem__(self, key: str) -> None:
        try:
            delattr(self, key)
        except AttributeError as exc:
            raise KeyError(f"'{key}' not found") from exc

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    @staticmethod
    def update_info_dict(
        plot_props: PlotProps,
        input_dict: dict,
        info_item_list: list,
        time: str,
        sim_num: str,
    ) -> PlotProps:
        """
        Updates various items in the plot dictionary.

        :param plot_props: Main plot script properties object.
        :param input_dict: Input dictionary containing information for each
            item (blocking, length, etc.)
        :param info_item_list: Keys of the dictionary to be updated.
        :param time: Simulation start time.
        :param sim_num: Simulation number.
        :return: The updated plot properties with the simulation information.
        :rtype: object
        """
        resp_plot_props = copy.deepcopy(plot_props)
        if resp_plot_props.plot_dict is not None:
            for info_item in info_item_list:
                resp_plot_props.plot_dict[time][sim_num][info_item] = input_dict[
                    info_item
                ]

        return resp_plot_props

    def __repr__(self) -> str:
        return f"PlotArgs({self.__dict__})"
