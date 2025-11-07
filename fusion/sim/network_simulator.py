"""
Legacy network simulator for FUSION.

This module provides the legacy multi-process network simulation capability.
New code should use the batch_runner module for improved orchestration.
"""

import copy
from datetime import datetime
from multiprocessing import Manager, Process
from typing import Any

from fusion.core.simulation import SimulationEngine
from fusion.sim.input_setup import create_input, save_input
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def _validate_bandwidth_consistency(engine_props: dict[str, Any]) -> None:
    """
    Validate bandwidth configuration consistency.

    Ensures that bandwidth configuration is consistent between request_distribution
    and mod_per_bw. This should be called after create_input populates mod_per_bw.

    :param engine_props: Engine properties containing configuration
    :type engine_props: dict[str, Any]
    :raises ValueError: If bandwidth configuration is inconsistent
    """
    request_distribution = engine_props.get("request_distribution", {})
    mod_per_bw = engine_props.get("mod_per_bw", {})

    if not request_distribution:
        return  # No request distribution to validate

    if not mod_per_bw:
        raise ValueError(
            "mod_per_bw is empty after input setup. "
            "Check mod_assumption_path configuration."
        )

    # Check that all bandwidths in request_distribution exist in mod_per_bw
    missing_bandwidths = []
    for bandwidth in request_distribution.keys():
        if bandwidth not in mod_per_bw:
            missing_bandwidths.append(bandwidth)

    if missing_bandwidths:
        available_bandwidths = list(mod_per_bw.keys())
        mod_assumption_path = engine_props.get("mod_assumption_path", "Unknown")
        raise ValueError(
            f"Bandwidth configuration mismatch: request_distribution references "
            f"bandwidths {missing_bandwidths} that are not available in mod_per_bw. "
            f"Available bandwidths in mod_per_bw: {available_bandwidths}. "
            f"Current mod_assumption_path: {mod_assumption_path}. "
            f"Please ensure your mod_assumption file contains all required bandwidths "
            f"or update request_distribution to only use available bandwidths."
        )


class NetworkSimulator:
    """
    Legacy network simulator controller.

    This class provides the original multi-process simulation control.
    New implementations should use BatchRunner from batch_runner module.
    """

    def __init__(self) -> None:
        """Initialize the network simulator."""
        self.properties: dict[str, Any] = {}

    def _run_generic_sim(
        self,
        erlang: float,
        first_erlang: bool,
        erlang_index: int,
        progress_dict: Any,
        done_offset: int,
    ) -> int:
        """
        Handle simulation for one Erlang value in this process.

        Sets arrival_rate, passes done_offset to avoid resetting progress
        between volumes, and calls Engine.run().

        :param erlang: Traffic load in Erlangs
        :type erlang: float
        :param first_erlang: Whether this is the first Erlang in sequence
        :type first_erlang: bool
        :param erlang_index: Index of current Erlang
        :type erlang_index: int
        :param progress_dict: Shared progress dictionary
        :type progress_dict: Any
        :param done_offset: Offset for progress tracking
        :type done_offset: int
        :return: Final number of done units
        :rtype: int
        """

        # Remove unpickleable keys so we can deepcopy
        unpickleable_keys = {}
        for key in ["log_queue", "progress_queue", "stop_flag"]:
            if key in self.properties:
                unpickleable_keys[key] = self.properties.pop(key)

        engine_props = copy.deepcopy(self.properties)

        # Restore the queues
        self.properties.update(unpickleable_keys)
        if "log_queue" in unpickleable_keys:
            engine_props["log_queue"] = unpickleable_keys["log_queue"]
        if "progress_queue" in unpickleable_keys:
            engine_props["progress_queue"] = unpickleable_keys["progress_queue"]
        if "stop_flag" in unpickleable_keys:
            engine_props["stop_flag"] = unpickleable_keys["stop_flag"]

        # Insert progress tracking
        engine_props["progress_dict"] = progress_dict
        engine_props["progress_key"] = erlang_index

        # Set 'erlang' and 'arrival_rate'
        engine_props["erlang"] = erlang
        engine_props["arrival_rate"] = (
            engine_props["cores_per_link"] * erlang
        ) / engine_props["holding_time"]

        # Pass how many units of work have been done so far
        engine_props["done_offset"] = done_offset

        engine_props["my_iteration_units"] = self.properties.get(
            "my_iteration_units", engine_props["max_iters"]
        )

        # Create sanitized copy for saving
        clean_engine_props = engine_props.copy()
        for badkey in [
            "progress_dict",
            "progress_key",
            "log_queue",
            "progress_queue",
            "done_offset",
            "stop_flag",
        ]:
            clean_engine_props.pop(badkey, None)

        updated_props = create_input(base_fp="data", engine_props=clean_engine_props)
        engine_props.update(updated_props)

        # Save input if first Erlang
        if first_erlang:
            save_input(
                base_fp="data",
                properties=clean_engine_props,
                file_name=f"sim_input_{updated_props['thread_num']}.json",
                data_dict=updated_props,
            )

        logger.debug(
            "[Simulation %s] progress_dict id: %s",
            erlang_index,
            id(engine_props["progress_dict"]),
        )

        engine = SimulationEngine(engine_props=engine_props)
        final_done_units = engine.run()

        return final_done_units

    def run_generic_sim(self) -> None:
        """
        Run multiple Erlangs sequentially in this single process.

        :raises ValueError: If properties are not properly configured
        """
        start, stop = self.properties["erlang_start"], self.properties["erlang_stop"]
        step = self.properties["erlang_step"]
        # Add 1 to stop to make range inclusive (user specifies the last erlang to run)
        erlang_list = [float(x) for x in range(int(start), int(stop) + 1, int(step))]
        logger.info("Launching simulations for erlangs: %s", erlang_list)

        max_iters = self.properties["max_iters"]
        my_iteration_units = len(erlang_list) * max_iters
        self.properties["my_iteration_units"] = my_iteration_units

        if "progress_dict" not in self.properties:
            manager = Manager()
            self.properties["progress_dict"] = manager.dict()

        progress_dict = self.properties["progress_dict"]
        logger.debug("Initial shared progress dict: %s", dict(progress_dict))

        done_units_so_far = 0

        for erlang_index, erlang in enumerate(erlang_list):
            first_erlang = erlang_index == 0

            done_units_so_far = self._run_generic_sim(
                erlang=erlang,
                first_erlang=first_erlang,
                erlang_index=erlang_index,
                progress_dict=progress_dict,
                done_offset=done_units_so_far,
            )

    def run_sim(self, **kwargs: Any) -> None:
        """
        Set up internal state and trigger the run_generic_sim().

        :param kwargs: Keyword arguments containing simulation parameters
        :type kwargs: Any
        """
        self.properties = kwargs["thread_params"]
        self.properties["date"] = kwargs["sim_start"].split("_")[0]

        tmp_list = kwargs["sim_start"].split("_")
        time_string = f"{tmp_list[1]}_{tmp_list[2]}_{tmp_list[3]}_{tmp_list[4]}"
        self.properties["sim_start"] = time_string
        self.properties["thread_num"] = kwargs["thread_num"]

        self.run_generic_sim()


def run(sims_dict: dict[str, Any], stop_flag: Any) -> None:
    """
    Spawn one process per simulation config, each handling multiple Erlangs.

    :param sims_dict: Dictionary of simulation configurations
    :type sims_dict: dict[str, Any]
    :param stop_flag: Multiprocessing event for stopping simulations
    :type stop_flag: multiprocessing.Event
    """
    any_conf = list(sims_dict.values())[0]
    log_queue = any_conf.get("log_queue")
    progress_queue = any_conf.get("progress_queue")

    def log(message: str) -> None:
        """
        Log message to queue or logger.

        :param message: Message to log
        :type message: str
        """
        if log_queue:
            log_queue.put(message)
        else:
            logger.info(message)

    processes = []

    if "sim_start" not in sims_dict["s1"]:
        sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
    else:
        sim_start = f"{sims_dict['s1']['date']}_{sims_dict['s1']['sim_start']}"

    for thread_num, thread_params in sims_dict.items():
        thread_params["progress_queue"] = progress_queue
        thread_params["stop_flag"] = stop_flag

        log(f"Starting simulation for thread {thread_num} at {sim_start}.")
        curr_sim = NetworkSimulator()

        p = Process(
            target=curr_sim.run_sim,
            kwargs={
                "thread_num": thread_num,
                "thread_params": thread_params,
                "sim_start": sim_start,
            },
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
