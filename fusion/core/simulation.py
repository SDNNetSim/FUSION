"""
Simulation engine module for running optical network simulations.

This module provides the main simulation engine functionality for running
optical network simulations with support for ML/RL models and various metrics.
"""

import copy
import os
import signal
import time
from typing import Any

import networkx as nx
import numpy as np

from fusion.core.metrics import SimStats
from fusion.core.ml_metrics import MLMetricsCollector
from fusion.core.persistence import StatsPersistence
from fusion.core.request import get_requests
from fusion.core.sdn_controller import SDNController
from fusion.modules.ml import load_model
from fusion.reporting.simulation_reporter import SimulationReporter
from fusion.utils.logging_config import get_logger, log_message

logger = get_logger(__name__)


def seed_all_rngs(seed: int) -> None:
    """
    Seed all random number generators for reproducibility.

    Seeds:
    - Python's built-in random module
    - NumPy's random state
    - PyTorch's random state (CPU and CUDA)

    Also sets PyTorch to deterministic mode to prevent
    non-deterministic operations.

    :param seed: Random seed (integer)
    :type seed: int

    Example:
        >>> seed_all_rngs(42)
        >>> # All subsequent random operations are reproducible
        >>> import random
        >>> random.random()
        0.6394267984578837
        >>> np.random.rand()
        0.3745401188473625
    """
    # Seed Python's random module
    import random
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Seed PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Enforce deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)

    except ImportError:
        # PyTorch not installed, skip
        pass


def generate_seed_from_time() -> int:
    """
    Generate a seed from current time.

    Used when no seed is explicitly provided.

    :return: Seed value
    :rtype: int

    Example:
        >>> seed = generate_seed_from_time()
        >>> print(seed)
        1678901234
    """
    return int(time.time() * 1000) % (2**31 - 1)


def validate_seed(seed: int) -> int:
    """
    Validate and normalize seed value.

    Ensures seed is in valid range for all RNGs.

    :param seed: Seed value
    :type seed: int
    :return: Validated seed
    :rtype: int
    :raises ValueError: If seed is out of valid range

    Example:
        >>> validate_seed(42)
        42
        >>> validate_seed(-1)
        ValueError: Seed must be non-negative
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    if seed > 2**31 - 1:
        raise ValueError(f"Seed must be < 2^31, got {seed}")

    return seed


class SimulationEngine:
    """
    Controls a single simulation.

    This class manages the execution of a single optical network simulation,
    including topology creation, request processing, and statistics collection.

    :param engine_props: Engine configuration properties
    :type engine_props: dict[str, Any]
    """

    def __init__(self, engine_props: dict[str, Any]) -> None:
        self.engine_props = engine_props
        self.network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]] = {}
        self.reqs_dict: dict[float, dict[str, Any]] | None = None
        self.reqs_status_dict: dict[int, dict[str, Any]] = {}

        self.iteration = 0
        self.topology = nx.Graph()
        self.sim_info = os.path.join(
            self.engine_props["network"],
            self.engine_props["date"],
            self.engine_props["sim_start"],
        )

        self.sdn_obj = SDNController(engine_props=self.engine_props)
        self.stats_obj = SimStats(
            engine_props=self.engine_props, sim_info=self.sim_info
        )
        self.reporter = SimulationReporter(logger=logger)
        self.persistence = StatsPersistence(
            engine_props=self.engine_props, sim_info=self.sim_info
        )

        # Initialize ML metrics collector if needed
        self.ml_metrics: MLMetricsCollector | None = (
            MLMetricsCollector(engine_props=self.engine_props, sim_info=self.sim_info)
            if engine_props.get("output_train_data", False)
            else None
        )

        self.ml_model = None
        self.stop_flag = engine_props.get("stop_flag")

    def update_arrival_params(self, current_time: float) -> None:
        """
        Update parameters for a request after attempted allocation.

        :param current_time: The current simulated time
        :type current_time: float
        """
        if self.reqs_dict is None or current_time not in self.reqs_dict:
            return

        sdn_props = self.sdn_obj.sdn_props
        self.stats_obj.iter_update(
            req_data=self.reqs_dict[current_time],
            sdn_data=sdn_props,
            network_spectrum_dict=self.network_spectrum_dict,
        )
        if sdn_props.was_routed:
            if sdn_props.number_of_transponders is not None:
                self.stats_obj.current_transponders = sdn_props.number_of_transponders

            self.reqs_status_dict.update(
                {
                    self.reqs_dict[current_time]["req_id"]: {
                        "mod_format": sdn_props.modulation_list,
                        "path": sdn_props.path_list,
                        "is_sliced": sdn_props.is_sliced,
                        "was_routed": sdn_props.was_routed,
                        "core_list": sdn_props.core_list,
                        "band": sdn_props.band_list,
                        "start_slot_list": sdn_props.start_slot_list,
                        "end_slot_list": sdn_props.end_slot_list,
                        "bandwidth_list": sdn_props.bandwidth_list,
                        "snr_cost": sdn_props.crosstalk_list,
                    }
                }
            )

    def handle_arrival(
        self,
        current_time: float,
        force_route_matrix: list[Any] | None = None,
        force_core: int | None = None,
        force_slicing: bool = False,
        forced_index: int | None = None,
        force_mod_format: str | None = None,
    ) -> None:
        """
        Update the SDN controller to handle an arrival request.

        :param current_time: The arrival time of the request
        :type current_time: float
        :param force_route_matrix: Passes forced routes to the SDN controller
        :type force_route_matrix: Optional[List[Any]]
        :param force_core: Force a certain core for allocation
        :type force_core: Optional[int]
        :param force_slicing: Forces slicing in the SDN controller
        :type force_slicing: bool
        :param forced_index: Forces an index in the SDN controller
        :type forced_index: Optional[int]
        :param force_mod_format: Forces a modulation format
        :type force_mod_format: Optional[str]
        """
        if self.reqs_dict is None or current_time not in self.reqs_dict:
            return

        for request_key, request_value in self.reqs_dict[current_time].items():
            if request_key == "mod_formats":
                request_key = "modulation_formats_dict"
            elif request_key == "req_id":
                request_key = "request_id"
            self.sdn_obj.sdn_props.update_params(
                key=request_key,
                spectrum_key=None,
                spectrum_obj=None,
                value=request_value,
            )

        self.sdn_obj.handle_event(
            self.reqs_dict[current_time],
            request_type="arrival",
            force_route_matrix=force_route_matrix,
            force_slicing=force_slicing,
            forced_index=forced_index,
            force_core=force_core,
            ml_model=self.ml_model,
            force_mod_format=force_mod_format,
        )
        if self.sdn_obj.sdn_props.network_spectrum_dict is not None:
            self.network_spectrum_dict = self.sdn_obj.sdn_props.network_spectrum_dict
        self.update_arrival_params(current_time=current_time)

    def handle_release(self, current_time: float) -> None:
        """
        Update the SDN controller to handle the release of a request.

        :param current_time: The arrival time of the request
        :type current_time: float
        """
        if self.reqs_dict is None or current_time not in self.reqs_dict:
            return

        for request_key, request_value in self.reqs_dict[current_time].items():
            if request_key == "mod_formats":
                request_key = "modulation_formats_dict"
            elif request_key == "req_id":
                request_key = "request_id"
            self.sdn_obj.sdn_props.update_params(
                key=request_key,
                spectrum_key=None,
                spectrum_obj=None,
                value=request_value,
            )

        if (
            self.reqs_dict is not None
            and current_time in self.reqs_dict
            and self.reqs_dict[current_time]["req_id"] in self.reqs_status_dict
        ):
            self.sdn_obj.sdn_props.path_list = self.reqs_status_dict[
                self.reqs_dict[current_time]["req_id"]
            ]["path"]
            self.sdn_obj.handle_event(
                self.reqs_dict[current_time], request_type="release"
            )
            sdn_spectrum_dict = self.sdn_obj.sdn_props.network_spectrum_dict
            if sdn_spectrum_dict is not None:
                self.network_spectrum_dict = sdn_spectrum_dict
        # Request was blocked, nothing to release

    def create_topology(self) -> None:
        """Create the physical topology of the simulation."""
        self.network_spectrum_dict = {}
        self.topology.add_nodes_from(self.engine_props["topology_info"]["nodes"])

        self.engine_props["band_list"] = []
        for band in ["c", "l", "s", "o", "e"]:
            try:
                if self.engine_props[f"{band}_band"]:
                    self.engine_props["band_list"].append(band)
            except KeyError:
                continue

        for link_num, link_data in self.engine_props["topology_info"]["links"].items():
            source = link_data["source"]
            dest = link_data["destination"]

            cores_matrix: dict[str, np.ndarray] = {}
            for band in self.engine_props["band_list"]:
                band_slots = self.engine_props[f"{band}_band"]
                cores_matrix[band] = np.zeros(
                    (link_data["fiber"]["num_cores"], band_slots)
                )

            self.network_spectrum_dict[(source, dest)] = {
                "cores_matrix": cores_matrix,
                "link_num": int(link_num),
                "usage_count": 0,
                "throughput": 0,
            }
            self.network_spectrum_dict[(dest, source)] = {
                "cores_matrix": cores_matrix,
                "link_num": int(link_num),
                "usage_count": 0,
                "throughput": 0,
            }
            self.topology.add_edge(
                source, dest, length=link_data["length"], nli_cost=None
            )

        self.engine_props["topology"] = self.topology
        self.stats_obj.topology = self.topology
        self.sdn_obj.sdn_props.network_spectrum_dict = self.network_spectrum_dict
        self.sdn_obj.sdn_props.topology = self.topology

    def generate_requests(self, seed: int) -> None:
        """
        Call the request generator to generate requests.

        :param seed: The seed to use for the random generation
        :type seed: int
        """
        self.reqs_dict = get_requests(seed=seed, engine_props=self.engine_props)
        self.reqs_dict = dict(sorted(self.reqs_dict.items()))

    def handle_request(self, current_time: float, request_number: int) -> None:
        """
        Carry out arrival or departure functions for a given request.

        :param current_time: The current simulated time
        :type current_time: float
        :param request_number: The request number
        :type request_number: int
        """
        if self.reqs_dict is None or current_time not in self.reqs_dict:
            return

        request_type = self.reqs_dict[current_time]["request_type"]
        if request_type == "arrival":
            old_network_spectrum_dict = copy.deepcopy(self.network_spectrum_dict)
            old_request_info_dict = copy.deepcopy(self.reqs_dict[current_time])
            self.handle_arrival(current_time=current_time)

            if (
                self.engine_props["save_snapshots"]
                and request_number % self.engine_props["snapshot_step"] == 0
            ):
                self.stats_obj.update_snapshot(
                    network_spectrum_dict=self.network_spectrum_dict,
                    request_number=request_number,
                )

            if self.engine_props["output_train_data"]:
                was_routed = self.sdn_obj.sdn_props.was_routed
                if was_routed:
                    if self.reqs_dict is not None and current_time in self.reqs_dict:
                        request_info_dict = self.reqs_status_dict[
                            self.reqs_dict[current_time]["req_id"]
                        ]
                        if self.ml_metrics:
                            self.ml_metrics.update_train_data(
                                old_request_info_dict=old_request_info_dict,
                                request_info_dict=request_info_dict,
                                network_spectrum_dict=old_network_spectrum_dict,
                                current_transponders=self.stats_obj.current_transponders,
                            )

        elif request_type == "release":
            self.handle_release(current_time=current_time)
        else:
            raise NotImplementedError(
                f"Request type unrecognized. Expected arrival or release, "
                f"got: {request_type}"
            )

    def end_iter(
        self, iteration: int, print_flag: bool = True, base_file_path: str | None = None
    ) -> bool:
        """
        Update iteration statistics.

        :param iteration: The current iteration
        :type iteration: int
        :param print_flag: Whether to print or not
        :type print_flag: bool
        :param base_file_path: The base file path to save output statistics
        :type base_file_path: Optional[str]
        :return: Whether confidence interval has been reached
        :rtype: bool
        """
        self.stats_obj.calculate_blocking_statistics()
        self.stats_obj.finalize_iteration_statistics()
        # Some form of ML/RL is being used, ignore confidence intervals
        # for training and testing
        if not self.engine_props["is_training"]:
            resp = bool(self.stats_obj.calculate_confidence_interval())
        else:
            resp = False
        if (
            (iteration + 1) % self.engine_props["print_step"] == 0
            or iteration == 0
            or (iteration + 1) == self.engine_props["max_iters"]
        ):
            # Use the reporter for output instead of metrics class
            if hasattr(self, "reporter"):
                self.reporter.report_iteration_stats(
                    iteration=iteration,
                    max_iterations=self.engine_props["max_iters"],
                    erlang=self.engine_props["erlang"],
                    blocking_list=self.stats_obj.stats_props.simulation_blocking_list,
                    print_flag=print_flag,
                )

        if (
            (iteration + 1) % self.engine_props["save_step"] == 0
            or iteration == 0
            or (iteration + 1) == self.engine_props["max_iters"]
        ):
            self._save_all_stats(base_file_path or "data")

        return resp

    def init_iter(
        self,
        iteration: int,
        seed: int | None = None,
        print_flag: bool = True,
        trial: int | None = None,
    ) -> None:
        """
        Initialize an iteration.

        :param iteration: The current iteration number
        :type iteration: int
        :param seed: The seed to use for the random generation
        :type seed: Optional[int]
        :param print_flag: Flag to determine printing iter info
        :type print_flag: bool
        :param trial: The trial number
        :type trial: Optional[int]
        """
        if trial is not None:
            self.engine_props["thread_num"] = f"s{trial + 1}"

        self.iteration = iteration

        self.stats_obj.iteration = iteration
        self.stats_obj.init_iter_stats()

        for link_key in self.network_spectrum_dict:
            self.network_spectrum_dict[link_key]["usage_count"] = 0
            self.network_spectrum_dict[link_key]["throughput"] = 0

        # To prevent incomplete saves
        try:
            signal.signal(signal.SIGINT, self._signal_save_handler)
            signal.signal(signal.SIGTERM, self._signal_save_handler)
        except ValueError:
            # Signal only works in the main thread
            pass

        if iteration == 0 and print_flag:
            logger.info(
                "Simulation started for Erlang: %s simulation number: %s",
                self.engine_props["erlang"],
                self.engine_props["thread_num"],
            )

            if self.engine_props["deploy_model"]:
                self.ml_model = load_model(engine_properties=self.engine_props)

        # You can pass a list of seeds, a constant seed, or default to iteration number
        if seed is None:
            seed = (
                self.engine_props["seeds"][iteration]
                if self.engine_props["seeds"]
                else iteration + 1
            )

        self.generate_requests(seed)

    def run(self, seed: int | None = None) -> int:
        """
        Run the simulation by creating the topology and processing requests.

        This method creates the topology, processes requests, and sends
        iteration-based updates to the parent's queue.

        :param seed: Optional seed for random generation
        :type seed: Optional[int]
        :return: Number of completed iteration units
        :rtype: int
        """
        simulation_context = self._setup_simulation_context()
        self._log_simulation_start(simulation_context)

        for iteration in range(self.engine_props["max_iters"]):
            if self._should_stop_simulation(simulation_context):
                break

            simulation_context["done_units"] = self._run_single_iteration(
                iteration, seed, simulation_context
            )

            if simulation_context["end_iter"]:
                break

        self._log_simulation_complete(simulation_context)
        return int(simulation_context["done_units"])

    def _setup_simulation_context(self) -> dict[str, Any]:
        """
        Initialize simulation context with necessary parameters.

        :return: Dictionary containing simulation context parameters
        :rtype: Dict[str, Any]
        """
        self.create_topology()

        return {
            "log_queue": self.engine_props.get("log_queue"),
            "max_iters": self.engine_props["max_iters"],
            "progress_queue": self.engine_props.get("progress_queue"),
            "thread_num": self.engine_props.get("thread_num", "unknown"),
            "my_iteration_units": self.engine_props.get(
                "my_iteration_units", self.engine_props["max_iters"]
            ),
            "done_offset": self.engine_props.get("done_offset", 0),
            "done_units": self.engine_props.get("done_offset", 0),
            "end_iter": False,
        }

    def _log_simulation_start(self, context: dict[str, Any]) -> None:
        """
        Log simulation start message.

        :param context: Simulation context dictionary
        :type context: Dict[str, Any]
        """
        log_message(
            message=(
                f"[Engine] thread={context['thread_num']}, "
                f"offset={context['done_offset']}, "
                f"my_iteration_units={context['my_iteration_units']}, "
                f"erlang={self.engine_props['erlang']}\n"
            ),
            log_queue=context["log_queue"],
        )

    def _should_stop_simulation(self, context: dict[str, Any]) -> bool:
        """
        Check if simulation should be stopped.

        :param context: Simulation context dictionary
        :type context: Dict[str, Any]
        :return: True if simulation should stop
        :rtype: bool
        """
        if self.stop_flag is not None and self.stop_flag.is_set():
            log_message(
                message=(
                    f"Simulation stopped for Erlang: {self.engine_props['erlang']} "
                    f"simulation number: {context['thread_num']}.\n"
                ),
                log_queue=context["log_queue"],
            )
            return True
        return False

    def _run_single_iteration(
        self, iteration: int, seed: int | None, context: dict[str, Any]
    ) -> int:
        """
        Execute a single simulation iteration.

        :param iteration: Current iteration number
        :type iteration: int
        :param seed: Random seed for request generation
        :type seed: Optional[int]
        :param context: Simulation context dictionary
        :type context: Dict[str, Any]
        :return: Updated done_units count
        :rtype: int
        """
        self.init_iter(iteration=iteration, seed=seed)
        self._process_all_requests()

        context["end_iter"] = self.end_iter(iteration=iteration)
        context["done_units"] += 1

        self._update_progress(iteration, context)
        time.sleep(0.2)

        return int(context["done_units"])

    def _process_all_requests(self) -> None:
        """Process all requests for the current iteration."""
        request_number = 1
        if self.reqs_dict is None:
            return
        for current_time in self.reqs_dict:
            self.handle_request(
                current_time=current_time, request_number=request_number
            )

            if (
                self.reqs_dict is not None
                and current_time in self.reqs_dict
                and self.reqs_dict[current_time]["request_type"] == "arrival"
            ):
                request_number += 1

    def _update_progress(self, iteration: int, context: dict[str, Any]) -> None:
        """
        Update progress tracking and logging.

        :param iteration: Current iteration number
        :type iteration: int
        :param context: Simulation context dictionary
        :type context: Dict[str, Any]
        """
        if context["progress_queue"]:
            context["progress_queue"].put(
                (context["thread_num"], context["done_units"])
            )

        log_message(
            message=(
                f"CHILD={context['thread_num']} iteration={iteration}, "
                f"done_units={context['done_units']}\n"
            ),
            log_queue=context["log_queue"],
        )

    def _log_simulation_complete(self, context: dict[str, Any]) -> None:
        """
        Log simulation completion message.

        :param context: Simulation context dictionary
        :type context: Dict[str, Any]
        """
        log_message(
            message=(
                f"Simulation finished for Erlang: {self.engine_props['erlang']} "
                f"finished for simulation number: {context['thread_num']}.\n"
            ),
            log_queue=context["log_queue"],
        )

    def _save_all_stats(self, base_file_path: str = "data") -> None:
        """
        Save all statistics using the persistence module.

        :param base_file_path: Base path for output files
        :type base_file_path: str
        """
        # Create save dictionary with iteration stats
        save_dict: dict[str, Any] = {"iter_stats": {}}

        # Get blocking statistics from metrics
        blocking_stats = self.stats_obj.get_blocking_statistics()

        # Save main statistics
        self.persistence.save_stats(
            stats_dict=save_dict,
            stats_props=self.stats_obj.stats_props,
            blocking_stats=blocking_stats,
            base_file_path=base_file_path,
        )

        # Save ML training data if available
        if self.ml_metrics:
            self.ml_metrics.save_train_data(
                iteration=self.stats_obj.iteration or 0,
                max_iterations=self.engine_props["max_iters"],
                base_file_path=base_file_path,
            )

    def _signal_save_handler(self, signum: int, frame: Any) -> None:  # pylint: disable=unused-argument
        """
        Handle save operation when receiving signals.

        :param signum: Signal number
        :param frame: Current stack frame
        """
        logger.warning("Received signal %d, saving statistics...", signum)
        self._save_all_stats()
        logger.info("Statistics saved due to signal")
