"""
Batch simulation runner orchestrator.

This module provides the main orchestration layer for running batch simulations,
supporting both single and multi-threaded execution with various traffic loads.
"""

import multiprocessing
import time
from datetime import datetime
from typing import Any

from fusion.core.simulation import SimulationEngine
from fusion.sim.input_setup import create_input
from fusion.utils.logging_config import log_message as _log_message


def log_message(message: str) -> None:
    """Wrapper for log_message that handles missing queue."""
    _log_message(message, None)


class BatchRunner:
    """
    Orchestrates batch simulation execution with support for:
    - Multiple traffic loads (Erlang values)
    - Parallel execution across multiple cores
    - Progress tracking and reporting
    - Result aggregation
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize batch runner with configuration.

        :param config: Configuration dictionary containing simulation parameters
        :type config: dict[str, Any]
        """
        self.config = config
        self.date = datetime.now().strftime("%m%d")
        self.sim_start = datetime.now().strftime("%H_%M_%S_%f")
        self.manager = multiprocessing.Manager()
        self.progress_dict = self.manager.dict()
        self.results: list[dict] = []

        # Configure logging based on config settings
        sim_params = config.get("s1", config)
        log_level = sim_params.get("log_level", "INFO")
        if isinstance(log_level, str):
            from fusion.utils.logging_config import set_global_log_level

            set_global_log_level(log_level)

    def prepare_simulation(self, sim_params: dict) -> dict:
        """
        Prepare simulation parameters and create necessary input files.

        :param sim_params: Base simulation parameters
        :type sim_params: dict
        :return: Updated simulation parameters with input data
        :rtype: dict
        """
        # Create input data and topology
        # Extract base_fp from sim_params or use default
        # Get base path from configuration or use project default
        base_fp = sim_params.get("base_fp", "data")

        # Set required fields for create_input
        if "thread_num" not in sim_params:
            sim_params["thread_num"] = "s1"  # Default thread identifier
        if "date" not in sim_params:
            sim_params["date"] = self.date
        if "sim_start" not in sim_params:
            sim_params["sim_start"] = self.sim_start

        sim_params = create_input(base_fp=base_fp, engine_props=sim_params)

        # Save input files if requested
        if sim_params.get("save_files", True):
            pass  # Save input implementation pending

        # Validate bandwidth consistency
        self._validate_bandwidth_config(sim_params)

        return sim_params

    def run_single_erlang(
        self,
        erlang: float,
        sim_params: dict,
        erlang_index: int,
        total_erlangs: int,
    ) -> dict:
        """
        Run simulation for a single Erlang load value.

        :param erlang: Traffic load in Erlangs
        :type erlang: float
        :param sim_params: Simulation parameters
        :type sim_params: dict
        :param erlang_index: Index of current Erlang in the list
        :type erlang_index: int
        :param total_erlangs: Total number of Erlang values
        :type total_erlangs: int
        :return: Dictionary containing simulation results
        :rtype: dict
        """
        # Update parameters for this Erlang value
        current_params = sim_params.copy()
        current_params["erlang"] = erlang
        current_params["arrival_rate"] = erlang / current_params["holding_time"]

        # Add stop_flag if not present (required by SimulationEngine)
        if "stop_flag" not in current_params:
            # Create a dummy event that's never set for single-threaded execution
            current_params["stop_flag"] = multiprocessing.Event()

        # Add seeds if not present (required by SimulationEngine)
        if "seeds" not in current_params:
            current_params["seeds"] = None  # Will use iteration number as seed

        # Log simulation start message
        log_message(f"Starting simulation for {erlang} Erlang (load {erlang_index + 1}/{total_erlangs})")

        # Create and run simulation engine
        engine = SimulationEngine(current_params)
        # Note: done_offset would be used for progress tracking in future implementation

        # Run simulation
        start_time = time.time()
        results = engine.run()
        elapsed_time = time.time() - start_time

        # Log completion
        log_message(f"Completed {erlang} Erlang in {elapsed_time:.2f} seconds")

        # Get stats from engine
        stats = {}
        if hasattr(engine, "stats_obj") and engine.stats_obj:
            try:
                stats = engine.stats_obj.__dict__
            except AttributeError:
                stats = {"blocking_probability": 0, "total_requests": 0}

        return {
            "erlang": erlang,
            "results": results,
            "elapsed_time": elapsed_time,
            "stats": stats,
        }

    def run_parallel_batch(
        self,
        erlangs: list[float],
        sim_params: dict,
        num_processes: int | None = None,
    ) -> list[dict]:
        """
        Run multiple Erlang simulations in parallel.

        :param erlangs: List of Erlang values to simulate
        :type erlangs: list[float]
        :param sim_params: Base simulation parameters
        :type sim_params: dict
        :param num_processes: Number of parallel processes (None for CPU count)
        :type num_processes: int | None
        :return: List of result dictionaries
        :rtype: list[dict]
        """
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        log_message(f"Running batch simulation with {len(erlangs)} loads across {num_processes} processes")

        with multiprocessing.Pool(processes=num_processes) as pool:
            # Create tasks for parallel execution
            tasks = []
            for i, erlang in enumerate(erlangs):
                task_params = (erlang, sim_params.copy(), i, len(erlangs))
                tasks.append(task_params)

            # Execute in parallel
            results = pool.starmap(self.run_single_erlang, tasks)

        return results

    def run_sequential_batch(self, erlangs: list[float], sim_params: dict) -> list[dict]:
        """
        Run multiple Erlang simulations sequentially.

        :param erlangs: List of Erlang values to simulate
        :type erlangs: list[float]
        :param sim_params: Base simulation parameters
        :type sim_params: dict
        :return: List of result dictionaries
        :rtype: list[dict]
        """
        results = []

        for i, erlang in enumerate(erlangs):
            result = self.run_single_erlang(erlang, sim_params, i, len(erlangs))
            results.append(result)

        return results

    def run(self, parallel: bool = False, num_processes: int | None = None) -> list[dict]:
        """
        Execute batch simulation run.

        :param parallel: Whether to run simulations in parallel
        :type parallel: bool
        :param num_processes: Number of parallel processes (if parallel=True)
        :type num_processes: int | None
        :return: List of simulation results
        :rtype: list[dict]
        """
        # Extract configuration
        sim_params = self.config.get("s1", self.config)
        sim_params["sim_start"] = self.sim_start

        # Prepare simulation
        sim_params = self.prepare_simulation(sim_params)

        # Get Erlang values
        erlangs = self._get_erlang_values(sim_params)

        # Initialize progress tracking
        total_work = len(erlangs) * sim_params.get("sim_thread_erlangs", 1)
        self.progress_dict["total"] = total_work
        self.progress_dict["done"] = 0

        # Run simulations
        if parallel and len(erlangs) > 1:
            results = self.run_parallel_batch(erlangs, sim_params, num_processes)
        else:
            results = self.run_sequential_batch(erlangs, sim_params)

        # Store results
        self.results = results

        # Log summary
        self._log_summary(results)

        return results

    def _get_erlang_values(self, sim_params: dict) -> list[float]:
        """
        Extract Erlang values from configuration.

        Uses erlang_start, erlang_stop, and erlang_step parameters.
        """
        start = float(sim_params["erlang_start"])
        stop = float(sim_params["erlang_stop"])
        step = float(sim_params.get("erlang_step", 100.0))

        # Generate erlang list inclusively (stop value is included)
        erlangs = []
        current = start
        while current <= stop:
            erlangs.append(current)
            current += step
        return erlangs

    def _validate_bandwidth_config(self, sim_params: dict) -> None:
        """Validate bandwidth configuration consistency."""
        request_distribution = sim_params.get("request_distribution", {})
        mod_per_bw = sim_params.get("mod_per_bw", {})

        if not request_distribution or not mod_per_bw:
            return

        # Check bandwidth availability
        missing = [bw for bw in request_distribution if bw not in mod_per_bw]

        if missing:
            available = list(mod_per_bw.keys())
            raise ValueError(f"Bandwidth mismatch: {missing} not in mod_per_bw. Available: {available}")

    def _log_summary(self, results: list[dict]) -> None:
        """Log summary of batch run results."""
        log_message("=" * 60)
        log_message("BATCH SIMULATION SUMMARY")
        log_message("=" * 60)

        for result in results:
            erlang = result["erlang"]
            stats = result["stats"]
            elapsed = result["elapsed_time"]

            log_message(f"\nErlang {erlang}:")
            log_message(f"  Time: {elapsed:.2f}s")
            log_message(f"  Blocking: {stats.get('blocking_probability', 0):.4f}")
            log_message(f"  Requests: {stats.get('total_requests', 0)}")

        total_time = sum(r["elapsed_time"] for r in results)
        log_message(f"\nTotal execution time: {total_time:.2f}s")
        log_message("=" * 60)


def run_batch_simulation(config: dict, parallel: bool = False, num_processes: int | None = None) -> list[dict]:
    """
    Convenience function to run batch simulation.

    :param config: Simulation configuration
    :type config: dict
    :param parallel: Whether to run in parallel
    :type parallel: bool
    :param num_processes: Number of parallel processes
    :type num_processes: int | None
    :return: List of simulation results
    :rtype: list[dict]
    """
    runner = BatchRunner(config)
    return runner.run(parallel=parallel, num_processes=num_processes)


def run_multi_seed_experiment(config: dict[str, Any], seed_list: list[int], output_dir: str = "results") -> list[dict[str, Any]]:
    """
    Run simulation with multiple seeds for statistical analysis.

    Executes the same simulation configuration with different random seeds
    to enable variance analysis and statistical significance testing.

    :param config: Base configuration
    :type config: dict[str, Any]
    :param seed_list: List of seeds to run
    :type seed_list: list[int]
    :param output_dir: Output directory for results
    :type output_dir: str
    :return: List of result dictionaries
    :rtype: list[dict[str, Any]]

    Example:
        >>> config = load_config('survivability_experiment.ini')
        >>> seeds = [42, 43, 44, 45, 46]
        >>> results = run_multi_seed_experiment(config, seeds, 'results/')
        >>> print(len(results))
        5
    """
    from pathlib import Path

    from fusion.core.simulation import seed_all_rngs

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for seed in seed_list:
        log_message(f"Running simulation with seed {seed}")

        # Create seed-specific config
        seed_config = config.copy()
        seed_config["seed"] = seed

        # Seed all RNGs before creating engine
        seed_all_rngs(seed)

        # Run simulation
        engine = SimulationEngine(seed_config)
        return_code = engine.run(seed=seed)

        # Store results
        result = {
            "seed": seed,
            "return_code": return_code,
            # Add stats if available
            "stats": engine.stats_obj.to_dict() if hasattr(engine.stats_obj, "to_dict") else {},
        }
        results.append(result)

        stats_dict = result.get("stats", {})
        bp = stats_dict.get("blocking_probability", 0) if isinstance(stats_dict, dict) else 0
        log_message(f"Seed {seed} complete: BP={bp:.4f}")

    return results
