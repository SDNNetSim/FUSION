"""
Batch simulation runner orchestrator.

This module provides the main orchestration layer for running batch simulations,
supporting both single and multi-threaded execution with various traffic loads.
"""

import multiprocessing
import time
from datetime import datetime
from typing import Dict, List, Optional

from fusion.sim.input_setup import create_input
from fusion.core.simulation import SimulationEngine
from fusion.sim.utils import log_message as _log_message


def log_message(message):
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

    def __init__(self, config: Dict[str, any]):
        """
        Initialize batch runner with configuration.

        Args:
            config: Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
        self.manager = multiprocessing.Manager()
        self.progress_dict = self.manager.dict()
        self.results = []

    def prepare_simulation(self, sim_params: Dict) -> Dict:
        """
        Prepare simulation parameters and create necessary input files.

        Args:
            sim_params: Base simulation parameters

        Returns:
            Updated simulation parameters with input data
        """
        # Create input data and topology
        # Extract base_fp from sim_params or use default
        base_fp = sim_params.get('base_fp', 'data')
        sim_params = create_input(base_fp=base_fp, engine_props=sim_params)

        # Save input files if requested
        if sim_params.get('save_files', True):
            # TODO: Implement proper save_input call when needed
            pass

        # Validate bandwidth consistency
        self._validate_bandwidth_config(sim_params)

        return sim_params

    def run_single_erlang(self, erlang: float, sim_params: Dict,
                          erlang_index: int, total_erlangs: int) -> Dict:
        """
        Run simulation for a single Erlang load value.

        Args:
            erlang: Traffic load in Erlangs
            sim_params: Simulation parameters
            erlang_index: Index of current Erlang in the lis
            total_erlangs: Total number of Erlang values

        Returns:
            Dictionary containing simulation results
        """
        # Update parameters for this Erlang value
        current_params = sim_params.copy()
        current_params['erlang'] = erlang
        current_params['arrival_rate'] = erlang / current_params['holding_time']

        # Log star
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
        if hasattr(engine, 'stats_obj') and engine.stats_obj:
            try:
                stats = engine.stats_obj.__dict__
            except AttributeError:
                stats = {'blocking_probability': 0, 'total_requests': 0}

        return {
            'erlang': erlang,
            'results': results,
            'elapsed_time': elapsed_time,
            'stats': stats
        }

    def run_parallel_batch(self, erlangs: List[float], sim_params: Dict,
                           num_processes: Optional[int] = None) -> List[Dict]:
        """
        Run multiple Erlang simulations in parallel.

        Args:
            erlangs: List of Erlang values to simulate
            sim_params: Base simulation parameters
            num_processes: Number of parallel processes (None for CPU count)

        Returns:
            List of result dictionaries
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

    def run_sequential_batch(self, erlangs: List[float], sim_params: Dict) -> List[Dict]:
        """
        Run multiple Erlang simulations sequentially.

        Args:
            erlangs: List of Erlang values to simulate
            sim_params: Base simulation parameters

        Returns:
            List of result dictionaries
        """
        results = []

        for i, erlang in enumerate(erlangs):
            result = self.run_single_erlang(erlang, sim_params, i, len(erlangs))
            results.append(result)

        return results

    def run(self, parallel: bool = False, num_processes: Optional[int] = None) -> List[Dict]:
        """
        Execute batch simulation run.

        Args:
            parallel: Whether to run simulations in parallel
            num_processes: Number of parallel processes (if parallel=True)

        Returns:
            List of simulation results
        """
        # Extract configuration
        sim_params = self.config.get('s1', self.config)
        sim_params['sim_start'] = self.sim_start

        # Prepare simulation
        sim_params = self.prepare_simulation(sim_params)

        # Get Erlang values
        erlangs = self._get_erlang_values(sim_params)

        # Initialize progress tracking
        total_work = len(erlangs) * sim_params.get('sim_thread_erlangs', 1)
        self.progress_dict['total'] = total_work
        self.progress_dict['done'] = 0

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

    def _get_erlang_values(self, sim_params: Dict) -> List[float]:
        """Extract Erlang values from configuration."""
        erlangs_str = sim_params.get('erlangs', '300')

        if ',' in erlangs_str:
            # Multiple values specified
            return [float(e.strip()) for e in erlangs_str.split(',')]
        if '-' in erlangs_str:
            # Range specified (start-end:step) or (start-end)
            parts = erlangs_str.split('-')
            start = float(parts[0])

            if ':' in parts[1]:
                # Has explicit step
                end_step = parts[1].split(':')
                end = float(end_step[0])
                step = float(end_step[1])
            else:
                # Default step
                end = float(parts[1])
                step = 100.0

            erlangs = []
            current = start
            while current <= end:
                erlangs.append(current)
                current += step
            return erlangs
        # Single value
        return [float(erlangs_str)]

    def _validate_bandwidth_config(self, sim_params: Dict):
        """Validate bandwidth configuration consistency."""
        request_distribution = sim_params.get('request_distribution', {})
        mod_per_bw = sim_params.get('mod_per_bw', {})

        if not request_distribution or not mod_per_bw:
            return

        # Check bandwidth availability
        missing = [bw for bw in request_distribution if bw not in mod_per_bw]

        if missing:
            available = list(mod_per_bw.keys())
            raise ValueError(
                f"Bandwidth mismatch: {missing} not in mod_per_bw. "
                f"Available: {available}"
            )

    def _log_summary(self, results: List[Dict]):
        """Log summary of batch run results."""
        log_message("=" * 60)
        log_message("BATCH SIMULATION SUMMARY")
        log_message("=" * 60)

        for result in results:
            erlang = result['erlang']
            stats = result['stats']
            elapsed = result['elapsed_time']

            log_message(f"\nErlang {erlang}:")
            log_message(f"  Time: {elapsed:.2f}s")
            log_message(f"  Blocking: {stats.get('blocking_probability', 0):.4f}")
            log_message(f"  Requests: {stats.get('total_requests', 0)}")

        total_time = sum(r['elapsed_time'] for r in results)
        log_message(f"\nTotal execution time: {total_time:.2f}s")
        log_message("=" * 60)


def run_batch_simulation(config: Dict, parallel: bool = False,
                         num_processes: Optional[int] = None) -> List[Dict]:
    """
    Convenience function to run batch simulation.

    Args:
        config: Simulation configuration
        parallel: Whether to run in parallel
        num_processes: Number of parallel processes

    Returns:
        List of simulation results
    """
    runner = BatchRunner(config)
    return runner.run(parallel=parallel, num_processes=num_processes)
