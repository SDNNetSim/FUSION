# fusion/sim/run_simulation.py

from fusion.sim.batch_runner import run_batch_simulation


def run_simulation(config_dict):
    """
    Legacy entry point maintained for backward compatibility.
    New code should use run_batch_simulation directly.
    """
    # Use the new batch runner with single-threaded execution
    results = run_batch_simulation(config_dict, parallel=False)

    # Return first result for compatibility
    return results[0] if results else None


def run_simulation_pipeline(args, stop_flag=None):  # pylint: disable=unused-argument
    """
    Pipeline function for running simulations from CLI.
    Now uses the new batch_runner orchestrator.

    Args:
        args: Parsed command line arguments
        stop_flag: Optional threading stop flag for cancellation
    """
    from fusion.cli.config_setup import load_and_validate_config  # pylint: disable=import-outside-toplevel

    # Convert args to config dictionary
    config_dict = load_and_validate_config(args)

    # Determine if parallel execution is requested
    parallel = args.parallel if hasattr(args, 'parallel') else False
    num_processes = args.num_processes if hasattr(args, 'num_processes') else None

    # Run using the new batch runner
    results = run_batch_simulation(config_dict, parallel=parallel, num_processes=num_processes)

    return results
