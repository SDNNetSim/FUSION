# fusion/sim/run_simulation.py

import json
import multiprocessing
from datetime import datetime
from fusion.sim.network_simulator import NetworkSimulator  # we'll move class here


def run_simulation(config_dict):
    """
    Entry point for CLI. Kicks off one-thread simulation with config_dict.
    If you want multi-threaded erlang later, extend this.
    """
    stop_flag = multiprocessing.Event()
    sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")

    # Wrap into the old sims_dict format
    config_s1 = config_dict.get('s1', {})

    # Add default values for parameters that might be missing but accessed by simulation
    defaults = {
        'seeds': None,  # Use default seed behavior (iteration + 1)
        'request_distribution': {"100": 1.0},  # Default single bandwidth
        'max_segments': 4,
        'fixed_grid': False,
        'pre_calc_mod_selection': False,
        # SNR parameters
        'beta': 0.5,
        'theta': 0.0,
        'phi': {"QPSK": 1},
        'xt_noise': False,
        'requested_xt': {"QPSK": -26.19},
        'xt_type': 'without_length',
        # RL/ML parameters that might be accessed
        'is_training': False,
        'ml_training': False,
        'output_train_data': False,
        'ml_model': None,
    }

    # Parse JSON strings from config into dictionaries
    def parse_json_param(value):
        if isinstance(value, str) and value.startswith('{'):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        return value

    # Apply JSON parsing to config values
    parsed_config = {}
    for key, value in config_s1.items():
        if key in ['request_distribution', 'phi', 'requested_xt']:
            parsed_config[key] = parse_json_param(value)
        else:
            parsed_config[key] = value

    sims_dict = {
        's1': {
            **defaults,
            **parsed_config,  # Parsed config values override defaults
            'sim_start': sim_start,
        }
    }

    # Create and launch the simulation with proper stop_flag
    simulator = NetworkSimulator()
    sims_dict['s1']['stop_flag'] = stop_flag
    simulator.run_sim(thread_num='s1', thread_params=sims_dict['s1'], sim_start=sim_start)


def run_simulation_pipeline(args, stop_flag=None):  # pylint: disable=unused-argument
    """
    Pipeline function for running simulations from CLI.

    Args:
        args: Parsed command line arguments
        stop_flag: Optional threading stop flag for cancellation (currently unused)
    """
    from fusion.cli.config_setup import load_and_validate_config  # pylint: disable=import-outside-toplevel

    # Convert args to config dictionary
    config_dict = load_and_validate_config(args)

    # Run the simulation (stop_flag not yet implemented in legacy code)
    run_simulation(config_dict)
