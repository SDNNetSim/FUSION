# fusion/sim/run_simulation.py

# import multiprocessing
from datetime import datetime
from fusion.sim.network_simulator import NetworkSimulator  # we'll move class here

def run_simulation(config_dict):
    """
    Entry point for CLI. Kicks off one-thread simulation with config_dict.
    If you want multi-threaded erlang later, extend this.
    """
    # TODO: Why isn't this used?
    # stop_flag = multiprocessing.Event()
    sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")

    # Wrap into the old sims_dict format
    sims_dict = {
        's1': {
            **config_dict,
            'sim_start': sim_start
        }
    }

    # Create and launch the simulation
    simulator = NetworkSimulator()
    simulator.run_sim(thread_num='s1', thread_params=sims_dict['s1'], sim_start=sim_start)
