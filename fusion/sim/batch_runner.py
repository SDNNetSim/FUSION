# fusion/sim/batch_runner.py

from fusion.core.simulation import SimulationEngine
from fusion.core.request import get_requests
from fusion.cli.config_setup import ConfigManager


def run_simulation(args):
    """
    Runs a generic simulation (no AI).
    """
    config = ConfigManager.from_args(args)
    engine = SimulationEngine(engine_props=config.get())
    engine.reqs_dict = get_requests(seed=config.get()["seed"], engine_props=config.get())

    for curr_time in sorted(engine.reqs_dict.keys()):
        req_type = engine.reqs_dict[curr_time]["request_type"]
        if req_type == "arrival":
            engine.handle_arrival(curr_time)
        elif req_type == "release":
            engine.sdn_obj.release()
        engine.update_arrival_params(curr_time)

    engine.stats_obj.save_stats(base_fp="data")
