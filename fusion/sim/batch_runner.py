# TODO: Haven't used yet

# fusion/sim/batch_runner.py

from fusion.core.simulation import SimulationEngine
from fusion.core.request import get_requests
from fusion.configs.config import ConfigManager


def run_simulation(config_path: str):
    config = ConfigManager.load(config_path)
    engine = SimulationEngine(engine_props=config)
    engine.reqs_dict = get_requests(seed=config['seed'], engine_props=config)

    # Simulation loop
    for curr_time in sorted(engine.reqs_dict.keys()):
        req_type = engine.reqs_dict[curr_time]["request_type"]
        if req_type == "arrival":
            engine.handle_arrival(curr_time)
        elif req_type == "release":
            engine.sdn_obj.release()
        engine.update_arrival_params(curr_time)

    engine.stats_obj.dump_results()
