# run_sim.py
import time
import multiprocessing
import copy
from datetime import datetime
import concurrent.futures

from helper_scripts.setup_helpers import create_input, save_input
from src.engine import Engine
from config_scripts.setup_config import read_config
from config_scripts.parse_args import parse_args

class NetworkSimulator:
    """
    Controls all simulations for this project.
    """
    def __init__(self):
        # Contains all the desired network simulator parameters for every simulation.
        self.properties = None

    def _run_generic_sim(self, erlang: float, first_erlang: bool, erlang_index: int,
                         total_erlangs: int, progress_dict):
        import copy
        from helper_scripts.setup_helpers import create_input, save_input
        from src.engine import Engine

        # Temporarily remove unpickleable keys from self.properties before deepcopy
        unpickleable_keys = {}
        for key in ['progress_dict', 'log_queue']:
            if key in self.properties:
                unpickleable_keys[key] = self.properties.pop(key)

        # Perform deepcopy on the cleaned self.properties
        engine_props = copy.deepcopy(self.properties)

        # Restore the removed keys to self.properties for future use
        self.properties.update(unpickleable_keys)

        # Inject the unpickleable objects into engine_props from the current context
        engine_props['progress_dict'] = progress_dict
        if 'log_queue' in unpickleable_keys:
            engine_props['log_queue'] = unpickleable_keys['log_queue']

        # Set additional simulation-specific parameters
        engine_props['arrival_rate'] = (engine_props['cores_per_link'] * erlang) / engine_props['holding_time']
        engine_props['erlang'] = erlang
        engine_props['erlang_index'] = erlang_index
        engine_props['total_erlangs'] = total_erlangs
        engine_props['progress_key'] = erlang_index
        engine_props['band_list'] = list()

        # Create a sanitized copy for saving/configuration updates.
        # Remove keys that are not JSON serializable.
        clean_engine_props = engine_props.copy()
        clean_engine_props.pop('progress_dict', None)
        clean_engine_props.pop('progress_key', None)
        clean_engine_props.pop('log_queue', None)

        # Update topology and other details.
        updated_props = create_input(base_fp='data', engine_props=clean_engine_props)
        engine_props.update(updated_props)

        if first_erlang:
            save_input(
                base_fp='data',
                properties=clean_engine_props,  # Use the sanitized dictionary for saving.
                file_name=f"sim_input_{updated_props['thread_num']}.json",
                data_dict=updated_props
            )

        engine = Engine(engine_props=engine_props)
        engine.run()

    def run_generic_sim(self):
        erlang_dict = self.properties['erlangs']
        start, stop, step = erlang_dict['start'], erlang_dict['stop'], erlang_dict['step']
        erlang_list = [float(erlang) for erlang in range(start, stop, step)]
        total_erlangs = len(erlang_list)

        # Use the shared dictionary if provided; else create a new one.
        if 'progress_dict' in self.properties:
            progress_dict = self.properties['progress_dict']
        else:
            manager = multiprocessing.Manager()
            progress_dict = manager.dict()
            for erlang_index, erlang in enumerate(erlang_list):
                progress_dict[erlang_index] = 0

        # Initialize all keys to 0.
        for erlang_index, erlang in enumerate(erlang_list):
            progress_dict[erlang_index] = 0

        if self.properties['thread_erlangs']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures_list = []
                for erlang_index, erlang in enumerate(erlang_list):
                    first_erlang = (erlang_index == 0)
                    future = executor.submit(self._run_generic_sim,
                                             erlang=erlang,
                                             first_erlang=first_erlang,
                                             erlang_index=erlang_index,
                                             total_erlangs=total_erlangs,
                                             progress_dict=progress_dict)
                    futures_list.append(future)
                for future in concurrent.futures.as_completed(futures_list):
                    future.result()
        else:
            for erlang_index, erlang in enumerate(erlang_list):
                first_erlang = (erlang_index == 0)
                self._run_generic_sim(erlang=erlang,
                                      first_erlang=first_erlang,
                                      erlang_index=erlang_index,
                                      total_erlangs=total_erlangs,
                                      progress_dict=progress_dict)

    def run_sim(self, **kwargs):
        self.properties = kwargs['thread_params']
        # The date and current time derived from the simulation start.
        self.properties['date'] = kwargs['sim_start'].split('_')[0]
        tmp_list = kwargs['sim_start'].split('_')
        time_string = f'{tmp_list[1]}_{tmp_list[2]}_{tmp_list[3]}_{tmp_list[4]}'
        self.properties['sim_start'] = time_string
        self.properties['thread_num'] = kwargs['thread_num']
        self.run_generic_sim()

def run(sims_dict: dict):
    from datetime import datetime
    import concurrent.futures

    # Retrieve the log_queue from the simulation config
    # (Assuming all simulation configs have the same log_queue)
    any_conf = list(sims_dict.values())[0]
    log_queue = any_conf.get('log_queue')

    def log(message):
        if log_queue:
            log_queue.put(message)
        else:
            print(message)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures_list = []
        sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
        for thread_num, thread_params in sims_dict.items():
            curr_sim = NetworkSimulator()
            class_inst = curr_sim.run_sim
            log(f"Starting simulation for thread {thread_num} at {sim_start}.")
            future = executor.submit(class_inst, thread_num=thread_num, thread_params=thread_params, sim_start=sim_start)
            futures_list.append(future)
        for future in concurrent.futures.as_completed(futures_list):
            future.result()


if __name__ == '__main__':
    args_dict = parse_args()
    all_sims_dict = read_config(args_dict=args_dict, config_path=args_dict['config_path'])
    run(sims_dict=all_sims_dict)
