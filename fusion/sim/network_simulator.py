# fusion/sim/network_simulator.py

import multiprocessing
import copy
from datetime import datetime
from multiprocessing import Manager, Process

from fusion.helper_scripts.setup_helpers import create_input, save_input
from fusion.core.engine import Engine


class NetworkSimulator:
    """
    Controls all simulations for this project.
    """
    def __init__(self):
        self.properties = None

    def _run_generic_sim(self, erlang: float, first_erlang: bool, erlang_index: int,
                         progress_dict, done_offset: int):
        """
        Handles one Erlang in this single process. Sets arrival_rate,
        passes done_offset to avoid resetting progress between volumes,
        and calls Engine.run(), which returns how many iteration units are now done.
        """

        # Remove unpickleable keys so we can deepcopy
        unpickleable_keys = {}
        for key in ['log_queue', 'progress_queue', 'stop_flag']:
            if key in self.properties:
                unpickleable_keys[key] = self.properties.pop(key)

        engine_props = copy.deepcopy(self.properties)

        # Restore the queues
        self.properties.update(unpickleable_keys)
        if 'log_queue' in unpickleable_keys:
            engine_props['log_queue'] = unpickleable_keys['log_queue']
        if 'progress_queue' in unpickleable_keys:
            engine_props['progress_queue'] = unpickleable_keys['progress_queue']
        if 'stop_flag' in unpickleable_keys:
            engine_props['stop_flag'] = unpickleable_keys['stop_flag']

        # Insert progress tracking
        engine_props['progress_dict'] = progress_dict
        engine_props['progress_key'] = erlang_index

        # Set 'erlang' and 'arrival_rate'
        engine_props['erlang'] = erlang
        engine_props['arrival_rate'] = (engine_props['cores_per_link'] * erlang) / engine_props['holding_time']

        # Pass how many units of work have been done so far
        engine_props['done_offset'] = done_offset

        engine_props['my_iteration_units'] = self.properties.get(
            'my_iteration_units',
            engine_props['max_iters']
        )

        # Create sanitized copy for saving
        clean_engine_props = engine_props.copy()
        for badkey in ['progress_dict', 'progress_key', 'log_queue', 'progress_queue', 'done_offset', 'stop_flag']:
            clean_engine_props.pop(badkey, None)

        updated_props = create_input(base_fp='data', engine_props=clean_engine_props)
        engine_props.update(updated_props)

        # Save input if first Erlang
        if first_erlang:
            save_input(
                base_fp='data',
                properties=clean_engine_props,
                file_name=f"sim_input_{updated_props['thread_num']}.json",
                data_dict=updated_props
            )

        print(f"[Simulation {erlang_index}] progress_dict id: {id(engine_props['progress_dict'])}")

        engine = Engine(engine_props=engine_props)
        final_done_units = engine.run()

        return final_done_units

    def run_generic_sim(self):
        """
        Run multiple Erlangs sequentially in this single process.
        """
        start, stop = self.properties['erlang_start'], self.properties['erlang_stop']
        step = self.properties['erlang_step']
        erlang_list = [float(x) for x in range(start, stop, step)]
        print("Launching simulations for erlangs:", erlang_list)

        max_iters = self.properties['max_iters']
        my_iteration_units = len(erlang_list) * max_iters
        self.properties['my_iteration_units'] = my_iteration_units

        if 'progress_dict' not in self.properties:
            manager = Manager()
            self.properties['progress_dict'] = manager.dict()

        progress_dict = self.properties['progress_dict']
        print("Initial shared progress dict:", dict(progress_dict))

        done_units_so_far = 0

        for erlang_index, erlang in enumerate(erlang_list):
            first_erlang = erlang_index == 0

            done_units_so_far = self._run_generic_sim(
                erlang=erlang,
                first_erlang=first_erlang,
                erlang_index=erlang_index,
                progress_dict=progress_dict,
                done_offset=done_units_so_far
            )

    def run_sim(self, **kwargs):
        """
        Sets up internal state and triggers the run_generic_sim().
        """
        self.properties = kwargs['thread_params']
        self.properties['date'] = kwargs['sim_start'].split('_')[0]

        tmp_list = kwargs['sim_start'].split('_')
        time_string = f'{tmp_list[1]}_{tmp_list[2]}_{tmp_list[3]}_{tmp_list[4]}'
        self.properties['sim_start'] = time_string
        self.properties['thread_num'] = kwargs['thread_num']

        self.run_generic_sim()


def run(sims_dict: dict, stop_flag: multiprocessing.Event):
    """
    Spawns one process per simulation config, each handling multiple Erlangs.
    """
    any_conf = list(sims_dict.values())[0]
    log_queue = any_conf.get('log_queue')
    progress_queue = any_conf.get('progress_queue')

    def log(message):
        if log_queue:
            log_queue.put(message)
        else:
            print(message)

    processes = []

    if 'sim_start' not in sims_dict['s1']:
        sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
    else:
        sim_start = f"{sims_dict['s1']['date']}_{sims_dict['s1']['sim_start']}"

    for thread_num, thread_params in sims_dict.items():
        thread_params['progress_queue'] = progress_queue
        thread_params['stop_flag'] = stop_flag

        log(f"Starting simulation for thread {thread_num} at {sim_start}.")
        curr_sim = NetworkSimulator()

        p = Process(
            target=curr_sim.run_sim,
            kwargs={
                'thread_num': thread_num,
                'thread_params': thread_params,
                'sim_start': sim_start
            }
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
