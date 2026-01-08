import multiprocessing
import copy
import time
import os  # Added import
import traceback  # Added import
from datetime import datetime
from multiprocessing import Manager, Process

# Import helpers
from helper_scripts.setup_helpers import create_input, save_input
from src.engine import Engine
from config_scripts.setup_config import read_config
from config_scripts.parse_args import parse_args


class NetworkSimulator:
    """
    Controls all simulations for this project.
    """
    def __init__(self):
        self.properties = None

    def _run_generic_sim(self, erlang: float, first_erlang: bool, erlang_index: int,
                         progress_dict, done_offset: int):
        """
        Handles one Erlang in this single process.
        """
        # Remove unpickleable keys so we can deepcopy
        unpickleable_keys = {}
        for key in ['log_queue', 'progress_queue', 'stop_flag']:
            if key in self.properties:
                unpickleable_keys[key] = self.properties.pop(key)

        engine_props = copy.deepcopy(self.properties)

        # Restore the queues
        self.properties.update(unpickleable_keys)

        # Restore flags/queues to engine_props
        if 'log_queue' in unpickleable_keys:
            engine_props['log_queue'] = unpickleable_keys['log_queue']
        if 'progress_queue' in unpickleable_keys:
            engine_props['progress_queue'] = unpickleable_keys['progress_queue']
        if 'stop_flag' in unpickleable_keys:
            engine_props['stop_flag'] = unpickleable_keys['stop_flag']

        # Setup progress tracking
        engine_props['progress_dict'] = progress_dict
        engine_props['progress_key'] = erlang_index

        # Set 'erlang' and 'arrival_rate'
        engine_props['erlang'] = erlang
        engine_props['arrival_rate'] = (engine_props['cores_per_link'] * erlang) / engine_props['holding_time']
        engine_props['done_offset'] = done_offset
        engine_props['my_iteration_units'] = self.properties.get('my_iteration_units', engine_props['max_iters'])

        # --- PATHS ARE ALREADY SET ---
        # The engine_props now contain the correct file paths because we updated
        # the main config in Step A below.
        
        # Override thread_num to empty string to ensure flat directory output
        # (Engine typically uses thread_num to create subfolders like s1_E50/)
        engine_props['thread_num'] = '' 

        print(f"[Process {self.properties.get('sim_id', 'Unknown')}] Running Erlang: {erlang}...")

        # --- UPDATE START: Safe Directory Creation & Error Handling ---
        try:
            # 1. Force Create Output Directory
            # Constructs path like: output/USNet/0108/00_47_52_...
            # This ensures the folder exists BEFORE the engine tries to save into it.
            topology = engine_props.get('topology', 'UnknownTopology')
            date_str = engine_props.get('date', 'UnknownDate')
            sim_start = engine_props.get('sim_start', 'UnknownTime')
            
            output_dir = os.path.join('output', topology, date_str, sim_start)
            os.makedirs(output_dir, exist_ok=True)

            # 2. Run Engine
            engine = Engine(engine_props=engine_props)
            final_done_units = engine.run()
            
            return final_done_units

        except Exception as e:
            print(f"!!! CRITICAL ERROR in Process {self.properties.get('sim_id', 'Unknown')} !!!")
            print(f"Failed while running Erlang: {erlang}")
            traceback.print_exc()
            return 0
        # --- UPDATE END ---

    def run_generic_sim(self):
        """
        Runs the assigned Erlang(s). 
        """
        start, stop = self.properties['erlang_start'], self.properties['erlang_stop']
        step = self.properties['erlang_step']
        
        erlang_list = [float(x) for x in range(int(start), int(stop), int(step))]
        
        max_iters = self.properties['max_iters']
        my_iteration_units = len(erlang_list) * max_iters
        self.properties['my_iteration_units'] = my_iteration_units

        if 'progress_dict' not in self.properties:
            manager = Manager()
            self.properties['progress_dict'] = manager.dict()

        progress_dict = self.properties['progress_dict']
        done_units_so_far = 0

        for erlang_index, erlang in enumerate(erlang_list):
            first_erlang = (erlang_index == 0)
            
            done_units_so_far = self._run_generic_sim(
                erlang=erlang,
                first_erlang=first_erlang,
                erlang_index=erlang_index,
                progress_dict=progress_dict,
                done_offset=done_units_so_far
            )

    def run_sim(self, **kwargs):
        """
        Entry point for this process.
        """
        # Wrap in try/except to catch any silent crashes in the future
        try:
            self.properties = kwargs['thread_params']
            
            # --- FIX START ---
            # Set the timestamp
            self.properties['sim_start'] = kwargs['sim_start']
            
            # DELETE THIS LINE FROM YOUR OLD CODE:
            # self.properties['date'] = kwargs['sim_start'].split('_')[0] 
            # Reason: This was overwriting the date '0108' with the hour '00'.
            # The correct 'date' is already in self.properties from __main__.
            # --- FIX END ---
            
            self.properties['thread_num'] = kwargs['thread_num']
            
            # Store the original ID (e.g., 's1')
            if '_E' in kwargs['thread_num']:
                self.properties['sim_id'] = kwargs['thread_num'].split('_E')[0]
            else:
                self.properties['sim_id'] = kwargs['thread_num']

            # Print where we are saving to ensure it's correct this time
            topo = self.properties.get('topology', 'Unknown')
            date = self.properties.get('date', 'Unknown')
            start = self.properties.get('sim_start', 'Unknown')
            print(f"Process {self.properties['sim_id']} target output: output/{topo}/{date}/{start}")

            self.run_generic_sim()

        except Exception as e:
            print(f"!!! CRITICAL FAILURE IN PROCESS {kwargs.get('thread_num')} !!!")
            import traceback
            traceback.print_exc()


def run(sims_dict: dict, stop_flag: multiprocessing.Event, sim_start_str: str):
    """
    Spawns processes.
    """
    if not sims_dict:
        print("No simulations to run.")
        return

    any_conf = list(sims_dict.values())[0]
    log_queue = any_conf.get('log_queue')
    progress_queue = any_conf.get('progress_queue')

    def log(message):
        if log_queue:
            log_queue.put(message)
        else:
            print(message)

    processes = []
    
    print(f"--- Launching {len(sims_dict)} Processes (Timestamp: {sim_start_str}) ---")

    for thread_num, thread_params in sims_dict.items():
        thread_params['progress_queue'] = progress_queue
        thread_params['stop_flag'] = stop_flag

        p = Process(
            target=NetworkSimulator().run_sim,
            kwargs={
                'thread_num': thread_num,
                'thread_params': thread_params,
                'sim_start': sim_start_str
            }
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    print("All simulations completed.")


if __name__ == '__main__':
    args_dict = parse_args()
    initial_sims = read_config(args_dict=args_dict, config_path=args_dict['config_path'])
    
    # --- STEP 0: GENERATE TIMESTAMP ---
    now = datetime.now()
    sim_start_str = now.strftime("%H_%M_%S_%f") 
    date_str = now.strftime("%m%d")

    # --- STEP A: LOAD DATA & UPDATE CONFIG ---
    print("Loading network data and generating input files...")
    
    # IMPORTANT: We iterate over keys to update 'initial_sims' IN PLACE
    for key in list(initial_sims.keys()):
        config = initial_sims[key]
        
        # Inject timestamp info
        config['date'] = date_str
        config['sim_start'] = sim_start_str
        config['thread_num'] = key  # Use 's1' so bw_info_s1.json is created

        # Run create_input. This calculates paths and loads network data.
        updated_data = create_input(base_fp='data', engine_props=config)
        
        # --- FIX: MERGE UPDATED DATA BACK INTO MAIN CONFIG ---
        # This ensures that when we copy this config for workers later,
        # it has the file paths!
        initial_sims[key].update(updated_data)
        
        # Save the single sim_input_s1.json
        save_input(
            base_fp='data',
            properties=initial_sims[key],
            file_name=f"sim_input_{key}.json",
            data_dict=updated_data
        )
    print("Data load complete. Paths propagated to workers.")

    # --- STEP B: EXPLODE FOR MULTIPROCESSING ---
    final_sims = {}
    for sim_key, config in initial_sims.items():
        thread_erlangs = config.get('thread_erlangs', True)
        start = int(config['erlang_start'])
        stop = int(config['erlang_stop'])
        step = int(config['erlang_step'])
        
        if thread_erlangs and (stop > start):
            erlang_range = range(start, stop, step)
            print(f"['{sim_key}'] Splitting into {len(erlang_range)} independent processes.")
            
            for i, erlang_val in enumerate(erlang_range):
                # Deepcopy now includes the paths from Step A!
                new_conf = copy.deepcopy(config)
                
                new_conf['erlang_start'] = erlang_val
                new_conf['erlang_stop'] = erlang_val + step 
                
                new_key = f"{sim_key}_E{erlang_val}"
                final_sims[new_key] = new_conf
        else:
            final_sims[sim_key] = config

    stop_flag = multiprocessing.Event()
    run(sims_dict=final_sims, stop_flag=stop_flag, sim_start_str=sim_start_str)