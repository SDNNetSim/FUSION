from run_sim import run
from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
import json
import sys
import os
import configparser

args_obj = parse_args()

def run_comparison():
    all_sims_dict = read_config(args_obj=args_obj, config_path=args_obj['config_path'])

    # Call the run function from run_sim.py
    run(sims_dict=all_sims_dict)

def find_type_and_saved_path(args_obj=args_obj, config_path=args_obj['config_path']):
    config = configparser.ConfigParser()

    #Following two blocks are from setup_config to reach the type of patb
    if config_path is None:
        config_path = os.path.join('ini', 'run_ini', 'config.ini')
    config.read(config_path)

    if not config.has_option('general_settings', 'sim_type'):
        config_path = os.path.join('ini', 'run_ini')
        create_dir(config_path)
        raise ValueError("Missing 'general_settings' section in the configuration file. "
                         "Please ensure you have a file called config.ini in the run_ini directory.")

    if config['general_settings']['sim_type'] == 'yue':
        return './data/run_comparison_data/yue_run_data.json'
    else:
        raise ValueError("Error: sim_type not supported by function.")

def compare_json_files(old_file, new_file):
    """Load and compare two JSON files."""
    with open(old_file, 'r') as f:
        old_data = json.load(f)

    with open(new_file, 'r') as g:
        new_data = json.load(g)

    if old_data == new_data:
        print("The comparison results pass.")
    else:
        print("The comparison results do not pass.")
        sys.exit(1)


def find_newest_file(directory):
    newest_file = None
    latest_time = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        mod_time = os.path.getmtime(file_path)
        if mod_time > latest_time:
            latest_time = mod_time
            newest_file = filename

    newest_file = directory + '/' + newest_file
    return newest_file


if __name__ == "__main__":
    run_comparison()
    old_saved_data_path = find_type_and_saved_path()
    path_to_output = './data/output/NSFNet'
    date_of_simulation = find_newest_file(path_to_output)
    time_of_data_path = find_newest_file(date_of_simulation)
    add_simulation_run_to_path = time_of_data_path + '/s1'
    new_saved_data_path = find_newest_file(add_simulation_run_to_path)
    compare_json_files(old_saved_data_path, new_saved_data_path)