# main.py
import sys
import multiprocessing
from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from PyQt5 import QtWidgets
from gui_scripts.main_window import MainWindow

if __name__ == '__main__':
    # Parse command-line arguments and configuration.
    args_dict = parse_args()
    all_sims_dict = read_config(args_dict=args_dict, config_path=args_dict['config_path'])
    print("all_sims_dict loaded:", all_sims_dict)  # Debug: Verify it's not None

    # Determine total number of erlang simulations from configuration.
    # (Assumes configuration like {'erlangs': {'start': 50, 'stop': 100, 'step': 50}})
    first_key = list(all_sims_dict.keys())[0]
    erlang_conf = all_sims_dict[first_key]['erlangs']
    total_erlangs = len(range(erlang_conf['start'], erlang_conf['stop'], erlang_conf['step']))

    # Create a single Manager and shared progress dictionary.
    manager = multiprocessing.Manager()
    shared_progress_dict = manager.dict()
    for i in range(total_erlangs):
        shared_progress_dict[i] = 0

    # Launch the GUI. (Simulation will only start when the user clicks “Run”.)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    # Inject simulation configuration and shared progress dictionary into the GUI.
    window.set_simulation_config(all_sims_dict)
    window.set_shared_progress_dict(shared_progress_dict)
    window.show()
    sys.exit(app.exec_())
