# pylint: disable=broad-exception-caught

import os
import re
from configparser import ConfigParser
from pathlib import Path

from fusion.utils.os import create_dir

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'ini', 'run_ini', 'config.ini')


def str_to_bool(string: str) -> bool:
    """
    Convert string to boolean value.

    Args:
        string: Input string to convert

    Returns:
        Boolean value
    """
    return string.lower() in ['true', 'yes', '1']


# Configuration option mappings for INI file validation
SIM_REQUIRED_OPTIONS = {
    'general_settings': {
        'erlang_start': float,
        'erlang_stop': float,
        'erlang_step': float,
        'mod_assumption': str,
        'mod_assumption_path': str,
        'holding_time': float,
        'thread_erlangs': str_to_bool,
        'guard_slots': int,
        'num_requests': int,
        'max_iters': int,
        'dynamic_lps': str_to_bool,
        'route_method': str,
        'allocation_method': str,
        'save_snapshots': str_to_bool,
        'snapshot_step': int,
        'print_step': int,
        'spectrum_priority': str,
        'save_step': int,
        'save_start_end_slots': str_to_bool,
    },
    'topology_settings': {
        'network': str,
        'bw_per_slot': float,
        'cores_per_link': int,
        'const_link_weight': str_to_bool,
        'multi_fiber': str_to_bool,
    },
    'spectrum_settings': {
        'c_band': int,
    },
    'snr_settings': {
        'snr_type': str,
        'input_power': float,
        'egn_model': str_to_bool,
    },
    'file_settings': {
        'file_type': str,
    },
}

OTHER_OPTIONS = {
    'general_settings': {
        'k_paths': int,
        'filter_mods': bool,
    },
    'topology_settings': {
        'bi_directional': str_to_bool,
    },
    'spectrum_settings': {
        'o_band': int,
        'e_band': int,
        's_band': int,
        'l_band': int,
    },
    'file_settings': {
        'run_id': str,
    },
}


def normalize_config_path(config_path: str) -> str:
    """
    Normalize the config file path.
    """
    config_path = os.path.expanduser(config_path)
    if not os.path.isabs(config_path):
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / config_path

    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Could not find config file at: {config_path}")

    return config_path


def setup_config_from_cli(args) -> dict:
    """
    Setup the config from command line input.
    """
    args_dict = vars(args)
    config_path = args_dict.get('config_path')

    try:
        config_data = load_config(config_path, args_dict)
        return config_data
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return {}


# (version 6.0.0) TODO: Fix this pylint error in the future
def load_config(config_path: str, args_dict: dict = None) -> dict:
    """
    Loads an existing config from a config file.
    """
    if args_dict is None:
        args_dict = {}

    config_dict = {'s1': dict()}
    config = ConfigParser()

    try:  # pylint: disable=too-many-nested-blocks
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        else:
            config_path = normalize_config_path(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find config file at: {config_path}")

        config.read(config_path)

        if not config.has_section('general_settings'):
            config_path = os.path.join('ini', 'run_ini')
            create_dir(config_path)
            raise ValueError("Missing 'general_settings' section in config file. "
                             "Ensure config.ini exists in ini/run_ini/.")

        required_dict = SIM_REQUIRED_OPTIONS
        other_dict = OTHER_OPTIONS

        for category, options_dict in required_dict.items():
            for option, type_obj in options_dict.items():
                if not config.has_option(category, option):
                    raise ValueError(f"Missing required option '{option}' in section [{category}].")

                try:
                    config_dict['s1'][option] = type_obj(config[category][option])
                except KeyError:
                    type_obj = other_dict[category][option]
                    config_dict['s1'][option] = type_obj(config[category][option])

                if args_dict.get(option) is not None:
                    config_dict['s1'][option] = args_dict[option]

        for category, options_dict in other_dict.items():
            for option, type_obj in options_dict.items():
                if option not in config[category]:
                    config_dict['s1'][option] = None
                else:
                    if args_dict.get(option) is not None:
                        config_dict['s1'][option] = args_dict[option]
                    else:
                        try:
                            config_dict['s1'][option] = type_obj(config[category][option])
                        except ValueError:
                            continue

        other_sections = config.sections()[1:]  # skip general_settings
        config_dict = _setup_threads(
            config=config,
            config_dict=config_dict,
            section_list=other_sections,
            types_dict=required_dict,
            other_dict=other_dict,
            args_dict=args_dict
        )

        return config_dict or {}

    except Exception as error:
        print(f"[ERROR] Failed to load config: {error}")
        return {}


def _setup_threads(config: ConfigParser, config_dict: dict, section_list: list,
                   types_dict: dict, other_dict: dict, args_dict: dict) -> dict:
    for new_thread in section_list:
        if not re.match(r'^s\d', new_thread):
            continue

        config_dict = _copy_dict_vals(dest_key=new_thread, dictionary=config_dict)

        for key, value in config.items(new_thread):
            category = _find_category(types_dict, key) or _find_category(other_dict, key)
            if category is None:
                continue

            try:
                type_obj = types_dict.get(category, {}).get(key) or other_dict.get(category, {}).get(key)
                config_dict[new_thread][key] = type_obj(value)
            except Exception:
                continue

            if args_dict.get(key) is not None:
                config_dict[new_thread][key] = args_dict[key]

    return config_dict


def _copy_dict_vals(dest_key: str, dictionary: dict) -> dict:
    dictionary[dest_key] = {k: v for k, v in dictionary['s1'].items()}  # pylint: disable=unnecessary-comprehension
    return dictionary


def _find_category(category_dict: dict, target_key: str):
    for category, subdict in category_dict.items():
        if target_key in subdict:
            return category
    return None


def load_and_validate_config(args):
    """
    Load and validate configuration from CLI arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        dict: Validated configuration dictionary
    """
    config_dict = load_config(args.config_path, vars(args))
    return config_dict

class ConfigManager:
    """
    Wrapper for command line input and configuration file use.
    """

    def __init__(self, config_dict, args):
        self._config = config_dict
        self._args = args

    @classmethod
    def from_args(cls, args):
        """
        Loads arguments from command line input.
        """
        config_dict = load_config(args.config_path, vars(args))
        return cls(config_dict, args)

    def as_dict(self):
        """
        Gets config as dict.
        """
        return self._config

    def get(self, thread='s1'):
        """
        Returns a single config thread.
        """
        return self._config.get(thread, {})

    def get_args(self):
        """
        Gets args.
        """
        return self._args


# For manual debugging
if __name__ == '__main__':
    dummy_args = {'run_id': 'debug_test'}
    print(load_config('ini/run_ini/config.ini', dummy_args))
