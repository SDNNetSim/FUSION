# fusion/cli/config_setup.py

import os
import re
from configparser import ConfigParser
from pathlib import Path

from fusion.helper_scripts.os_helpers import create_dir
from fusion.cli.args.run_sim_args import SIM_REQUIRED_OPTIONS, OTHER_OPTIONS


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'ini', 'run_ini', 'config.ini')

def normalize_config_path(config_path: str) -> str:
    config_path = os.path.expanduser(config_path)
    if not os.path.isabs(config_path):
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / config_path

    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Could not find config file at: {config_path}")

    return config_path


def setup_config_from_cli(args) -> dict:
    args_dict = vars(args)
    config_path = args_dict.get('config_path')

    try:
        config_data = load_config(config_path, args_dict)
        return config_data
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return {}

def load_config(config_path: str, args_dict: dict = None) -> dict:
    """
    Load and parse the .ini config file and return structured simulation parameters.

    :param config_path: Path to the INI config file.
    :param args_dict: Command-line overrides, if any.
    :return: Dictionary with keys like "s1", "s2", each containing simulation config.
    """
    if args_dict is None:
        args_dict = {}

    config_dict = {'s1': dict()}
    config = ConfigParser()

    try:
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

        # Handle additional threads/simulations like [s2], [s3], etc.
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

    # TODO: Change to logging file instead of print
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
    dictionary[dest_key] = {k: v for k, v in dictionary['s1'].items()}
    return dictionary


def _find_category(category_dict: dict, target_key: str):
    for category, subdict in category_dict.items():
        if target_key in subdict:
            return category
    return None


if __name__ == '__main__':
    dummy_args = {'run_id': 'debug_test'}
    print(load_config('ini/run_ini/config.ini', dummy_args))
