import os
import configparser
import re

from helper_scripts.os_helpers import create_dir
from arg_scripts.config_args import SIM_REQUIRED_OPTIONS, OTHER_OPTIONS


def _copy_dict_vals(dest_key: str, dictionary: dict):
    """
    Given the s1 simulation dictionary, copy the values to another simulation run.

    :param dest_key: The destination key where the values will be copied to.
    :param dictionary: The original s1 dictionary.
    :return: All values of s1 copied to the given simulation key.
    :rtype: dict
    """
    dictionary[dest_key] = dict()
    for key, val in dictionary['s1'].items():
        dictionary[dest_key][key] = val

    return dictionary


def _find_category(category_dict: dict, target_key: str):
    for category, subdict in category_dict.items():
        for sub_key in subdict:
            if sub_key == target_key:
                return category

    return None


def _setup_threads(config: configparser.ConfigParser, config_dict: dict, section_list: list, types_dict: dict,
                   other_dict: dict, args_dict: dict):
    """
    Checks if multiple threads/simulations should be run, structures each simulation's parameters.

    :param config: The configuration object.
    :param config_dict: The main simulations configuration params.
    :param section_list: Every section in the ini file.
    :param types_dict: Contains option conversion types.
    :param other_dict: Contains non-required options.
    :param args_dict: Arguments passed via the command line (if any).
    :return: Every simulation's structured parameters.
    :rtype: dict
    """
    for new_thread in section_list:
        if not re.match(r'^s\d', new_thread):
            continue

        config_dict = _copy_dict_vals(dest_key=new_thread, dictionary=config_dict)
        # Make desired changes for this thread
        for key, value in config.items(new_thread):
            category = _find_category(category_dict=types_dict, target_key=key)
            try:
                type_obj = types_dict[category][key]
            except KeyError:
                if category is None:
                    category = _find_category(category_dict=other_dict, target_key=key)
                type_obj = other_dict[category][key]
            config_dict[new_thread][key] = type_obj(value)
            # TODO: Only support for changing all s<values> as of now
            if args_dict[key] is not None:
                config_dict[new_thread][key] = args_dict[key]

    return config_dict


def read_config(args_dict: dict, config_path: str = None):
    """
    Structures necessary data from the configuration file in the run_ini directory.

    :param args_dict: Arguments passed via the command line (if any).
    :param config_path: The configuration file path.
    :type args_dict: dict
    """
    config_dict = {'s1': dict()}
    config = configparser.ConfigParser()

    try:  # pylint: disable=too-many-nested-blocks
        if config_path is None:
            config_path = os.path.join('ini', 'run_ini', 'config.ini')
        config.read(config_path)

        if not config.has_section('general_settings'):
            config_path = os.path.join('ini', 'run_ini')
            create_dir(config_path)
            raise ValueError("Missing 'general_settings' section in the configuration file. "
                             "Please ensure you have a file called config.ini in the run_ini directory.")

        required_dict = SIM_REQUIRED_OPTIONS
        other_dict = OTHER_OPTIONS

        for category, options_dict in required_dict.items():
            for option, type_obj in options_dict.items():
                if not config.has_option(category, option):
                    raise ValueError(f"Missing '{option}' in the {category} section.")

                try:
                    config_dict['s1'][option] = type_obj(config[category][option])
                except KeyError:
                    type_obj = other_dict[category][option]
                    config_dict['s1'][option] = type_obj(config[category][option])

                # TODO: Only support for changing all s<values> as of now
                # if cmdline argument was provided, prioritize that
                if args_dict[option] is not None:
                    config_dict['s1'][option] = args_dict[option]

        # Init other options to None if they haven't been specified
        for category, options_dict in other_dict.items():
            for option, type_obj in options_dict.items():
                if option not in config[category]:
                    config_dict['s1'][option] = None
                else:
                    if args_dict[option] is not None:
                        config_dict['s1'][option] = args_dict[option]
                    else:
                        try:
                            config_dict['s1'][option] = type_obj(config[category][option])
                        # The option was set to None, skip it
                        except ValueError:
                            continue

        # Ignoring index zero since we've already handled s1, the first simulation
        resp = _setup_threads(config=config, config_dict=config_dict, section_list=config.sections()[1:],
                              types_dict=required_dict, other_dict=other_dict, args_dict=args_dict)

        return resp

    except configparser.Error as error:
        print(f"Error reading configuration file: {error}")
        return None


if __name__ == '__main__':
    read_config(args_dict={'Test': None})
