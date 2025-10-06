"""
Statistics persistence module for FUSION.

Handles saving and loading simulation statistics to various formats.
"""

import copy
import json
import os
from datetime import datetime
from statistics import mean
from typing import Any

from fusion.utils.logging_config import get_logger
from fusion.utils.os import create_directory, find_project_root

PROJECT_ROOT = find_project_root()
logger = get_logger(__name__)


class StatsPersistence:
    """
    Handles persistence operations for simulation statistics.

    This class is responsible for saving simulation results to files
    in various formats (JSON, CSV, etc.).
    """

    def __init__(self, engine_props: dict[str, Any], sim_info: str):
        """
        Initialize the persistence handler.

        :param engine_props: Engine configuration properties
        :param sim_info: Simulation information string
        """
        self.engine_props = engine_props
        self.sim_info = sim_info

    def save_stats(
            self,
            stats_dict: dict[str, Any],
            stats_props: Any,
            blocking_stats: dict[str, float | None],
            base_file_path: str | None = None
    ) -> None:
        """
        Save simulation statistics to file.

        :param stats_dict: Dictionary to save
        :param stats_props: Statistics properties object
        :param blocking_stats: Blocking statistics (means, variances, CIs)
        :param base_file_path: Base path for output files
        """
        if self.engine_props['file_type'] not in ('json', 'csv'):
            raise NotImplementedError(
                f"Invalid file type: {self.engine_props['file_type']}, expected csv or "
                "json."
            )

        # Try to load existing data first to preserve accumulated iterations
        try:
            existing_file = self._get_save_path(base_file_path)
            if os.path.exists(existing_file):
                with open(existing_file, encoding='utf-8') as f:
                    save_dict = json.load(f)
            else:
                save_dict = {}
        except (OSError, json.JSONDecodeError):
            # If file doesn't exist or is corrupted, start fresh
            save_dict = {}

        # Update with current stats_dict data, but preserve iter_stats
        existing_iter_stats = save_dict.get('iter_stats', {})
        save_dict.update(copy.deepcopy(stats_dict))

        # Restore existing iter_stats if we had any
        if existing_iter_stats and 'iter_stats' in save_dict:
            save_dict['iter_stats'].update(existing_iter_stats)

        # Add link usage
        save_dict['link_usage'] = stats_props.link_usage_dict

        # Add blocking statistics
        save_dict['blocking_mean'] = blocking_stats.get('block_mean')
        save_dict['blocking_variance'] = blocking_stats.get('block_variance')
        save_dict['ci_rate_block'] = blocking_stats.get('block_ci')
        save_dict['ci_percent_block'] = blocking_stats.get('block_ci_percent')

        save_dict['bit_rate_blocking_mean'] = blocking_stats.get('bit_rate_block_mean')
        save_dict['bit_rate_blocking_variance'] = blocking_stats.get(
            'bit_rate_block_variance'
        )
        save_dict['ci_rate_bit_rate_block'] = blocking_stats.get('bit_rate_block_ci')
        save_dict['ci_percent_bit_rate_block'] = blocking_stats.get(
            'bit_rate_block_ci_percent'
        )

        # Prepare iteration statistics
        if 'iter_stats' not in save_dict:
            save_dict['iter_stats'] = {}

        # Save iteration statistics using actual iteration number as key
        # This preserves the original behavior of multiple iteration entries
        current_iteration = blocking_stats.get('iteration', 0)
        logger.info("Saving stats for iteration %s", current_iteration)
        iteration_data = self._prepare_iteration_stats(stats_props, current_iteration)

        # Always save iteration data, even if minimal - preserves original behavior
        save_dict['iter_stats'][current_iteration] = iteration_data

        # Save to file
        json_path = self._get_save_path(base_file_path)
        save_dir = os.path.dirname(json_path)

        logger.info("Saving to: %s", save_dir)
        create_directory(save_dir)

        sim_end_time = datetime.now().strftime("%m%d_%H_%M_%S_%f")
        save_dict['sim_end_time'] = sim_end_time

        if self.engine_props['file_type'] == 'json':
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(save_dict, file, indent=4)
        else:
            # CSV implementation would go here
            raise NotImplementedError("CSV output not yet implemented")

    def _prepare_iteration_stats(
        self, stats_props: Any, iteration: int | float | None = 0
    ) -> dict[str, Any]:  # pylint: disable=unused-argument
        """
        Prepare iteration statistics for saving.

        :param stats_props: Statistics properties object
        :param iteration: Current iteration number (reserved for future use)
        :return: Dictionary of prepared statistics
        """
        # Note: iteration parameter reserved for future functionality

        iter_stats: dict[str, Any] = {}

        # Process list statistics
        for stat_key in vars(stats_props).keys():
            if stat_key in (
                'transponders_list',
                'hops_list',
                'lengths_list',
                'route_times_list',
                'crosstalk_list',
            ):
                # Map new property names to old output key names for backward
                # compatibility
                if stat_key == 'transponders_list':
                    save_key = 'trans_'
                elif stat_key == 'crosstalk_list':
                    save_key = 'xt_'
                else:
                    save_key = f"{stat_key.split('_list')[0]}_"

                if stat_key == 'crosstalk_list':
                    stat_array = [
                        0 if stat is None else stat
                        for stat in getattr(stats_props, stat_key)
                    ]
                else:
                    stat_array = getattr(stats_props, stat_key)

                # Handle empty arrays
                if len(stat_array) == 0:
                    iter_stats[f'{save_key}mean'] = None
                    iter_stats[f'{save_key}min'] = None
                    iter_stats[f'{save_key}max'] = None
                else:
                    iter_stats[f'{save_key}mean'] = round(float(mean(stat_array)), 2)
                    iter_stats[f'{save_key}min'] = round(float(min(stat_array)), 2)
                    iter_stats[f'{save_key}max'] = round(float(max(stat_array)), 2)
            else:
                # Map new property names to old output keys for backward compatibility
                if stat_key == 'simulation_blocking_list':
                    output_key = 'sim_block_list'
                elif stat_key == 'simulation_bitrate_blocking_list':
                    output_key = 'sim_br_block_list'
                elif stat_key == 'modulations_used_dict':
                    output_key = 'mods_used_dict'
                elif stat_key == 'bandwidth_blocking_dict':
                    output_key = 'block_bw_dict'
                elif stat_key == 'number_of_transponders':
                    output_key = 'num_trans'
                elif stat_key == 'request_id':
                    output_key = 'req_id'
                else:
                    output_key = stat_key

                # Handle start/end slot lists based on configuration
                if (
                    stat_key in ['start_slot_list', 'end_slot_list']
                    and not self.engine_props.get('save_start_end_slots', False)
                ):
                    iter_stats[output_key] = []
                else:
                    iter_stats[output_key] = copy.deepcopy(
                        getattr(stats_props, stat_key)
                    )

        return iter_stats

    def _get_save_path(self, base_file_path: str | None = None) -> str:
        """
        Get the full path where the JSON file will be saved.

        :param base_file_path: Base path for output files
        :return: Full path to the JSON file
        """
        if base_file_path is None:
            base_file_path = 'data'

        save_path = os.path.join(
            PROJECT_ROOT,
            base_file_path,
            'output',
            self.sim_info,
            self.engine_props['thread_num']
        )

        return os.path.join(save_path, f"{self.engine_props['erlang']}_erlang.json")

    def load_stats(self, file_path: str) -> dict[str, Any]:
        """
        Load statistics from a saved file.

        :param file_path: Path to the statistics file
        :return: Loaded statistics dictionary
        """
        if file_path.endswith('.json'):
            with open(file_path, encoding='utf-8') as file:
                loaded_data = json.load(file)
                if not isinstance(loaded_data, dict):
                    data_type = type(loaded_data)
                    raise ValueError(
                        f"Expected dictionary in JSON file, got {data_type}"
                    )
                return loaded_data
        else:
            raise NotImplementedError(f"Loading from {file_path} not supported")
