import math
from statistics import mean, variance, stdev

import numpy as np

from fusion.core.properties import StatsProps
from fusion.core.properties import SNAP_KEYS_LIST
from fusion.sim.utils import find_path_len
from fusion.analysis.network_analysis import NetworkAnalyzer
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


class SimStats:
    """
    The SimStats class finds and stores all relevant statistics in simulations.
    """

    def __init__(self, engine_props: dict, sim_info: str, stats_props: dict = None):
        if stats_props is not None:
            self.stats_props = stats_props
        else:
            self.stats_props = StatsProps()

        self.engine_props = engine_props
        self.sim_info = sim_info

        # Used to track transponders for a single allocated request
        self.current_transponders = 0
        # Used to track transponders for an entire simulation on average
        self.total_transponders = 0
        self.blocked_requests = 0
        self.block_mean = None
        self.block_variance = None
        self.block_ci = None
        self.block_ci_percent = None
        self.bit_rate_request = None
        self.bit_rate_blocked = 0
        self.bit_rate_block_mean = None
        self.bit_rate_block_variance = None
        self.bit_rate_block_ci = None
        self.bit_rate_block_ci_percent = None
        self.topology = None
        self.iteration = None

    @staticmethod
    def _get_snapshot_info(network_spectrum_dict: dict, path_list: list):
        """
        Retrieves relative information for simulation snapshots.

        :param network_spectrum_dict: The current network spectrum database.
        :param path_list: A path to find snapshot info, if empty, does this for the entire network.
        :return: The occupied slots, number of guard bands, and active requests.
        :rtype: tuple
        """
        active_reqs_set = set()
        occupied_slots = 0
        guard_slots = 0
        # Skip by two because the link is bidirectional, no need to check both arrays e.g., (0, 1) and (1, 0)
        for link in list(network_spectrum_dict.keys())[::2]:
            if path_list is not None and link not in path_list:
                continue
            link_data = network_spectrum_dict[link]
            for core in link_data['cores_matrix']:
                requests_set = set(core[core > 0])
                for curr_req in requests_set:
                    active_reqs_set.add(curr_req)
                occupied_slots += len(np.where(core != 0)[0])
                guard_slots += len(np.where(core < 0)[0])

        return occupied_slots, guard_slots, len(active_reqs_set)

    def update_snapshot(self, network_spectrum_dict: dict, req_num: int, path_list: list = None):
        """
        Finds the total number of occupied slots and guard bands currently allocated in the network or a specific path.

        :param network_spectrum_dict: The current network spectrum database.
        :param req_num: The current request number.
        :param path_list: The desired path to find the occupied slots on.
        :return: None
        """
        occupied_slots, guard_slots, active_reqs = self._get_snapshot_info(network_spectrum_dict=network_spectrum_dict,
                                                                           path_list=path_list)
        link_usage = NetworkAnalyzer.get_link_usage_summary(network_spectrum_dict)
        blocking_prob = self.blocked_requests / req_num
        bit_rate_block_prob = self.bit_rate_blocked / self.bit_rate_request

        self.stats_props.snapshots_dict[req_num]['occupied_slots'].append(occupied_slots)
        self.stats_props.snapshots_dict[req_num]['guard_slots'].append(guard_slots)
        self.stats_props.snapshots_dict[req_num]['active_requests'].append(active_reqs)
        self.stats_props.snapshots_dict[req_num]["blocking_prob"].append(blocking_prob)
        self.stats_props.snapshots_dict[req_num]['num_segments'].append(self.current_transponders)
        self.stats_props.snapshots_dict[req_num]["bit_rate_blocking_prob"].append(bit_rate_block_prob)
        self.stats_props.snapshots_dict[req_num]['link_usage'] = link_usage

    def _init_snapshots(self):
        for req_num in range(0, self.engine_props['num_requests'] + 1, self.engine_props['snapshot_step']):
            self.stats_props.snapshots_dict[req_num] = dict()
            for key in SNAP_KEYS_LIST:
                self.stats_props.snapshots_dict[req_num][key] = list()

    def _init_mods_weights_bws(self):
        for bandwidth, obj in self.engine_props['mod_per_bw'].items():
            self.stats_props.modulations_used_dict[bandwidth] = dict()
            self.stats_props.weights_dict[bandwidth] = dict()
            for modulation in obj.keys():
                self.stats_props.weights_dict[bandwidth][modulation] = list()
                self.stats_props.modulations_used_dict[bandwidth][modulation] = 0
                if modulation not in self.stats_props.modulations_used_dict or isinstance(
                        self.stats_props.modulations_used_dict[modulation]['length']['overall'], dict):
                    self.stats_props.modulations_used_dict[modulation] = dict()
                    self.stats_props.modulations_used_dict[modulation]['length'] = dict()
                    self.stats_props.modulations_used_dict[modulation]['length']['overall'] = list()
                    for band in self.engine_props['band_list']:
                        self.stats_props.modulations_used_dict[modulation][band] = 0
                        self.stats_props.modulations_used_dict[modulation]['length'][band] = list()

            self.stats_props.bandwidth_blocking_dict[bandwidth] = 0

    def _init_stat_dicts(self):
        for stat_key, data_type in vars(self.stats_props).items():
            if not isinstance(data_type, dict):
                continue
            if stat_key in ('modulations_used_dict', 'weights_dict', 'bandwidth_blocking_dict'):
                self._init_mods_weights_bws()
            elif stat_key == 'snapshots_dict':
                if self.engine_props['save_snapshots']:
                    self._init_snapshots()
            elif stat_key == 'cores_dict':
                self.stats_props.cores_dict = {core: 0 for core in range(self.engine_props['cores_per_link'])}
            elif stat_key == 'block_reasons_dict':
                self.stats_props.block_reasons_dict = {'distance': 0, 'congestion': 0, 'xt_threshold': 0}
            elif stat_key == 'link_usage_dict':
                self.stats_props.link_usage_dict = dict()
            elif stat_key != 'iter_stats':
                raise ValueError('Dictionary statistic was not reset in props.')

    def _init_stat_lists(self):
        for stat_key in vars(self.stats_props).keys():
            data_type = getattr(self.stats_props, stat_key)
            if isinstance(data_type, list):
                # Only reset sim_block_list when we encounter a new traffic volume
                if self.iteration != 0 and stat_key in ['simulation_blocking_list', 'simulation_bitrate_blocking_list']:
                    continue
                if stat_key == 'path_index_list':
                    continue
                setattr(self.stats_props, stat_key, list())

    def init_iter_stats(self):
        """
        Initializes data structures used in other methods of this class.

        :return: None
        """
        self._init_stat_dicts()
        self._init_stat_lists()

        self.blocked_requests = 0
        self.bit_rate_blocked = 0
        self.bit_rate_request = 0
        self.total_transponders = 0

        k_paths = self.engine_props.get('k_paths')
        self.stats_props.path_index_list = [0] * k_paths

    def calculate_blocking_statistics(self):
        """
        Gets the current blocking probability.

        :return: None
        """
        if self.engine_props['num_requests'] == 0:
            blocking_prob = 0
            bit_rate_blocking_prob = 0
        else:
            blocking_prob = self.blocked_requests / self.engine_props['num_requests']
            bit_rate_blocking_prob = (self.bit_rate_blocked / self.bit_rate_request
                                      if self.bit_rate_request > 0 else 0)

        self.stats_props.simulation_blocking_list.append(blocking_prob)
        self.stats_props.simulation_bitrate_blocking_list.append(bit_rate_blocking_prob)

    def _handle_iter_lists(self, sdn_data: object):
        for stat_key in sdn_data.stat_key_list:
            curr_sdn_data = sdn_data.get_data(key=stat_key)
            if stat_key == 'crosstalk_list':
                # (drl_path_agents) fixme
                if curr_sdn_data == [None]:
                    break
            for i, data in enumerate(curr_sdn_data):
                if stat_key == 'core_list':
                    self.stats_props.cores_dict[data] += 1
                elif stat_key == 'modulation_list':
                    bandwidth = sdn_data.bandwidth_list[i]
                    band = sdn_data.band_list[i]
                    self.stats_props.modulations_used_dict[bandwidth][data] += 1
                    self.stats_props.modulations_used_dict[data][band] += 1
                    self.stats_props.modulations_used_dict[data]['length'][band].append(sdn_data.path_weight)
                    self.stats_props.modulations_used_dict[data]['length']['overall'].append(sdn_data.path_weight)
                elif stat_key == 'start_slot_list':
                    self.stats_props.start_slot_list.append(int(data))
                elif stat_key == 'end_slot_list':
                    self.stats_props.end_slot_list.append(int(data))
                elif stat_key == 'modulation_list':
                    self.stats_props.modulation_list.append(int(data))
                elif stat_key == 'bandwidth_list':
                    self.stats_props.bandwidth_list.append(int(data))

    def iter_update(self, req_data: dict, sdn_data: object, network_spectrum_dict: dict):
        """
        Continuously updates the statistical data for each request allocated/blocked in the current iteration.

        :param req_data: Holds data relevant to the current request.
        :param sdn_data: Hold the response data from the sdn controller.
        :return: None
        """
        # Request was blocked
        if not sdn_data.was_routed:
            self.blocked_requests += 1
            self.bit_rate_blocked += int(sdn_data.bandwidth)
            self.bit_rate_request += int(sdn_data.bandwidth)
            self.stats_props.block_reasons_dict[sdn_data.block_reason] += 1
            self.stats_props.bandwidth_blocking_dict[req_data['bandwidth']] += 1
        else:
            num_hops = len(sdn_data.path_list) - 1
            self.stats_props.hops_list.append(num_hops)

            path_len = find_path_len(path_list=sdn_data.path_list, topology=self.topology)
            self.stats_props.lengths_list.append(round(float(path_len), 2))

            self._handle_iter_lists(sdn_data=sdn_data)
            self.stats_props.route_times_list.append(sdn_data.route_time)
            self.total_transponders += sdn_data.number_of_transponders
            bandwidth = sdn_data.bandwidth
            mod_format = sdn_data.modulation_list[0]
            self.stats_props.path_index_list[sdn_data.path_index] += 1

            self.bit_rate_request += int(sdn_data.bandwidth)
            self.stats_props.weights_dict[bandwidth][mod_format].append(round(float(sdn_data.path_weight), 2))
            self.stats_props.link_usage_dict = NetworkAnalyzer.get_link_usage_summary(network_spectrum_dict)

    def _get_iter_means(self):
        for _, curr_snapshot in self.stats_props.snapshots_dict.items():
            for snap_key, data_list in curr_snapshot.items():
                if data_list:
                    curr_snapshot[snap_key] = mean(data_list)
                else:
                    curr_snapshot[snap_key] = None

        for _, mod_obj in self.stats_props.weights_dict.items():
            for modulation, data_list in mod_obj.items():
                # Skip if already processed (data_list is already a dict with statistics)
                if isinstance(data_list, dict):
                    continue

                # Modulation was never used
                if len(data_list) == 0:
                    mod_obj[modulation] = {'mean': None, 'std': None, 'min': None, 'max': None}
                elif len(data_list) == 1:
                    mod_obj[modulation] = {'mean': mean(data_list), 'std': 0.0,
                                           'min': min(data_list), 'max': max(data_list)}
                else:
                    mod_obj[modulation] = {'mean': mean(data_list), 'std': stdev(data_list),
                                           'min': min(data_list), 'max': max(data_list)}
                for key, value in self.stats_props.modulations_used_dict[modulation]['length'].items():
                    if not isinstance(value, list):
                        continue
                    if len(value) == 0:
                        self.stats_props.modulations_used_dict[modulation]['length'][key] = {'mean': None, 'std': None,
                                                                                      'min': None, 'max': None}
                    else:
                        if len(value) == 1:
                            deviation = 0.0
                        else:
                            deviation = stdev(value)
                        self.stats_props.modulations_used_dict[modulation]['length'][key] = {
                            'mean': round(float(mean(value)), 2), 'std': round(float(deviation), 2),
                            'min': round(float(min(value)), 2), 'max': round(float(max(value)), 2)}

    def finalize_iteration_statistics(self):
        """
        Updates relevant stats after an iteration has finished.

        :return: None
        """
        if self.engine_props['num_requests'] == self.blocked_requests:
            self.stats_props.transponders_list.append(0)
        else:
            trans_mean = self.total_transponders / float(self.engine_props['num_requests'] - self.blocked_requests)
            self.stats_props.transponders_list.append(trans_mean)

        if self.blocked_requests > 0:
            # Check if already normalized (values are between 0 and 1)
            current_values = list(self.stats_props.block_reasons_dict.values())
            is_already_normalized = all(isinstance(v, float) and 0 <= v <= 1 for v in current_values if v > 0)

            if not is_already_normalized:
                for block_type, num_times in self.stats_props.block_reasons_dict.items():
                    self.stats_props.block_reasons_dict[block_type] = num_times / float(self.blocked_requests)

        self._get_iter_means()

    def calculate_confidence_interval(self):
        """
        Get the confidence interval for every iteration so far.

        :return: Whether the simulations should end for this erlang.
        :rtype: bool
        """
        self.block_mean = mean(self.stats_props.simulation_blocking_list)
        self.bit_rate_block_mean = mean(self.stats_props.simulation_bitrate_blocking_list)
        if len(self.stats_props.simulation_blocking_list) <= 1:
            return False

        self.block_variance = variance(self.stats_props.simulation_blocking_list)
        self.bit_rate_block_variance = variance(self.stats_props.simulation_bitrate_blocking_list)

        if self.block_mean == 0.0:
            return False

        try:
            # Using 1.96 for 95% confidence level (1.645 for 90%)
            block_ci_rate = 1.96 * (math.sqrt(self.block_variance) / math.sqrt(len(self.stats_props.simulation_blocking_list)))
            self.block_ci = block_ci_rate
            block_ci_percent = ((2 * block_ci_rate) / self.block_mean) * 100
            self.block_ci_percent = block_ci_percent
            # Bit rate blocking confidence interval calculation
            bit_rate_block_ci = 1.96 * (
                    math.sqrt(self.bit_rate_block_variance) / math.sqrt(len(self.stats_props.simulation_bitrate_blocking_list)))
            self.bit_rate_block_ci = bit_rate_block_ci
            bit_rate_block_ci_percent = ((2 * bit_rate_block_ci) / self.bit_rate_block_mean) * 100
            self.bit_rate_block_ci_percent = bit_rate_block_ci_percent
        except ZeroDivisionError:
            return False

        # CI percent threshold should be configurable (tracked in core/TODO.md)
        if block_ci_percent <= 5:
            logger.info("Confidence interval of %.2f%% reached. %d, ending for Erlang: %s",
                        block_ci_percent, self.iteration + 1, self.engine_props['erlang'])
            return True

        return False

    def get_blocking_statistics(self):
        """
        Get all blocking-related statistics for persistence.

        :return: Dictionary containing blocking statistics
        :rtype: dict
        """
        return {
            'block_mean': self.block_mean,
            'block_variance': self.block_variance,
            'block_ci': self.block_ci,
            'block_ci_percent': self.block_ci_percent,
            'bit_rate_block_mean': self.bit_rate_block_mean,
            'bit_rate_block_variance': self.bit_rate_block_variance,
            'bit_rate_block_ci': self.bit_rate_block_ci,
            'bit_rate_block_ci_percent': self.bit_rate_block_ci_percent,
            'iteration': self.iteration
        }

    # Backward compatibility methods
    def end_iter_update(self):
        """
        Backward compatibility wrapper for finalize_iteration_statistics.
        """
        return self.finalize_iteration_statistics()

    def save_stats(self, base_fp: str = 'data'):
        """
        Backward compatibility method for saving statistics.
        
        :param base_fp: Base file path for saving
        """
        # Import here to avoid circular imports
        from fusion.core.persistence import StatsPersistence  # pylint: disable=import-outside-toplevel

        # Ensure iteration is set to 0 if not initialized (for backward compatibility)
        if self.iteration is None:
            self.iteration = 0

        persistence = StatsPersistence(
            engine_props=self.engine_props,
            sim_info=self.sim_info
        )

        # Prepare save dict with iter_stats structure
        save_dict = {'iter_stats': {}}

        # Get blocking statistics
        blocking_stats = self.get_blocking_statistics()

        # Save using the persistence module
        persistence.save_stats(
            stats_dict=save_dict,
            stats_props=self.stats_props,
            blocking_stats=blocking_stats,
            base_file_path=base_fp
        )
