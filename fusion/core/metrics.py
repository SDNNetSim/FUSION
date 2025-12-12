import math
from statistics import mean, stdev, variance
from typing import Any

import numpy as np

from fusion.analysis.network_analysis import NetworkAnalyzer
from fusion.core.properties import SNAP_KEYS_LIST, SDNProps, StatsProps
from fusion.utils.logging_config import get_logger
from fusion.utils.network import find_path_length

logger = get_logger(__name__)


class SimStats:
    """
    The SimStats class finds and stores all relevant statistics in simulations.
    """

    def __init__(
        self,
        engine_props: dict[str, Any],
        sim_info: str,
        stats_props: StatsProps | None = None,
    ):
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
        self.block_mean: float | None = None
        self.block_variance: float | None = None
        self.block_ci: float | None = None
        self.block_ci_percent: float | None = None
        self.bit_rate_request: int = 0
        self.bit_rate_blocked = 0
        self.bit_rate_block_mean: float | None = None
        self.bit_rate_block_variance: float | None = None
        self.bit_rate_block_ci: float | None = None
        self.bit_rate_block_ci_percent: float | None = None
        self.topology: Any = None
        self.iteration: int | None = None

        # Recovery time tracking (Phase 3 - Survivability)
        self.recovery_times_ms: list[float] = []
        self.failure_window_bp: list[float] = []
        self.recovery_events: list[dict[str, Any]] = []

        # Failure window parameters
        self.failure_window_size = engine_props.get("recovery_timing_settings", {}).get(
            "failure_window_size", 1000
        )

        # Fragmentation and decision time metrics
        self.fragmentation_scores: list[float] = []
        self.decision_times_ms: list[float] = []

        # P3.4: New path tracking for orchestrator mode
        self.total_requests: int = 0
        self.groomed_requests: int = 0
        self.sliced_requests: int = 0
        self.protected_requests: int = 0
        self.snapshot_interval: int = engine_props.get('snapshot_interval', 100)

        # Debug: Track mods_dict updates for v5/v6 comparison
        self.mods_dict_updates_log: list[dict[str, Any]] = []

    @staticmethod
    def _get_snapshot_info(
        network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
        path_list: list[tuple[int, int]] | None,
    ) -> tuple[int, int, int]:
        """
        Retrieves relative information for simulation snapshots.

        :param network_spectrum_dict: The current network spectrum database.
        :param path_list: A path to find snapshot info, if empty, does this for the
            entire network.
        :return: The occupied slots, number of guard bands, and active requests.
        :rtype: tuple
        """
        active_reqs_set = set()
        occupied_slots = 0
        guard_slots = 0
        # Skip by two because the link is bidirectional, no need to check both arrays
        # e.g., (0, 1) and (1, 0)
        for link in list(network_spectrum_dict.keys())[::2]:
            if path_list is not None and link not in path_list:
                continue
            link_data = network_spectrum_dict[link]
            for core in link_data["cores_matrix"]:
                requests_set = set(core[core > 0])
                for curr_req in requests_set:
                    active_reqs_set.add(curr_req)
                occupied_slots += len(np.where(core != 0)[0])
                guard_slots += len(np.where(core < 0)[0])

        return occupied_slots, guard_slots, len(active_reqs_set)

    def update_snapshot(
        self,
        network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
        request_number: int,
        path_list: list[tuple[int, int]] | None = None,
    ) -> None:
        """
        Finds the total number of occupied slots and guard bands currently allocated
        in the network or a specific path.

        :param network_spectrum_dict: The current network spectrum database.
        :param request_number: The current request number.
        :param path_list: The desired path to find the occupied slots on.
        :return: None
        """
        # occupied_slots, guard_slots, active_reqs = self._get_snapshot_info(
        #     network_spectrum_dict=network_spectrum_dict, path_list=path_list
        # )
        link_usage = NetworkAnalyzer.get_link_usage_summary(network_spectrum_dict)
        # blocking_prob = self.blocked_requests / request_number
        # bit_rate_block_prob = (
        #     self.bit_rate_blocked / self.bit_rate_request
        #     if self.bit_rate_request > 0
        #     else 0
        # )
        #
        # self.stats_props.snapshots_dict[request_number]["occupied_slots"].append(
        #     occupied_slots
        # )
        # self.stats_props.snapshots_dict[request_number]["guard_slots"].append(
        #     guard_slots
        # )
        # self.stats_props.snapshots_dict[request_number]["active_requests"].append(
        #     active_reqs
        # )
        # self.stats_props.snapshots_dict[request_number]["blocking_prob"].append(
        #     blocking_prob
        # )
        # self.stats_props.snapshots_dict[request_number]["num_segments"].append(
        #     self.current_transponders
        # )
        # self.stats_props.snapshots_dict[request_number][
        #     "bit_rate_blocking_prob"
        # ].append(bit_rate_block_prob)
        # self.stats_props.snapshots_dict[request_number]["link_usage"] = link_usage

    def _init_snapshots(self) -> None:
        for req_num in range(
            0, self.engine_props["num_requests"] + 1, self.engine_props["snapshot_step"]
        ):
            self.stats_props.snapshots_dict[req_num] = {}
            for key in SNAP_KEYS_LIST:
                self.stats_props.snapshots_dict[req_num][key] = []

    def _init_mods_weights_bws(self) -> None:
        # Initialize weights_dict and modulations_used_dict as nested dicts
        if not isinstance(self.stats_props.weights_dict, dict):
            self.stats_props.weights_dict = {}
        if not isinstance(self.stats_props.modulations_used_dict, dict):
            self.stats_props.modulations_used_dict = {}
        if not isinstance(self.stats_props.demand_realization_ratio, dict):
            self.stats_props.demand_realization_ratio = {}

        # Initialize demand_realization_ratio with overall key
        self.stats_props.demand_realization_ratio["overall"] = []

        for bandwidth, obj in self.engine_props["mod_per_bw"].items():
            # Convert bandwidth to string to match tracking logic
            bandwidth_key = str(bandwidth)
            # Ensure bandwidth keys exist as dicts
            if bandwidth_key not in self.stats_props.modulations_used_dict:
                self.stats_props.modulations_used_dict[bandwidth_key] = {}
            if bandwidth_key not in self.stats_props.weights_dict:
                self.stats_props.weights_dict[bandwidth_key] = {}

            # Initialize demand_realization_ratio per bandwidth
            self.stats_props.demand_realization_ratio[bandwidth_key] = []

            for modulation in obj.keys():
                # Initialize nested dict structure for weights
                if bandwidth_key not in self.stats_props.weights_dict:
                    self.stats_props.weights_dict[bandwidth_key] = {}
                self.stats_props.weights_dict[bandwidth_key][modulation] = []

                # Initialize nested dict structure for modulations_used
                if bandwidth_key not in self.stats_props.modulations_used_dict:
                    self.stats_props.modulations_used_dict[bandwidth_key] = {}
                self.stats_props.modulations_used_dict[bandwidth_key][modulation] = 0

                # Initialize modulation-specific tracking
                mod_dict = self.stats_props.modulations_used_dict
                mod_entry = mod_dict.get(modulation, {})
                length_entry = mod_entry.get("length", {})
                overall_entry = length_entry.get("overall")

                if (
                    modulation not in mod_dict
                    or not isinstance(mod_entry, dict)
                    or "length" not in mod_entry
                    or isinstance(overall_entry, dict)
                ):
                    self.stats_props.modulations_used_dict[modulation] = {}
                    self.stats_props.modulations_used_dict[modulation]["length"] = {}
                    self.stats_props.modulations_used_dict[modulation]["length"][
                        "overall"
                    ] = []
                    self.stats_props.modulations_used_dict[modulation]["hop"] = {}
                    self.stats_props.modulations_used_dict[modulation]["hop"]["overall"] = []
                    self.stats_props.modulations_used_dict[modulation]["snr"] = {}
                    self.stats_props.modulations_used_dict[modulation]["snr"]["overall"] = []
                    self.stats_props.modulations_used_dict[modulation]["xt_cost"] = {}
                    self.stats_props.modulations_used_dict[modulation]["xt_cost"]["overall"] = []
                    for band in self.engine_props["band_list"]:
                        self.stats_props.modulations_used_dict[modulation][band] = 0
                        self.stats_props.modulations_used_dict[modulation]["length"][
                            band
                        ] = []
                        self.stats_props.modulations_used_dict[modulation]["hop"][band] = []
                        self.stats_props.modulations_used_dict[modulation]["snr"][band] = []
                        self.stats_props.modulations_used_dict[modulation]["xt_cost"][band] = []

            self.stats_props.bandwidth_blocking_dict[bandwidth_key] = 0

    def _init_frag_dict(self) -> None:
        """Initialize fragmentation dictionary for tracking fragmentation metrics."""
        if not isinstance(self.stats_props.frag_dict, dict):
            self.stats_props.frag_dict = {}

        frag_metrics = self.engine_props.get("fragmentation_metrics", [])
        # Handle case where config parsing returns string "[]" instead of empty list
        if isinstance(frag_metrics, str) and frag_metrics in ("[]", ""):
            frag_metrics = []
        elif not isinstance(frag_metrics, list):
            frag_metrics = []

        for method in frag_metrics:
            self.stats_props.frag_dict[method] = {}
            for req_cnt in range(
                self.engine_props.get("frag_calc_step", 100),
                self.engine_props["num_requests"] + 1,
                self.engine_props.get("frag_calc_step", 100),
            ):
                self.stats_props.frag_dict[method][req_cnt] = {"arrival": {}, "release": {}}

    def _init_lp_bw_utilization_dict(self) -> None:
        """Initialize lightpath bandwidth utilization dictionary."""
        if not isinstance(self.stats_props.lp_bw_utilization_dict, dict):
            self.stats_props.lp_bw_utilization_dict = {}

        for bandwidth, obj in self.engine_props["mod_per_bw"].items():
            bandwidth_key = str(bandwidth)
            self.stats_props.lp_bw_utilization_dict[bandwidth_key] = {}
            for band in self.engine_props["band_list"]:
                self.stats_props.lp_bw_utilization_dict[bandwidth_key][band] = {}
                for core_num in range(self.engine_props["cores_per_link"]):
                    self.stats_props.lp_bw_utilization_dict[bandwidth_key][band][core_num] = []
        self.stats_props.lp_bw_utilization_dict["overall"] = []

    def _init_stat_dicts(self) -> None:
        for stat_key, data_type in vars(self.stats_props).items():
            if not isinstance(data_type, dict):
                continue
            if stat_key in (
                "modulations_used_dict",
                "weights_dict",
                "bandwidth_blocking_dict",
                "demand_realization_ratio",
            ):
                self._init_mods_weights_bws()
            elif stat_key == "frag_dict":
                self._init_frag_dict()
            elif stat_key == "lp_bw_utilization_dict":
                self._init_lp_bw_utilization_dict()
            elif stat_key == "snapshots_dict":
                if self.engine_props["save_snapshots"]:
                    self._init_snapshots()
            elif stat_key == "cores_dict":
                cores_range = range(self.engine_props["cores_per_link"])
                self.stats_props.cores_dict = dict.fromkeys(cores_range, 0)
            elif stat_key == "block_reasons_dict":
                self.stats_props.block_reasons_dict = {
                    "distance": 0,
                    "congestion": 0,
                    "xt_threshold": 0,
                }
            elif stat_key == "link_usage_dict":
                self.stats_props.link_usage_dict = {}
            elif stat_key != "iter_stats":
                raise ValueError("Dictionary statistic was not reset in props.")

    def _init_stat_lists(self) -> None:
        for stat_key in vars(self.stats_props).keys():
            data_type = getattr(self.stats_props, stat_key)
            if isinstance(data_type, list):
                # Only reset sim_block_list when we encounter a new traffic volume
                if self.iteration != 0 and stat_key in [
                    "simulation_blocking_list",
                    "simulation_bitrate_blocking_list",
                ]:
                    continue
                if stat_key == "path_index_list":
                    continue
                setattr(self.stats_props, stat_key, [])

    def init_iter_stats(self) -> None:
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

        k_paths = self.engine_props.get("k_paths", 1)
        if k_paths is not None:
            self.stats_props.path_index_list = [0] * k_paths
        else:
            self.stats_props.path_index_list = [0]

    def calculate_blocking_statistics(self) -> None:
        """
        Gets the current blocking probability.

        :return: None
        """
        if self.engine_props["num_requests"] == 0:
            blocking_prob = 0.0
            bit_rate_blocking_prob = 0.0
        else:
            num_requests = self.engine_props["num_requests"]
            blocking_prob = float(self.blocked_requests) / float(num_requests)
            if self.bit_rate_request > 0:
                bit_rate_blocking_prob = float(self.bit_rate_blocked) / float(
                    self.bit_rate_request
                )
            else:
                bit_rate_blocking_prob = 0.0

        self.stats_props.simulation_blocking_list.append(blocking_prob)
        self.stats_props.simulation_bitrate_blocking_list.append(bit_rate_blocking_prob)

    def _handle_iter_lists(self, sdn_data: SDNProps) -> None:
        for stat_key in sdn_data.stat_key_list:
            curr_sdn_data = sdn_data.get_data(key=stat_key)
            # Track overall SNR statistics (mean of newly established lightpaths)
            # Note: crosstalk is handled separately and only when xt_type is configured
            if stat_key == "snr_list":
                if set(curr_sdn_data) <= {None}:
                    continue  # Skip if all values are None
                # Filter to only newly established lightpaths (always filter like v5)
                new_lp_values = []
                for i, value in enumerate(curr_sdn_data):
                    if i < len(sdn_data.lightpath_id_list):
                        lp_id = sdn_data.lightpath_id_list[i]
                        if lp_id in sdn_data.was_new_lp_established and value is not None:
                            new_lp_values.append(value)
                # Append mean of new lightpaths to overall statistics list
                if len(new_lp_values) > 0:
                    self.stats_props.snr_list.append(mean(new_lp_values))
            if stat_key == "crosstalk_list":
                # (drl_path_agents) fixme
                if curr_sdn_data == [None]:
                    continue  # Skip this stat_key, don't break entire loop
            for i, data in enumerate(curr_sdn_data):
                # Only process NEW lightpaths when grooming is enabled (skip EXISTING groomed lightpaths)
                if self.engine_props.get("is_grooming_enabled", False):
                    if i < len(sdn_data.lightpath_id_list):
                        lp_id = sdn_data.lightpath_id_list[i]
                        if lp_id not in sdn_data.was_new_lp_established:
                            continue  # Skip EXISTING lightpaths

                if stat_key == "core_list":
                    if data not in self.stats_props.cores_dict:
                        self.stats_props.cores_dict[data] = 0
                    self.stats_props.cores_dict[data] += 1
                elif stat_key == "modulation_list":
                    bandwidth = sdn_data.lightpath_bandwidth_list[i]  # Use lightpath bandwidth
                    band = sdn_data.band_list[i]
                    # Ensure the nested dict structure exists - convert float to int to match dict keys
                    bandwidth_key = str(int(float(bandwidth))) if bandwidth is not None else None
                    mod_dict = self.stats_props.modulations_used_dict
                    bw_dict = mod_dict.get(bandwidth_key)
                    if bandwidth_key and isinstance(bw_dict, dict):
                        if data in bw_dict:
                            old_bw_count = bw_dict[data]
                            bw_dict[data] += 1
                            self.mods_dict_updates_log.append({
                                'req_id': sdn_data.request_id,
                                'mod': data,
                                'bw': bandwidth_key,
                                'band': None,
                                'action': 'bw_count_increment',
                                'old': old_bw_count,
                                'new': bw_dict[data]
                            })
                        else:
                            # Initialize if not present
                            bw_dict[data] = 1
                            self.mods_dict_updates_log.append({
                                'req_id': sdn_data.request_id,
                                'mod': data,
                                'bw': bandwidth_key,
                                'band': None,
                                'action': 'bw_count_init',
                                'value': 1
                            })
                    data_mod_dict = mod_dict.get(data)
                    if isinstance(data_mod_dict, dict):
                        if band in data_mod_dict:
                            old_band_count = data_mod_dict[band]
                            data_mod_dict[band] += 1
                            self.mods_dict_updates_log.append({
                                'req_id': sdn_data.request_id,
                                'mod': data,
                                'bw': None,
                                'band': band,
                                'action': 'band_count_increment',
                                'old': old_band_count,
                                'new': data_mod_dict[band]
                            })
                        else:
                            # Initialize if not present
                            data_mod_dict[band] = 1
                            self.mods_dict_updates_log.append({
                                'req_id': sdn_data.request_id,
                                'mod': data,
                                'bw': None,
                                'band': band,
                                'action': 'band_count_init',
                                'value': 1
                            })
                        length_dict = data_mod_dict.get("length")
                        has_length = "length" in data_mod_dict
                        if has_length and isinstance(length_dict, dict):
                            if band in length_dict:
                                length_dict[band].append(sdn_data.path_weight)
                                self.mods_dict_updates_log.append({
                                    'req_id': sdn_data.request_id,
                                    'mod': data,
                                    'bw': None,
                                    'band': band,
                                    'action': 'length_append',
                                    'value': sdn_data.path_weight
                                })
                            if "overall" in length_dict:
                                length_dict["overall"].append(sdn_data.path_weight)
                                self.mods_dict_updates_log.append({
                                    'req_id': sdn_data.request_id,
                                    'mod': data,
                                    'bw': None,
                                    'band': 'overall',
                                    'action': 'length_append',
                                    'value': sdn_data.path_weight
                                })

                        # Track hop count
                        hop_dict = data_mod_dict.get("hop")
                        if hop_dict and isinstance(hop_dict, dict):
                            num_hops = len(sdn_data.path_list) - 1
                            if band in hop_dict:
                                hop_dict[band].append(num_hops)
                                self.mods_dict_updates_log.append({
                                    'req_id': sdn_data.request_id,
                                    'mod': data,
                                    'bw': None,
                                    'band': band,
                                    'action': 'hop_append',
                                    'value': num_hops
                                })
                            if "overall" in hop_dict:
                                hop_dict["overall"].append(num_hops)
                                self.mods_dict_updates_log.append({
                                    'req_id': sdn_data.request_id,
                                    'mod': data,
                                    'bw': None,
                                    'band': 'overall',
                                    'action': 'hop_append',
                                    'value': num_hops
                                })

                        # Track SNR or XT cost
                        if self.engine_props.get("snr_type") != "None":
                            if i < len(sdn_data.snr_list) and sdn_data.snr_list[i] is not None:
                                if self.engine_props.get("snr_type") == "xt_calculation":
                                    # Track xt_cost
                                    xt_cost_dict = data_mod_dict.get("xt_cost")
                                    if xt_cost_dict and isinstance(xt_cost_dict, dict):
                                        if band in xt_cost_dict:
                                            xt_cost_dict[band].append(sdn_data.snr_list[i])
                                            self.mods_dict_updates_log.append({
                                                'req_id': sdn_data.request_id,
                                                'mod': data,
                                                'bw': None,
                                                'band': band,
                                                'action': 'xt_cost_append',
                                                'value': sdn_data.snr_list[i]
                                            })
                                        if "overall" in xt_cost_dict:
                                            xt_cost_dict["overall"].append(sdn_data.snr_list[i])
                                            self.mods_dict_updates_log.append({
                                                'req_id': sdn_data.request_id,
                                                'mod': data,
                                                'bw': None,
                                                'band': 'overall',
                                                'action': 'xt_cost_append',
                                                'value': sdn_data.snr_list[i]
                                            })
                                else:
                                    # Track snr
                                    snr_dict = data_mod_dict.get("snr")
                                    if snr_dict and isinstance(snr_dict, dict):
                                        if band in snr_dict:
                                            snr_dict[band].append(sdn_data.snr_list[i])
                                            self.mods_dict_updates_log.append({
                                                'req_id': sdn_data.request_id,
                                                'mod': data,
                                                'bw': None,
                                                'band': band,
                                                'action': 'snr_append',
                                                'value': sdn_data.snr_list[i]
                                            })
                                        if "overall" in snr_dict:
                                            snr_dict["overall"].append(sdn_data.snr_list[i])
                                            self.mods_dict_updates_log.append({
                                                'req_id': sdn_data.request_id,
                                                'mod': data,
                                                'bw': None,
                                                'band': 'overall',
                                                'action': 'snr_append',
                                                'value': sdn_data.snr_list[i]
                                            })
                elif stat_key == "start_slot_list":
                    self.stats_props.start_slot_list.append(int(data))
                elif stat_key == "end_slot_list":
                    self.stats_props.end_slot_list.append(int(data))
                elif stat_key == "modulation_list":
                    self.stats_props.modulation_list.append(str(data))
                elif stat_key == "bandwidth_list":
                    self.stats_props.bandwidth_list.append(float(data))

    def iter_update(
        self,
        req_data: dict[str, Any],
        sdn_data: SDNProps,
        network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    ) -> None:
        """
        Continuously updates the statistical data for each request allocated/blocked in
        the current iteration.

        :param req_data: Holds data relevant to the current request.
        :param sdn_data: Hold the response data from the sdn controller.
        :return: None
        """
        # Request was blocked
        if not sdn_data.was_routed:
            self.blocked_requests += 1
            if sdn_data.bandwidth is not None:
                self.bit_rate_blocked += int(sdn_data.bandwidth)
                self.bit_rate_request += int(sdn_data.bandwidth)
            if (
                sdn_data.block_reason is not None
                and sdn_data.block_reason in self.stats_props.block_reasons_dict
            ):
                block_reason = sdn_data.block_reason
                current_val = self.stats_props.block_reasons_dict[block_reason]
                if current_val is not None:
                    self.stats_props.block_reasons_dict[block_reason] = current_val + 1
                else:
                    self.stats_props.block_reasons_dict[block_reason] = 1
            req_bandwidth = req_data.get("bandwidth")
            bw_block_dict = self.stats_props.bandwidth_blocking_dict
            if req_bandwidth is not None and req_bandwidth in bw_block_dict:
                self.stats_props.bandwidth_blocking_dict[req_bandwidth] += 1
        else:
            # Skip stats tracking for fully groomed requests (v5 behavior)
            if sdn_data.was_groomed:
                if sdn_data.bandwidth is not None:
                    self.bit_rate_request += int(sdn_data.bandwidth)
                return

            # Track bit rate for requests
            if sdn_data.bandwidth is not None:
                self.bit_rate_request += int(sdn_data.bandwidth)

            # Track blocked bandwidth for partial allocations (slicing or grooming)
            # This applies regardless of grooming setting (matches v5 behavior)
            remaining_bw = getattr(sdn_data, "remaining_bw", None)
            if remaining_bw is not None and remaining_bw != "0":
                self.bit_rate_blocked += int(remaining_bw)
                was_new_lps = getattr(sdn_data, "was_new_lp_established", [])
                if not was_new_lps:
                    return

            if sdn_data.path_list is not None:
                num_hops = len(sdn_data.path_list) - 1
                self.stats_props.hops_list.append(float(num_hops))

                path_len = find_path_length(
                    path_list=sdn_data.path_list, topology=self.topology
                )
                if path_len is not None:
                    self.stats_props.lengths_list.append(round(float(path_len), 2))

            # Capture before state for request 40
            if sdn_data.request_id == 40:
                import copy
                before_mods_dict = {}
                for bw, bw_data in self.stats_props.modulations_used_dict.items():
                    if isinstance(bw_data, dict):
                        before_mods_dict[bw] = {mod: count for mod, count in bw_data.items() if isinstance(count, int)}

            self._handle_iter_lists(sdn_data=sdn_data)

            # Print modulation usage counts for all bandwidths
            mods_by_bw = {}
            for bw, bw_data in self.stats_props.modulations_used_dict.items():
                if isinstance(bw_data, dict):
                    # Extract only the modulation counts (skip nested dicts like 'length', 'hop', etc.)
                    mods_by_bw[bw] = {mod: count for mod, count in bw_data.items() if isinstance(count, int)}
            if sdn_data.route_time is not None:
                self.stats_props.route_times_list.append(sdn_data.route_time)
            if sdn_data.number_of_transponders is not None:
                self.total_transponders += sdn_data.number_of_transponders

            # Track path index from first modulation
            if sdn_data.modulation_list and len(sdn_data.modulation_list) > 0:
                if sdn_data.path_index is not None and 0 <= sdn_data.path_index < len(
                    self.stats_props.path_index_list
                ):
                    self.stats_props.path_index_list[sdn_data.path_index] += 1

            # Track weights for NEW lightpaths only when grooming is enabled
            if (
                hasattr(sdn_data, "bandwidth_list")
                and hasattr(sdn_data, "lightpath_id_list")
                and hasattr(sdn_data, "modulation_list")
                and sdn_data.path_weight is not None
            ):
                was_new_lps = getattr(sdn_data, "was_new_lp_established", [])
                for i in range(len(sdn_data.bandwidth_list)):
                    # Check if this lightpath should be tracked
                    should_track = False
                    if self.engine_props.get("is_grooming_enabled", False):
                        # When grooming is enabled, only track newly established lightpaths
                        if (
                            isinstance(was_new_lps, list)
                            and i < len(sdn_data.lightpath_id_list)
                            and sdn_data.lightpath_id_list[i] in was_new_lps
                        ):
                            should_track = True
                    else:
                        # When grooming is disabled, track all lightpaths
                        should_track = True

                    if should_track:
                        # FIX: Use lightpath_bandwidth (capacity) not bandwidth (dedicated) for weights_dict key
                        bandwidth_key = str(sdn_data.lightpath_bandwidth_list[i])
                        mod_format = sdn_data.modulation_list[i] if i < len(sdn_data.modulation_list) else None

                        if mod_format and bandwidth_key in self.stats_props.weights_dict:
                            if mod_format in self.stats_props.weights_dict[bandwidth_key]:
                                self.stats_props.weights_dict[bandwidth_key][mod_format].append(
                                    round(float(sdn_data.path_weight), 2)
                                )
                            else:
                                # Initialize if not present
                                self.stats_props.weights_dict[bandwidth_key][mod_format] = [
                                    round(float(sdn_data.path_weight), 2)
                                ]

            # Track demand realization ratio for partial grooming
            if self.engine_props.get("can_partially_serve"):
                bandwidth_key = str(sdn_data.bandwidth) if sdn_data.bandwidth is not None else None
                if bandwidth_key and bandwidth_key in self.stats_props.demand_realization_ratio:
                    remaining_bw = getattr(sdn_data, "remaining_bw", 0)
                    if remaining_bw is None:
                        remaining_bw = 0
                    original_bw = int(sdn_data.bandwidth)
                    served_bw = original_bw - int(remaining_bw)
                    realization_ratio = served_bw / original_bw
                    self.stats_props.demand_realization_ratio[bandwidth_key].append(realization_ratio)
                    self.stats_props.demand_realization_ratio["overall"].append(realization_ratio)
            self.stats_props.link_usage_dict = NetworkAnalyzer.get_link_usage_summary(
                network_spectrum_dict
            )

    def update_utilization_dict(self, utilization_dict: dict[int, dict[str, Any]]) -> None:
        """
        Update lightpath bandwidth utilization statistics.

        Called after lightpath release to track bandwidth utilization metrics
        per bandwidth/band/core combination.

        :param utilization_dict: Dictionary mapping lightpath_id to utilization info
            with keys: 'bit_rate', 'band', 'core', 'utilization'
        """
        erlang = self.engine_props.get('erlang', 0)
        iteration = self.iteration
        for lp_id in utilization_dict:
            lp_info = utilization_dict[lp_id]
            # Convert to int first to match initialization keys (e.g., '600' not '600.0')
            bit_rate_key = str(int(float(lp_info["bit_rate"])))
            band = lp_info["band"]
            core = lp_info["core"]
            utilization = lp_info["utilization"]

            # Track per-bandwidth/band/core (match grooming-new: direct append)
            # If keys don't exist, KeyError will surface the configuration issue
            self.stats_props.lp_bw_utilization_dict[bit_rate_key][band][core].append(
                utilization
            )

            # Track overall
            self.stats_props.lp_bw_utilization_dict["overall"].append(utilization)

    def _get_iter_means(self) -> None:
        for _, curr_snapshot in self.stats_props.snapshots_dict.items():
            for snap_key, data_list in curr_snapshot.items():
                # Skip if already a scalar (P3.4 new path stores scalars directly)
                if isinstance(data_list, (int, float)):
                    continue
                # Handle list case (legacy path)
                if data_list:
                    curr_snapshot[snap_key] = mean(data_list)
                else:
                    curr_snapshot[snap_key] = None

        # Process weights_dict if it's properly structured
        if isinstance(self.stats_props.weights_dict, dict):
            for _bandwidth, mod_obj in self.stats_props.weights_dict.items():
                if not isinstance(mod_obj, dict):
                    continue
                for modulation, data_list in mod_obj.items():
                    # Skip if already processed (data_list is already a dict with
                    # statistics)
                    if isinstance(data_list, dict):
                        continue

                    # Modulation was never used
                    if len(data_list) == 0:
                        mod_obj[modulation] = {
                            "mean": None,
                            "std": None,
                            "min": None,
                            "max": None,
                        }
                    elif len(data_list) == 1:
                        mod_obj[modulation] = {
                            "mean": mean(data_list),
                            "std": 0.0,
                            "min": min(data_list),
                            "max": max(data_list),
                        }
                    else:
                        mod_obj[modulation] = {
                            "mean": mean(data_list),
                            "std": stdev(data_list),
                            "min": min(data_list),
                            "max": max(data_list),
                        }
                    # Process modulations_used_dict if it has expected structure
                    mod_used_dict = self.stats_props.modulations_used_dict
                    mod_entry = mod_used_dict.get(modulation, {})
                    mod_length = mod_entry.get("length", {})
                    if (
                        isinstance(mod_used_dict, dict)
                        and modulation in mod_used_dict
                        and isinstance(mod_entry, dict)
                        and "length" in mod_entry
                        and isinstance(mod_length, dict)
                    ):
                        for key, value in mod_length.items():
                            if not isinstance(value, list):
                                continue
                            # Filter out None values before calculating statistics
                            filtered_value = [v for v in value if v is not None]
                            if len(filtered_value) == 0:
                                mod_length[key] = {
                                    "mean": None,
                                    "std": None,
                                    "min": None,
                                    "max": None,
                                }
                            else:
                                if len(filtered_value) == 1:
                                    deviation = 0.0
                                else:
                                    deviation = stdev(filtered_value)
                                self.stats_props.modulations_used_dict[modulation][
                                    "length"
                                ][key] = {
                                    "mean": round(float(mean(filtered_value)), 2),
                                    "std": round(float(deviation), 2),
                                    "min": round(float(min(filtered_value)), 2),
                                    "max": round(float(max(filtered_value)), 2),
                                }

                        # Process hop and snr (xt_cost remains as empty list like grooming-new)
                        for route_spec in ["hop", "snr"]:
                            # Skip SNR if snr_type is None
                            if self.engine_props.get("snr_type") == "None" and route_spec == "snr":
                                continue

                            route_dict = mod_entry.get(route_spec, {})
                            if isinstance(route_dict, dict):
                                for key, value in route_dict.items():
                                    if not isinstance(value, list):
                                        continue
                                    if len(value) == 0:
                                        route_dict[key] = {
                                            "mean": None,
                                            "std": None,
                                            "min": None,
                                            "max": None,
                                        }
                                    else:
                                        deviation = 0.0 if len(value) == 1 else stdev(value)
                                        self.stats_props.modulations_used_dict[modulation][
                                            route_spec
                                        ][key] = {
                                            "mean": round(float(mean(value)), 2),
                                            "std": round(float(deviation), 2),
                                            "min": round(float(min(value)), 2),
                                            "max": round(float(max(value)), 2),
                                        }

        # Process demand_realization_ratio
        if self.engine_props.get("can_partially_serve"):
            for bw, bw_obj in self.stats_props.demand_realization_ratio.items():
                if not isinstance(bw_obj, list):
                    continue
                if len(bw_obj) == 0:
                    self.stats_props.demand_realization_ratio[bw] = {
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
                    }
                    continue
                deviation = 0.0 if len(bw_obj) == 1 else stdev(bw_obj)
                num_full_served = sum(1 for val in bw_obj if val == 1)
                ratio_dict = {
                    "mean": round(mean(bw_obj), 2),
                    "std": round(deviation, 2),
                    "min": min(bw_obj),
                    "max": max(bw_obj),
                    "num_full_served": num_full_served,
                }
                if bw == "overall":
                    ratio_dict["ratio_full_served"] = num_full_served / self.engine_props["num_requests"]
                else:
                    request_dist = self.engine_props.get("request_distribution", {})
                    if bw in request_dist:
                        total_bw_requests = request_dist[bw] * self.engine_props["num_requests"]
                        ratio_dict["ratio_full_served"] = num_full_served / total_bw_requests if total_bw_requests > 0 else 0
                self.stats_props.demand_realization_ratio[bw] = ratio_dict

        # Process lp_bw_utilization_dict
        overall_was_converted = False
        for bw, bw_obj in self.stats_props.lp_bw_utilization_dict.items():
            if bw == "overall":
                if isinstance(bw_obj, list):
                    overall_was_converted = True
                    if len(bw_obj) == 0:
                        self.stats_props.lp_bw_utilization_dict[bw] = {
                            "mean": None,
                            "std": None,
                            "min": None,
                            "max": None,
                        }
                    else:
                        deviation = 0.0 if len(bw_obj) == 1 else stdev(bw_obj)
                        self.stats_props.lp_bw_utilization_dict[bw] = {
                            "mean": round(mean(bw_obj), 2),
                            "std": round(deviation, 2),
                            "min": round(min(bw_obj), 2),
                            "max": round(max(bw_obj), 2),
                        }
            else:
                if isinstance(bw_obj, dict):
                    for band, band_obj in bw_obj.items():
                        if isinstance(band_obj, dict):
                            for core, data_list in band_obj.items():
                                if isinstance(data_list, list):
                                    if len(data_list) == 0:
                                        band_obj[core] = {
                                            "mean": None,
                                            "std": None,
                                            "min": None,
                                            "max": None,
                                        }
                                    else:
                                        deviation = 0.0 if len(data_list) == 1 else stdev(data_list)
                                        band_obj[core] = {
                                            "mean": round(mean(data_list), 2),
                                            "std": round(deviation, 2),
                                            "min": round(min(data_list), 2),
                                            "max": round(max(data_list), 2),
                                        }

        # Track sim_lp_utilization_list from lp_bw_utilization_dict overall mean
        # Only append if we converted the list to dict in this call (prevents duplicates)
        if overall_was_converted:
            overall_util = self.stats_props.lp_bw_utilization_dict["overall"]
            if isinstance(overall_util, dict) and "mean" in overall_util:
                if overall_util["mean"] is not None:
                    self.stats_props.sim_lp_utilization_list.append(overall_util["mean"])

    def finalize_iteration_statistics(self) -> None:
        """
        Updates relevant stats after an iteration has finished.

        :return: None
        """
        if self.engine_props["num_requests"] == self.blocked_requests:
            self.stats_props.transponders_list.append(0)
        else:
            trans_mean = self.total_transponders / float(
                self.engine_props["num_requests"] - self.blocked_requests
            )
            self.stats_props.transponders_list.append(trans_mean)

        if self.blocked_requests > 0:
            # Check if already normalized (values are between 0 and 1)
            current_values = list(self.stats_props.block_reasons_dict.values())
            is_already_normalized = all(
                isinstance(v, float) and 0 <= v <= 1
                for v in current_values
                if v is not None and v > 0
            )

            if not is_already_normalized:
                for (
                    block_type,
                    num_times,
                ) in self.stats_props.block_reasons_dict.items():
                    if num_times is not None:
                        self.stats_props.block_reasons_dict[block_type] = (
                            num_times / float(self.blocked_requests)
                        )

        self._get_iter_means()

    def calculate_confidence_interval(self) -> bool:
        """
        Get the confidence interval for every iteration so far.

        :return: Whether the simulations should end for this erlang.
        :rtype: bool
        """
        self.block_mean = mean(self.stats_props.simulation_blocking_list)
        self.bit_rate_block_mean = mean(
            self.stats_props.simulation_bitrate_blocking_list
        )
        if len(self.stats_props.simulation_blocking_list) <= 1:
            return False

        self.block_variance = variance(self.stats_props.simulation_blocking_list)
        self.bit_rate_block_variance = variance(
            self.stats_props.simulation_bitrate_blocking_list
        )

        # Calculate bit rate blocking CI separately (always calculate, even when blocking is 0)
        # When variance is 0, this evaluates to 0.0 (matching v5 behavior)
        try:
            bit_rate_block_ci = 1.645 * (
                math.sqrt(self.bit_rate_block_variance)
                / math.sqrt(len(self.stats_props.simulation_bitrate_blocking_list))
            )
            self.bit_rate_block_ci = bit_rate_block_ci
            if self.bit_rate_block_mean > 0:
                bit_rate_block_ci_percent = (
                    (2 * bit_rate_block_ci) / self.bit_rate_block_mean
                ) * 100
                self.bit_rate_block_ci_percent = bit_rate_block_ci_percent
        except (ZeroDivisionError, ValueError):
            # If calculation fails, leave as None
            pass

        if self.block_mean == 0.0:
            # No request blocking means no CI to calculate for request blocking
            # Request blocking CI values remain None (initialized state)
            return False

        try:
            # Using 1.645 for 90% confidence level (matching v5 behavior)
            block_ci_rate = 1.645 * (
                math.sqrt(self.block_variance)
                / math.sqrt(len(self.stats_props.simulation_blocking_list))
            )
            self.block_ci = block_ci_rate
            block_ci_percent = ((2 * block_ci_rate) / self.block_mean) * 100
            self.block_ci_percent = block_ci_percent
        except ZeroDivisionError:
            return False

        # CI percent threshold should be configurable (tracked in core/TODO.md)
        if block_ci_percent <= 5:
            iter_val = self.iteration
            iteration_display = (iter_val + 1) if iter_val is not None else 1
            logger.info(
                "Confidence interval of %.2f%% reached. %d, ending for Erlang: %s",
                block_ci_percent,
                iteration_display,
                self.engine_props["erlang"],
            )
            return True

        return False

    def get_blocking_statistics(self) -> dict[str, Any]:
        """
        Get all blocking-related statistics for persistence.

        :return: Dictionary containing blocking statistics
        :rtype: dict
        """
        return {
            "block_mean": self.block_mean,
            "block_variance": self.block_variance,
            "block_ci": self.block_ci,
            "block_ci_percent": self.block_ci_percent,
            "bit_rate_block_mean": self.bit_rate_block_mean,
            "bit_rate_block_variance": self.bit_rate_block_variance,
            "bit_rate_block_ci": self.bit_rate_block_ci,
            "bit_rate_block_ci_percent": self.bit_rate_block_ci_percent,
            "iteration": self.iteration,
        }

    # Backward compatibility methods
    def end_iter_update(self) -> None:
        """
        Backward compatibility wrapper for finalize_iteration_statistics.
        """
        return self.finalize_iteration_statistics()

    def save_stats(self, base_fp: str = "data") -> None:
        """
        Backward compatibility method for saving statistics.

        :param base_fp: Base file path for saving
        """
        # Save mods_dict updates log for debugging
        import json
        from pathlib import Path
        if base_fp:
            log_fp = str(Path(base_fp).with_name(Path(base_fp).stem + '_mods_dict_log.json'))
            with open(log_fp, 'w') as f:
                json.dump(self.mods_dict_updates_log, f, indent=2)

        # Import here to avoid circular imports
        from fusion.core.persistence import (
            StatsPersistence,  # pylint: disable=import-outside-toplevel
        )

        # Ensure iteration is set to 0 if not initialized (for backward compatibility)
        if self.iteration is None:
            self.iteration = 0

        persistence = StatsPersistence(
            engine_props=self.engine_props, sim_info=self.sim_info
        )

        # Prepare save dict with iter_stats structure
        save_dict: dict[str, Any] = {"iter_stats": {}}

        # Get blocking statistics
        blocking_stats = self.get_blocking_statistics()

        # Save using the persistence module
        persistence.save_stats(
            stats_dict=save_dict,
            stats_props=self.stats_props,
            blocking_stats=blocking_stats,
            base_file_path=base_fp,
        )

    # Phase 3 - Survivability: Recovery Time Tracking Methods

    def record_recovery_event(
        self,
        failure_time: float,
        recovery_time: float,
        affected_requests: int,
        recovery_type: str,  # "protection" or "restoration"
    ) -> None:
        """
        Record a recovery event.

        :param failure_time: Time of failure occurrence
        :type failure_time: float
        :param recovery_time: Time when recovery completed
        :type recovery_time: float
        :param affected_requests: Number of affected requests
        :type affected_requests: int
        :param recovery_type: Type of recovery mechanism
        :type recovery_type: str

        Example:
            >>> stats.record_recovery_event(
            ...     failure_time=100.0,
            ...     recovery_time=100.05,  # 50ms later
            ...     affected_requests=5,
            ...     recovery_type='protection'
            ... )
        """
        # Compute recovery duration and round to avoid floating-point precision errors
        recovery_duration_ms = round((recovery_time - failure_time) * 1000, 10)

        # Store recovery time
        self.recovery_times_ms.append(recovery_duration_ms)

        # Store full event details
        event = {
            "failure_time": failure_time,
            "recovery_time": recovery_time,
            "recovery_duration_ms": recovery_duration_ms,
            "affected_requests": affected_requests,
            "recovery_type": recovery_type,
        }
        self.recovery_events.append(event)

        logger.info(
            "Recovery event: type=%s, duration=%.2fms, affected=%d",
            recovery_type,
            recovery_duration_ms,
            affected_requests,
        )

    def get_recovery_stats(self) -> dict[str, float]:
        """
        Get recovery time statistics.

        Computes mean, P95, and max recovery times across all
        recovery events.

        :return: Dict with recovery statistics
        :rtype: dict[str, float]

        Example:
            >>> stats = get_recovery_stats()
            >>> print(stats)
            {
                'mean_ms': 52.3,
                'p95_ms': 98.5,
                'max_ms': 105.2,
                'count': 12
            }
        """
        if not self.recovery_times_ms:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "count": 0}

        return {
            "mean_ms": float(np.mean(self.recovery_times_ms)),
            "p95_ms": float(np.percentile(self.recovery_times_ms, 95)),
            "max_ms": float(np.max(self.recovery_times_ms)),
            "count": len(self.recovery_times_ms),
        }

    def compute_failure_window_bp(
        self,
        failure_time: float,
        arrival_times: list[float],
        blocked_requests: list[int],
    ) -> float:
        """
        Compute blocking probability within failure window.

        Measures BP in the window [failure_time, failure_time + window_size]
        where window_size is specified in number of arrivals.

        :param failure_time: Failure occurrence time
        :type failure_time: float
        :param arrival_times: List of all request arrival times
        :type arrival_times: list[float]
        :param blocked_requests: List of blocked request indices
        :type blocked_requests: list[int]
        :return: BP within failure window
        :rtype: float

        Example:
            >>> bp = stats.compute_failure_window_bp(
            ...     failure_time=100.0,
            ...     arrival_times=all_arrivals,
            ...     blocked_requests=blocked_ids
            ... )
            >>> print(f"Failure window BP: {bp:.4f}")
            0.0823
        """
        # Find arrival index at failure time
        failure_index = np.searchsorted(arrival_times, failure_time)

        # Define window [failure_index, failure_index + window_size]
        window_end = min(failure_index + self.failure_window_size, len(arrival_times))

        # Count arrivals and blocks in window
        window_arrivals = window_end - failure_index
        window_blocks = sum(
            1 for req_id in blocked_requests if failure_index <= req_id < window_end
        )

        if window_arrivals == 0:
            return 0.0

        bp_value: float = float(window_blocks) / float(window_arrivals)
        self.failure_window_bp.append(bp_value)

        logger.info(
            "Failure window BP: %.4f (%d/%d blocked)",
            bp_value,
            window_blocks,
            window_arrivals,
        )

        return bp_value

    def get_failure_window_stats(self) -> dict[str, float]:
        """
        Get failure window BP statistics.

        :return: Dict with mean and P95 failure window BP
        :rtype: dict[str, float]
        """
        if not self.failure_window_bp:
            return {"mean": 0.0, "p95": 0.0, "count": 0}

        return {
            "mean": float(np.mean(self.failure_window_bp)),
            "p95": float(np.percentile(self.failure_window_bp, 95)),
            "count": len(self.failure_window_bp),
        }

    def get_recovery_csv_row(self) -> dict[str, Any]:
        """
        Export recovery statistics as CSV row.

        :return: Dict with all recovery metric values
        :rtype: dict[str, Any]
        """
        recovery_stats = self.get_recovery_stats()
        window_stats = self.get_failure_window_stats()

        return {
            # Recovery metrics
            "recovery_time_mean_ms": recovery_stats["mean_ms"],
            "recovery_time_p95_ms": recovery_stats["p95_ms"],
            "recovery_time_max_ms": recovery_stats["max_ms"],
            "recovery_event_count": recovery_stats["count"],
            # Failure window metrics
            "bp_window_fail_mean": window_stats["mean"],
            "bp_window_fail_p95": window_stats["p95"],
            "failure_window_count": window_stats["count"],
        }

    # Fragmentation and Decision Time Methods

    def compute_fragmentation_proxy(
        self,
        path: list[int],
        network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    ) -> float:
        """
        Compute fragmentation proxy for a path.

        Fragmentation = 1 - (largest_contiguous_block / total_free_slots)

        Higher values indicate more fragmentation.

        :param path: Path node list
        :type path: list[int]
        :param network_spectrum_dict: Spectrum state
        :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
        :return: Fragmentation score [0, 1]
        :rtype: float

        Example:
            >>> frag = stats.compute_fragmentation_proxy(path, spectrum_dict)
            >>> print(f"Fragmentation: {frag:.3f}")
            0.347
        """
        total_free = 0
        largest_contig = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            reverse_link = (path[i + 1], path[i])

            link_spectrum = network_spectrum_dict.get(
                link, network_spectrum_dict.get(reverse_link, {})
            )

            if not link_spectrum:
                continue

            # Get cores matrix
            cores_matrix = link_spectrum.get("cores_matrix", [])
            if not cores_matrix:
                continue

            # Process first core (for simplicity)
            slots = cores_matrix[0] if len(cores_matrix) > 0 else []

            # Find free blocks
            free_blocks = self._find_free_blocks(slots)

            if free_blocks:
                link_total = sum(block[1] - block[0] for block in free_blocks)
                link_largest = max(block[1] - block[0] for block in free_blocks)

                total_free += link_total
                largest_contig = max(largest_contig, link_largest)

        if total_free == 0:
            return 1.0  # Fully fragmented

        frag = 1.0 - (largest_contig / total_free)
        return frag

    def _find_free_blocks(self, slots: np.ndarray) -> list[tuple[int, int]]:
        """
        Find contiguous free blocks in spectrum.

        :param slots: Spectrum slot array
        :type slots: np.ndarray
        :return: List of (start, end) tuples for free blocks
        :rtype: list[tuple[int, int]]
        """
        blocks = []
        start = None

        for i, slot in enumerate(slots):
            if slot == 0:  # Free
                if start is None:
                    start = i
            else:  # Occupied
                if start is not None:
                    blocks.append((start, i))
                    start = None

        if start is not None:
            blocks.append((start, len(slots)))

        return blocks

    def record_fragmentation(
        self,
        path: list[int],
        network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    ) -> None:
        """
        Record fragmentation score for a path.

        :param path: Path node list
        :type path: list[int]
        :param network_spectrum_dict: Spectrum state
        :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
        """
        frag_score = self.compute_fragmentation_proxy(path, network_spectrum_dict)
        self.fragmentation_scores.append(frag_score)

    def record_decision_time(self, decision_time_ms: float) -> None:
        """
        Record policy decision time.

        :param decision_time_ms: Decision time in milliseconds
        :type decision_time_ms: float
        """
        self.decision_times_ms.append(decision_time_ms)

    def get_fragmentation_stats(self) -> dict[str, float]:
        """
        Get fragmentation statistics.

        :return: Dict with mean and P95 fragmentation scores
        :rtype: dict[str, float]
        """
        if not self.fragmentation_scores:
            return {"mean": 0.0, "p95": 0.0, "count": 0}

        return {
            "mean": float(np.mean(self.fragmentation_scores)),
            "p95": float(np.percentile(self.fragmentation_scores, 95)),
            "count": len(self.fragmentation_scores),
        }

    def get_decision_time_stats(self) -> dict[str, float]:
        """
        Get decision time statistics.

        :return: Dict with mean and P95 decision times
        :rtype: dict[str, float]
        """
        if not self.decision_times_ms:
            return {"mean": 0.0, "p95": 0.0, "count": 0}

        return {
            "mean": float(np.mean(self.decision_times_ms)),
            "p95": float(np.percentile(self.decision_times_ms, 95)),
            "count": len(self.decision_times_ms),
        }

    def to_csv_row(self) -> dict[str, Any]:
        """
        Export all statistics as CSV row.

        Includes standard metrics plus survivability-specific metrics.

        :return: Dict with all metric values
        :rtype: dict[str, Any]
        """
        # Get survivability stats
        recovery_stats = self.get_recovery_stats()
        window_stats = self.get_failure_window_stats()
        frag_stats = self.get_fragmentation_stats()
        decision_stats = self.get_decision_time_stats()

        # Get failure settings
        failure_settings = self.engine_props.get("failure_settings", {})
        routing_settings = self.engine_props.get("routing_settings", {})
        rl_settings = self.engine_props.get("offline_rl_settings", {})

        return {
            # Experiment parameters
            "topology": self.engine_props.get("network", "unknown"),
            "load": self.engine_props.get("erlang", 0),
            "failure_type": failure_settings.get("failure_type", "none"),
            "k_paths": routing_settings.get("k_paths", 1),
            "policy": rl_settings.get("policy_type", "ksp_ff"),
            "seed": self.engine_props.get("seed", 0),
            # Standard metrics
            "bp_overall": self.block_mean if self.block_mean is not None else 0.0,
            "bp_variance": self.block_variance
            if self.block_variance is not None
            else 0.0,
            "bp_ci_percent": (
                self.block_ci_percent if self.block_ci_percent is not None else 0.0
            ),
            "bit_rate_bp": (
                self.bit_rate_block_mean
                if self.bit_rate_block_mean is not None
                else 0.0
            ),
            # Failure window metrics
            "bp_window_fail_mean": window_stats["mean"],
            "bp_window_fail_p95": window_stats["p95"],
            # Recovery metrics
            "recovery_time_mean_ms": recovery_stats["mean_ms"],
            "recovery_time_p95_ms": recovery_stats["p95_ms"],
            "recovery_time_max_ms": recovery_stats["max_ms"],
            # Fragmentation
            "frag_proxy_mean": frag_stats["mean"],
            "frag_proxy_p95": frag_stats["p95"],
            # Decision times
            "decision_time_mean_ms": decision_stats["mean"],
            "decision_time_p95_ms": decision_stats["p95"],
        }

    # =========================================================================
    # P3.4 Stats Integration: New orchestrator path methods
    # =========================================================================

    def record_arrival(
        self,
        request: Any,  # fusion.domain.request.Request
        result: Any,  # fusion.domain.results.AllocationResult
        network_state: Any,  # fusion.domain.network_state.NetworkState
        was_rollback: bool = False,
    ) -> None:
        """
        Record statistics from orchestrator allocation result.

        This method provides the same functionality as iter_update but
        works with Phase 1 domain objects instead of legacy sdn_props.

        Args:
            request: The Request that was processed
            result: The AllocationResult from orchestrator
            network_state: Current NetworkState for spectrum data
            was_rollback: True if this was a rollback (skip utilization tracking)

        Side Effects:
            - Updates blocked_requests count
            - Updates bandwidth statistics
            - Updates resource usage metrics
            - Updates quality metrics (if available)
            - Updates block reasons (if blocked)
            - Takes periodic snapshots

        P3.6 Gap Coverage:
            - Gap 5: SNR_RECHECK_FAIL block reason support
            - Gap 5: Skip utilization for rolled-back lightpaths
            - Gap 1: Grooming rollback utilization adjustment
        """
        # Increment total requests for tracking
        self.total_requests += 1

        if not result.success:
            self._record_blocked_request_new(request, result)
        else:
            self._record_successful_allocation_new(
                request, result, network_state, was_rollback
            )

        # Take periodic snapshot
        self._maybe_take_snapshot_new(network_state)

    def _record_blocked_request_new(
        self,
        request: Any,  # fusion.domain.request.Request
        result: Any,  # fusion.domain.results.AllocationResult
    ) -> None:
        """
        Record stats for a blocked request (new orchestrator path).

        Args:
            request: The Request that was blocked
            result: The AllocationResult with block_reason

        P3.6 Gap 5 Coverage:
            - Maps SNR_RECHECK_FAIL to 'xt_threshold' for backwards compat
        """
        self.blocked_requests += 1

        # Track bandwidth blocking
        if hasattr(request, 'bandwidth_gbps'):
            self.bit_rate_blocked += int(request.bandwidth_gbps)
            self.bit_rate_request += int(request.bandwidth_gbps)

            # Track by bandwidth class
            bw_class = str(request.bandwidth_gbps)
            if bw_class in self.stats_props.bandwidth_blocking_dict:
                self.stats_props.bandwidth_blocking_dict[bw_class] += 1

        # Update block reason
        if result.block_reason:
            reason_key = self._map_block_reason_new(result.block_reason)
            current = self.stats_props.block_reasons_dict.get(reason_key)
            if current is None:
                self.stats_props.block_reasons_dict[reason_key] = 1
            else:
                self.stats_props.block_reasons_dict[reason_key] = current + 1

    def _record_successful_allocation_new(
        self,
        request: Any,  # fusion.domain.request.Request
        result: Any,  # fusion.domain.results.AllocationResult
        network_state: Any,  # fusion.domain.network_state.NetworkState
        was_rollback: bool = False,
    ) -> None:
        """
        Record stats for a successful allocation (new orchestrator path).

        Args:
            request: The Request that was allocated
            result: The AllocationResult with allocation details
            network_state: Current NetworkState
            was_rollback: True if rollback occurred (skip utilization tracking)

        P3.6 Gap Coverage:
            - Gap 5: Skip utilization tracking if was_rollback=True
            - Gap 1: Adjusts grooming utilization appropriately
        """
        # Track allocation type counters
        if result.is_groomed:
            self.groomed_requests += 1
        if getattr(result, 'is_sliced', False):
            self.sliced_requests += 1
        if getattr(result, 'is_protected', False):
            self.protected_requests += 1

        # Skip further stats tracking for fully groomed requests (v5 behavior)
        if result.is_groomed and not result.is_partially_groomed:
            if hasattr(request, 'bandwidth_gbps'):
                self.bit_rate_request += int(request.bandwidth_gbps)
            return

        # Track bit rate for requests
        if hasattr(request, 'bandwidth_gbps'):
            self.bit_rate_request += int(request.bandwidth_gbps)

        # Resource metrics - transponders (count of lightpaths created)
        transponders = len(result.lightpaths_created) if result.lightpaths_created else 1
        self.total_transponders += transponders

        # Track path metrics once (using first lightpath for path/hops)
        first_lp_details = self._get_lightpath_details_new(result, network_state)
        if first_lp_details.get('path'):
            path = first_lp_details['path']
            num_hops = len(path) - 1
            self.stats_props.hops_list.append(float(num_hops))

            path_len = self._calculate_path_length_new(path, network_state)
            if path_len is not None:
                self.stats_props.lengths_list.append(round(float(path_len), 2))

        # Track path index from allocation result
        path_idx = getattr(result, 'path_index', 0)
        if 0 <= path_idx < len(self.stats_props.path_index_list):
            self.stats_props.path_index_list[path_idx] += 1

        # Track per-lightpath stats (core, modulation, weights) for ALL lightpaths
        # This is critical for sliced requests where multiple lightpaths are created
        if result.lightpaths_created and network_state is not None:
            # Collect SNR values for all lightpaths in this request
            # Legacy behavior: append mean of all lightpaths' SNR for a single request
            request_snr_values: list[float] = []

            for lp_id in result.lightpaths_created:
                lp = network_state.get_lightpath(lp_id)
                if lp is None:
                    continue

                # Track core usage (like legacy _handle_iter_lists)
                core = getattr(lp, 'core', None)
                if core is not None:
                    if core not in self.stats_props.cores_dict:
                        self.stats_props.cores_dict[core] = 0
                    self.stats_props.cores_dict[core] += 1

                # Get lightpath details for modulation tracking
                modulation = getattr(lp, 'modulation', None)
                band = getattr(lp, 'band', None)
                snr_db = getattr(lp, 'snr_db', None)
                lp_path = getattr(lp, 'path', None)
                # Use LIGHTPATH bandwidth, not request bandwidth (critical for slicing)
                lp_bandwidth = getattr(lp, 'total_bandwidth_gbps', None)

                # Calculate path weight for this lightpath
                path_weight = None
                num_hops = None
                if lp_path:
                    path_weight = self._calculate_path_length_new(lp_path, network_state)
                    num_hops = len(lp_path) - 1

                # Track modulation with correct lightpath bandwidth
                if modulation:
                    self._increment_modulation_count(
                        modulation,
                        bandwidth_gbps=lp_bandwidth,
                        band=band,
                        path_weight=path_weight,
                        num_hops=num_hops,
                        snr_value=snr_db,
                    )

                # Collect SNR for this lightpath
                if snr_db is not None:
                    request_snr_values.append(snr_db)

            # Append mean of all lightpaths' SNR for this request (legacy behavior)
            if request_snr_values:
                self.stats_props.snr_list.append(mean(request_snr_values))
        else:
            # Fallback for non-lightpath allocations (shouldn't happen often)
            if first_lp_details.get('modulation'):
                bandwidth = request.bandwidth_gbps if hasattr(request, 'bandwidth_gbps') else None
                path_weight = None
                num_hops = None
                if first_lp_details.get('path'):
                    path_weight = self._calculate_path_length_new(first_lp_details['path'], network_state)
                    num_hops = len(first_lp_details['path']) - 1
                self._increment_modulation_count(
                    first_lp_details['modulation'],
                    bandwidth_gbps=bandwidth,
                    band=first_lp_details.get('band'),
                    path_weight=path_weight,
                    num_hops=num_hops,
                    snr_value=first_lp_details.get('snr_db'),
                )

            if first_lp_details.get('snr_db') is not None:
                self.stats_props.snr_list.append(first_lp_details['snr_db'])

    def _get_lightpath_details_new(
        self,
        result: Any,  # fusion.domain.results.AllocationResult
        network_state: Any,  # fusion.domain.network_state.NetworkState
    ) -> dict[str, Any]:
        """
        Extract lightpath details from result and network state.

        Args:
            result: AllocationResult
            network_state: NetworkState

        Returns:
            Dict with path, modulation, snr_db, etc.
        """
        details: dict[str, Any] = {
            'path': None,
            'modulation': None,
            'snr_db': None,
            'crosstalk_db': None,
            'core': None,
            'band': None,
            'start_slot': None,
            'end_slot': None,
        }

        # Try to get from lightpaths_created
        if result.lightpaths_created and network_state is not None:
            lp_id = result.lightpaths_created[0]
            lp = network_state.get_lightpath(lp_id)

            if lp is not None:
                details['path'] = getattr(lp, 'path', None)
                details['modulation'] = getattr(lp, 'modulation', None)
                details['snr_db'] = getattr(lp, 'snr_db', None)
                details['crosstalk_db'] = getattr(lp, 'crosstalk_db', None)
                details['core'] = getattr(lp, 'core', None)
                details['band'] = getattr(lp, 'band', None)
                details['start_slot'] = getattr(lp, 'start_slot', None)
                details['end_slot'] = getattr(lp, 'end_slot', None)

        # Fall back to result fields if lightpath lookup failed
        if details['modulation'] is None and result.modulations:
            details['modulation'] = result.modulations[0]

        if details['snr_db'] is None and result.snr_values:
            details['snr_db'] = result.snr_values[0]

        return details

    def _map_block_reason_new(self, reason: Any) -> str:
        """
        Map BlockReason enum to stats dict key.

        Args:
            reason: BlockReason enum value

        Returns:
            Legacy stats dict key (distance, congestion, xt_threshold, failure)

        P3.6 Gap 5 Coverage:
            - SNR_RECHECK_FAIL maps to 'xt_threshold' for backwards compat
        """
        from fusion.domain.request import BlockReason

        mapping = {
            BlockReason.NO_PATH: 'distance',
            BlockReason.DISTANCE: 'distance',
            BlockReason.CONGESTION: 'congestion',
            BlockReason.SNR_THRESHOLD: 'xt_threshold',
            BlockReason.SNR_RECHECK_FAIL: 'xt_threshold',  # P3.6 Gap 5
            BlockReason.XT_THRESHOLD: 'xt_threshold',
            BlockReason.FAILURE: 'failure',
            BlockReason.LINK_FAILURE: 'failure',
            BlockReason.NODE_FAILURE: 'failure',
            BlockReason.GROOMING_FAIL: 'congestion',
            BlockReason.SLICING_FAIL: 'congestion',
            BlockReason.PROTECTION_FAIL: 'congestion',
        }
        return mapping.get(reason, 'congestion')

    def _calculate_path_length_new(
        self,
        path: tuple[str, ...] | list[str],
        network_state: Any,  # fusion.domain.network_state.NetworkState
    ) -> float | None:
        """
        Calculate total path length in km.

        Args:
            path: Tuple or list of node IDs
            network_state: NetworkState with topology

        Returns:
            Total path length in km or None if cannot calculate
        """
        if len(path) < 2:
            return 0.0

        if network_state is None:
            return None

        topology = getattr(network_state, 'topology', None)
        if topology is None:
            return None

        total_length = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if topology.has_edge(u, v):
                edge_data = topology.edges[u, v]
                total_length += edge_data.get('length', edge_data.get('weight', 0.0))
            elif topology.has_edge(v, u):
                edge_data = topology.edges[v, u]
                total_length += edge_data.get('length', edge_data.get('weight', 0.0))

        return total_length

    def record_grooming_rollback(
        self,
        request: Any,  # fusion.domain.request.Request
        rolled_back_lightpath_ids: list[int],
        network_state: Any,  # fusion.domain.network_state.NetworkState
    ) -> None:
        """
        Record that grooming was rolled back - adjust utilization stats.

        Called when partial grooming succeeds but new lightpath allocation
        fails, requiring rollback of the groomed bandwidth.

        Args:
            request: The Request being processed
            rolled_back_lightpath_ids: IDs of lightpaths that had grooming rolled back
            network_state: Current NetworkState

        P3.6 Gap 1 Coverage:
            - Adjusts utilization tracking when grooming is rolled back
            - Does NOT add utilization entries for rolled-back allocations
        """
        # For P3.6 Gap 1: We don't need to undo utilization here because
        # utilization is only tracked at release time. The grooming rollback
        # in the orchestrator restores the bandwidth to the lightpath,
        # so when the lightpath eventually releases, it will have the
        # correct utilization.
        logger.debug(
            "Grooming rollback recorded for request %s on lightpaths %s",
            getattr(request, 'request_id', 'unknown'),
            rolled_back_lightpath_ids,
        )

    def _maybe_take_snapshot_new(
        self,
        network_state: Any,  # fusion.domain.network_state.NetworkState
    ) -> None:
        """
        Take periodic network snapshot (new orchestrator path).

        Snapshots are taken at configurable intervals to track simulation
        progress over time.

        Args:
            network_state: Current NetworkState for spectrum data
        """
        # Check if snapshots are enabled
        if not self.engine_props.get("save_snapshots", False):
            return

        # Check if it's time for a snapshot
        snapshot_interval = getattr(self, 'snapshot_interval', 100)
        if self.total_requests % snapshot_interval != 0:
            return

        # Skip if no requests yet
        if self.total_requests == 0:
            return

        snapshot = {
            'request_number': self.total_requests,
            'blocked_count': self.blocked_requests,
            'spectrum_utilization': self._calculate_spectrum_utilization_new(
                network_state
            ),
        }

        self.stats_props.snapshots_dict[self.total_requests] = snapshot

    def _calculate_spectrum_utilization_new(
        self,
        network_state: Any,  # fusion.domain.network_state.NetworkState
    ) -> float:
        """
        Calculate overall spectrum utilization from network state.

        Args:
            network_state: Current NetworkState

        Returns:
            Utilization as fraction (0.0 to 1.0)
        """
        if network_state is None:
            return 0.0

        total_slots = 0
        used_slots = 0

        # Get spectrum utilization from network state
        network_spectrum_dict = getattr(network_state, 'network_spectrum_dict', None)
        if network_spectrum_dict is None:
            return 0.0

        for link_key, link_data in network_spectrum_dict.items():
            cores_matrix = link_data.get('cores_matrix', {})
            for band, cores in cores_matrix.items():
                for core_spectrum in cores:
                    if hasattr(core_spectrum, '__len__'):
                        total_slots += len(core_spectrum)
                        used_slots += sum(1 for s in core_spectrum if s != 0)

        if total_slots == 0:
            return 0.0
        return used_slots / total_slots

    def _record_bandwidth_new(
        self,
        bandwidth_gbps: int,
        blocked: bool,
    ) -> None:
        """
        Record bandwidth for blocking statistics (new path).

        Args:
            bandwidth_gbps: Bandwidth of the request
            blocked: True if request was blocked
        """
        # Track total requested (using existing bit_rate_request)
        self.bit_rate_request += bandwidth_gbps

        if blocked:
            # Track blocked bandwidth (using existing bit_rate_blocked)
            self.bit_rate_blocked += bandwidth_gbps

            # Track by bandwidth class
            bw_class = str(bandwidth_gbps)
            current = self.stats_props.bandwidth_blocking_dict.get(bw_class, 0)
            self.stats_props.bandwidth_blocking_dict[bw_class] = current + 1

    def _increment_modulation_count(
        self,
        modulation: str,
        bandwidth_gbps: int | None = None,
        band: str | None = None,
        path_weight: float | None = None,
        num_hops: int | None = None,
        snr_value: float | None = None,
    ) -> None:
        """
        Increment modulation usage count and track detailed metrics (new orchestrator path).

        Updates modulations_used_dict and weights_dict to track modulation usage including
        length, hop, SNR, and XT cost statistics per band.

        Args:
            modulation: Modulation format name (e.g., 'QPSK', '16-QAM')
            bandwidth_gbps: Optional bandwidth for bandwidth-keyed tracking
            band: Optional band for band-specific tracking
            path_weight: Path length/weight in km for length tracking
            num_hops: Number of hops for hop tracking
            snr_value: SNR or XT cost value for quality tracking
        """
        if modulation is None:
            return

        mod_dict = self.stats_props.modulations_used_dict

        # Track by bandwidth key if provided
        if bandwidth_gbps is not None:
            bandwidth_key = str(int(bandwidth_gbps))
            bw_dict = mod_dict.get(bandwidth_key)
            if isinstance(bw_dict, dict):
                if modulation in bw_dict:
                    bw_dict[modulation] += 1
                else:
                    bw_dict[modulation] = 1

            # Also track in weights_dict (path weight per bandwidth/modulation)
            if path_weight is not None:
                weights_dict = self.stats_props.weights_dict
                if bandwidth_key in weights_dict:
                    if modulation in weights_dict[bandwidth_key]:
                        weights_dict[bandwidth_key][modulation].append(
                            round(float(path_weight), 2)
                        )
                    else:
                        weights_dict[bandwidth_key][modulation] = [
                            round(float(path_weight), 2)
                        ]

        # Track by modulation name with band
        data_mod_dict = mod_dict.get(modulation)
        if isinstance(data_mod_dict, dict):
            # Increment band count
            if band and band in data_mod_dict:
                data_mod_dict[band] += 1
            elif band:
                data_mod_dict[band] = 1

            # Track length
            if path_weight is not None:
                length_dict = data_mod_dict.get("length")
                if length_dict and isinstance(length_dict, dict):
                    if band and band in length_dict:
                        length_dict[band].append(path_weight)
                    if "overall" in length_dict:
                        length_dict["overall"].append(path_weight)

            # Track hop count
            if num_hops is not None:
                hop_dict = data_mod_dict.get("hop")
                if hop_dict and isinstance(hop_dict, dict):
                    if band and band in hop_dict:
                        hop_dict[band].append(num_hops)
                    if "overall" in hop_dict:
                        hop_dict["overall"].append(num_hops)

            # Track SNR or XT cost
            if snr_value is not None:
                snr_type = self.engine_props.get("snr_type")
                if snr_type == "xt_calculation":
                    xt_dict = data_mod_dict.get("xt_cost")
                    if xt_dict and isinstance(xt_dict, dict):
                        if band and band in xt_dict:
                            xt_dict[band].append(snr_value)
                        if "overall" in xt_dict:
                            xt_dict["overall"].append(snr_value)
                else:
                    snr_dict = data_mod_dict.get("snr")
                    if snr_dict and isinstance(snr_dict, dict):
                        if band and band in snr_dict:
                            snr_dict[band].append(snr_value)
                        if "overall" in snr_dict:
                            snr_dict["overall"].append(snr_value)
