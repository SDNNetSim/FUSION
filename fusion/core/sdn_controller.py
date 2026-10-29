"""
Software-defined network controller module for managing network requests.

This module provides the main SDN controller functionality for routing and spectrum
allocation in software-defined optical networks.
"""

import copy
import json
import math
import time
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

import numpy as np

from fusion.core.grooming import Grooming
from fusion.core.hcf import HCFCandidate, HCFMode, dbm_to_watts, hcf_power_objective
from fusion.core.properties import SDNProps
from fusion.core.routing import Routing
from fusion.core.snr_measurements import SnrMeasurements
from fusion.core.spectrum_assignment import SpectrumAssignment
from fusion.modules.ml import get_ml_obs
from fusion.modules.spectrum.light_path_slicing import LightPathSlicingManager
from fusion.utils.bandwidth import parse_bandwidth
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


class SDNController:
    """
    Software-defined network controller for managing network requests.

    This class provides functionality for routing, spectrum allocation, and resource
    management in software-defined optical networks. It handles request allocation,
    release, and various slicing strategies.

    :param engine_props: Engine configuration properties
    :type engine_props: dict[str, Any]
    """

    def __init__(self, engine_props: dict[str, Any]) -> None:
        self.engine_props = engine_props
        self.sdn_props = SDNProps()

        self.ai_obj = None
        self.route_obj = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.spectrum_obj = SpectrumAssignment(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            route_props=self.route_obj.route_props,
        )
        self.slicing_manager = LightPathSlicingManager(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_obj=self.spectrum_obj,
        )
        self.grooming_obj = Grooming(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.stats = None

        # FailureManager reference for path feasibility checking
        # (set by SimulationEngine)
        self.failure_manager: Any | None = None

        # Separate, opt-in HCF state; the legacy handle_event path is unchanged.
        self.hcf_power_used: dict[tuple, float] = {}
        self.hcf_allocations: dict[int, HCFCandidate] = {}
        self.hcf_last_result: dict[str, Any] = {}

    def _hcf_settings(self) -> tuple[list[HCFMode], list[float]]:
        """Validate the deliberately narrow single-carrier HCF experiment contract."""
        props = self.engine_props
        required = {"fixed_grid": True, "hollow_core_fiber": True}
        if any(props.get(key) != value for key, value in required.items()):
            raise ValueError("HCF allocation requires fixed_grid and hollow_core_fiber=True")
        bidirectional = props.get("bi_directional", props.get("topology_settings", {}).get("bi_directional", False))
        if bidirectional is not True:
            raise ValueError("HCF allocation requires bi_directional=True")
        if props.get("cores_per_link") != 1 or props.get("guard_slots", 0) != 0 or props.get("max_segments", 1) != 1:
            raise ValueError("HCF allocation requires one core, zero guard slots and max_segments=1")
        disabled = ("snr_recheck", "dynamic_lps", "is_grooming_enabled", "can_partially_serve", "transponder_usage_per_node")
        if any(props.get(key, False) for key in disabled):
            raise ValueError("HCF allocation does not support rechecks, slicing, grooming, partial service or transponder limits")
        if props.get("route_method") not in ("shortest_path", "k_shortest_path") or props.get("protection_enabled", False):
            raise ValueError("HCF allocation requires unprotected shortest/k-shortest paths")
        if self.failure_manager is not None:
            raise ValueError("HCF allocation does not support failure/recovery experiments")
        if props.get("hcf_selection_policy", "joint") not in ("joint", "power", "first_feasible", "first_channel_ranked", "power_spectrum"):
            raise ValueError("Unknown hcf_selection_policy")
        objective = hcf_power_objective(props)
        if objective == "fixed":
            fixed_power = float(props.get("input_power", 0))
            if not math.isfinite(fixed_power) or fixed_power <= 0:
                raise ValueError("HCF fixed input_power must be finite and positive watts")
        weight = float(props.get("hcf_power_weight", 0.5))
        margin = float(props.get("hcf_margin_db", 0.0))
        if not 0 <= weight <= 1 or not math.isfinite(margin) or margin < 0:
            raise ValueError("HCF power weight must be in [0,1] and margin finite and nonnegative")
        modes_data = props.get("hcf_modes")
        powers_data = props.get("hcf_power_levels_dbm", [])
        modes_data = json.loads(modes_data) if isinstance(modes_data, str) else modes_data
        powers_data = json.loads(powers_data) if isinstance(powers_data, str) else powers_data
        if modes_data is None:
            # Reuse the simulator's existing rate/threshold/symbol mappings.
            snr = self.spectrum_obj.snr_measurements.snr_props
            capacities = snr.bandwidth_mapping_dict[float(props["bw_per_slot"])]
            modes = [HCFMode(name, rate, rate / (2 * snr.bits_per_symbol[name]), snr.req_snr[name]) for name, rate in capacities.items()]
        else:
            modes = [HCFMode(**item) for item in modes_data]
        powers = sorted({dbm_to_watts(float(value)) for value in powers_data})
        if not modes or (objective == "discrete" and not powers) or len({mode.name for mode in modes}) != len(modes):
            raise ValueError("HCF modes must be nonempty and unique; discrete power requires a nonempty grid")
        if float(props.get("bw_per_slot", 0)) <= 0:
            raise ValueError("HCF channel spacing must be positive")
        if int(props.get("k_paths", 1)) < 1:
            raise ValueError("HCF k_paths must be positive")
        if props.get("hcf_amplifier_budget_dbm") is None:
            raise ValueError("Provide hcf_amplifier_budget_dbm: per-band signal output budgets")
        return modes, powers

    def _hcf_link(self, source: Any, destination: Any) -> dict[str, Any]:
        network = self.sdn_props.network_spectrum_dict
        if network is None:
            raise ValueError("HCF spectrum state is not initialized")
        link_id = network[(source, destination)]["link_num"]
        return self.engine_props["topology_info"]["links"][link_id]

    def _hcf_paths(self, source: Any, destination: Any) -> list[list[Any]]:
        """Use the existing KSP implementation for probe endpoint pairs."""
        k = 1 if self.engine_props["route_method"] == "shortest_path" else int(self.engine_props.get("k_paths", 3))
        algorithm = self.route_obj._get_algorithm_for_method("k_shortest_path")
        return algorithm.get_paths(source, destination, k=k)




    @contextmanager
    def _hcf_transaction(self, candidate: HCFCandidate):
        """Restore touched resources and bookkeeping if allocation or release fails."""
        props, spectrum = self.sdn_props, self.spectrum_obj.spectrum_props
        persistent = {"network_spectrum_dict", "topology", "lightpath_status_dict", "transponder_usage_dict", "allocated_requests"}
        request_before = {key: copy.deepcopy(value) for key, value in vars(props).items() if key not in persistent}
        spectrum_before = copy.deepcopy(vars(spectrum))
        endpoint = tuple(sorted((candidate.path[0], candidate.path[-1])))
        statuses = props.lightpath_status_dict
        status_before = None if statuses is None or endpoint not in statuses else copy.deepcopy(statuses[endpoint])
        tracking_before = copy.deepcopy(props.allocated_requests.get(props.request_id))
        cells = []
        for source, destination in zip(candidate.path, candidate.path[1:], strict=False):
            for edge in ((source, destination), (destination, source)):
                link = props.network_spectrum_dict[edge]
                array = link["cores_matrix"][candidate.band][0]
                cells.append((link, array, array[candidate.channel], link["usage_count"], link["throughput"]))
        try:
            yield
        except Exception:
            for link, array, old_value, usage, throughput in cells:
                array[candidate.channel] = old_value
                link["usage_count"], link["throughput"] = usage, throughput
            for key, value in request_before.items():
                setattr(props, key, value)
            vars(spectrum).clear()
            vars(spectrum).update(spectrum_before)
            if statuses is None:
                props.lightpath_status_dict = None
            elif status_before is None:
                statuses.pop(endpoint, None)
            else:
                statuses[endpoint] = status_before
            if tracking_before is None:
                props.allocated_requests.pop(props.request_id, None)
            else:
                props.allocated_requests[props.request_id] = tracking_before
            raise

    def _commit_hcf_candidate(self, candidate: HCFCandidate, path_index: int, route_time: float, score: float) -> None:
        """Commit exactly one channel and its power; never invoke a GSNR recheck."""
        channel_available = candidate.channel in self.spectrum_obj.get_hcf_free_channels(candidate.path, candidate.band)
        if not channel_available or not self.spectrum_obj.hcf_power_fits(candidate, self.hcf_power_used):
            raise BufferError("HCF resources changed before commitment")
        with self._hcf_transaction(candidate):
            props, spectrum = self.sdn_props, self.spectrum_obj.spectrum_props
            props.path_list = list(candidate.path)
            props.path_index = path_index
            props.path_weight = sum(self._hcf_link(a, b)["length"] for a, b in zip(candidate.path, candidate.path[1:], strict=False))
            props.route_time = route_time
            spectrum.path_list = list(candidate.path)
            spectrum.backup_path = None
            spectrum.slots_needed = 1
            spectrum.start_slot = spectrum.end_slot = candidate.channel
            spectrum.core_number = 0
            spectrum.current_band = candidate.band
            spectrum.modulation = candidate.mode.name
            spectrum.lightpath_bandwidth = candidate.mode.net_rate_gbps
            spectrum.crosstalk_cost = candidate.gsnr_db
            spectrum.is_free = True
            spectrum.lightpath_id = props.get_lightpath_id()
            props.was_new_lp_established = [spectrum.lightpath_id]
            props.spectrum_object = spectrum
            props.is_sliced = False
            self.allocate()
            self._update_request_statistics(parse_bandwidth(props.bandwidth))
            self.spectrum_obj._update_lightpath_status()
            metadata = {"launch_power_w": candidate.power_w, "gsnr_db": candidate.gsnr_db, "hcf_score": score}
            endpoint = tuple(sorted((candidate.path[0], candidate.path[-1])))
            props.lightpath_status_dict[endpoint][spectrum.lightpath_id].update(metadata)
            props.allocated_requests[props.request_id].update(metadata)
            props.block_reason = None
            props.was_routed = True
            # No fallible callbacks after the ledger is committed.
            for key, power, _ in candidate.demands:
                self.hcf_power_used[key] = self.hcf_power_used.get(key, 0.0) + power
            self.hcf_allocations[spectrum.lightpath_id] = candidate

    def reset_hcf_state(self) -> None:
        """Clear HCF resources at an independent iteration boundary before ID reuse."""
        for lp_id, candidate in self.hcf_allocations.items():
            endpoint = tuple(sorted((candidate.path[0], candidate.path[-1])))
            if self.sdn_props.lightpath_status_dict:
                self.sdn_props.lightpath_status_dict.get(endpoint, {}).pop(lp_id, None)
            network = self.sdn_props.network_spectrum_dict
            if network:
                for source, destination in zip(candidate.path, candidate.path[1:], strict=False):
                    for edge in ((source, destination), (destination, source)):
                        array = network[edge]["cores_matrix"][candidate.band][0]
                        if array[candidate.channel] == lp_id:
                            array[candidate.channel] = 0
        self.sdn_props.allocated_requests.clear()
        self.hcf_allocations.clear()
        self.hcf_power_used.clear()
        self.hcf_last_result = {}
        self.spectrum_obj.clear_hcf_probe_cache()
        self.spectrum_obj.clear_hcf_transmission_cache()
        self.spectrum_obj.snr_measurements.clear_hcf_physical_cache()

    def handle_hcf_event(
        self,
        request_dict: dict[str, Any],
        request_type: str,
        force_slicing: bool = False,
        force_route_matrix: list[Any] | None = None,
        forced_index: int | None = None,
        force_core: int | None = None,
        ml_model: Any | None = None,
        force_mod_format: str | list[str] | None = None,
        forced_band: str | None = None,
    ) -> None:
        """Full-HCF dynamic WDM allocation: path -> wavelength -> candidate GSNR.

        Existing policies commit on the first feasible path. first_channel_ranked
        compares the first common-free channel of every path before commitment.
        Modes and power levels are inspected inside the wavelength loop. Established
        channels are neither retuned nor re-evaluated. SimulationEngine prepares
        SDNProps before calling, exactly as for the unchanged handle_event().
        """
        if request_type == "release":
            ids = self.sdn_props.lightpath_id_list
            if not ids or all(lp_id not in self.hcf_allocations for lp_id in ids):
                return
            if len(ids) != 1:
                raise ValueError("HCF release expects one lightpath per request")
            lp_id = ids[0]
            candidate = self.hcf_allocations[lp_id]
            with self._hcf_transaction(candidate):
                self.sdn_props.path_list = list(candidate.path)
                self.handle_event(request_dict, request_type="release")
                self.sdn_props.allocated_requests.pop(self.sdn_props.request_id, None)
            for key, power, _ in candidate.demands:
                remaining = self.hcf_power_used[key] - power
                if remaining <= power * 1e-12:
                    self.hcf_power_used.pop(key, None)
                else:
                    self.hcf_power_used[key] = remaining
            del self.hcf_allocations[lp_id]
            return
        if request_type != "arrival":
            raise ValueError(f"Unsupported HCF event: {request_type}")
        modes, powers = self._hcf_settings()
        if force_slicing or ml_model is not None or force_core not in (None, 0):
            raise ValueError("HCF allocation supports core 0, no slicing and no ML overrides")
        if forced_band is not None and forced_band not in self.engine_props["band_list"]:
            raise ValueError("Forced HCF band is not configured")
        started = time.perf_counter()
        self._initialize_request_statistics()
        self.sdn_props.was_routed = False
        self.sdn_props.block_reason = None
        self.sdn_props.is_protected = False
        self.sdn_props.backup_path = None
        self.sdn_props.number_of_transponders = 1
        self.spectrum_obj.spectrum_props.is_free = False
        paths, route_time = self._setup_routing(force_route_matrix, force_mod_format)
        rate = parse_bandwidth(self.sdn_props.bandwidth)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("HCF request bitrate must be positive and finite")
        allowed = None if force_mod_format is None else ([force_mod_format] if isinstance(force_mod_format, str) else force_mod_format)
        eligible = [m for m in modes if m.net_rate_gbps >= rate and (allowed is None or m.name in allowed)]
        diagnostics = {"free_channels": 0, "qot_feasible_channels": 0, "power_rejections": 0}
        self.hcf_last_result = diagnostics

        def valid_paths():
            for path_index, path in enumerate(paths):
                if path is False:
                    continue
                if (
                    len(path) < 2 or len(set(path)) != len(path)
                    or path[0] != self.sdn_props.source or path[-1] != self.sdn_props.destination
                ):
                    raise ValueError("HCF paths must be simple and match the request endpoints")
                yield path_index, path

        search_args = {
            "power_used": MappingProxyType(self.hcf_power_used), "get_probe_paths": self._hcf_paths,
            "probe_modes": modes, "forced_index": forced_index, "forced_band": forced_band,
        }
        ranked = self.engine_props.get("hcf_selection_policy") in ("first_channel_ranked", "power_spectrum")
        if ranked:
            results = self.spectrum_obj.rank_hcf_first_channels(list(valid_paths()), eligible, powers, **search_args)
            for _, result in results:
                for key in ("free_channels", "qot_feasible_channels", "power_rejections"):
                    diagnostics[key] += getattr(result, key)
            diagnostics["ranked_candidates"] = sum(result.candidate is not None for _, result in results)
        else:
            # Keep existing policies lazy: do not evaluate paths after acceptance.
            results = (
                (index, self.spectrum_obj.get_spectrum_hcf(path, eligible, powers, **search_args))
                for index, path in valid_paths()
            )
        stale_reason = None
        for rank, (path_index, result) in enumerate(results, 1):
            if not ranked:
                for key in ("free_channels", "qot_feasible_channels", "power_rejections"):
                    diagnostics[key] += getattr(result, key)
            best = result.candidate
            if best is not None:
                if ranked:
                    # Revalidate resources only, never established-channel GSNR.
                    # Normally the first rank succeeds in the sequential event loop.
                    if best.channel not in self.spectrum_obj.get_hcf_free_channels(best.path, best.band):
                        stale_reason = "congestion"
                        continue
                    if not self.spectrum_obj.hcf_power_fits(best, self.hcf_power_used):
                        stale_reason = "hcf_power"
                        continue
                self._commit_hcf_candidate(best, path_index, route_time, result.score)
                if ranked:
                    diagnostics["selected_rank"] = rank
                diagnostics.update(
                    {
                        "accepted": True, "score": result.score, "launch_power_w": best.power_w, "gsnr_db": best.gsnr_db,
                        "path": list(best.path), "band": best.band, "channel": best.channel, "mode": best.mode.name,
                        "requested_rate_gbps": rate, "carrier_rate_gbps": best.mode.net_rate_gbps,
                        "gsnr_margin_db": best.gsnr_db - best.mode.threshold_db - float(self.engine_props.get("hcf_margin_db", 0)),
                    }
                )
                break
        if not self.sdn_props.was_routed:
            reason = "xt_threshold"
            if not diagnostics["free_channels"]:
                reason = "congestion"
            elif diagnostics["qot_feasible_channels"]:
                reason = "hcf_power"
            if stale_reason is not None:
                reason = stale_reason
            if not eligible or not paths:
                reason = "distance"
            self.sdn_props.block_reason = reason
            diagnostics.update({"accepted": False, "block_reason": reason})
        diagnostics["allocation_time_seconds"] = time.perf_counter() - started

    def _update_throughput(self) -> None:
        """
        Update throughput statistics for the current request.

        Should be called once per request release, before releasing individual lightpaths.
        This prevents overcounting throughput when multiple lightpaths are used for a
        single groomed request.
        """
        try:
            if (
                self.sdn_props.depart is None
                or self.sdn_props.arrive is None
                or self.sdn_props.bandwidth is None
                or self.sdn_props.path_list is None
                or self.sdn_props.network_spectrum_dict is None
            ):
                logger.warning("Missing data for throughput calculation")
                return

            # Permanent designs report carried Gbps, dynamic runs report Gbps-seconds.
            duration = (
                self.sdn_props.depart - self.sdn_props.arrive if self.engine_props.get("traffic_type", "dynamic") == "dynamic" else 1.0
            )
            bandwidth = parse_bandwidth(self.sdn_props.bandwidth)  # Gbps
            data_transferred = bandwidth * duration  # Gbps-seconds for dynamic traffic

            for source, dest in zip(self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False):
                self.sdn_props.network_spectrum_dict[(source, dest)]["throughput"] += data_transferred
                self.sdn_props.network_spectrum_dict[(dest, source)]["throughput"] += data_transferred
        except (TypeError, ValueError) as e:
            logger.warning("Throughput update skipped: %s", e)

    def release(self, lightpath_id: int | None = None, slicing_flag: bool = False, skip_validation: bool = False) -> None:
        """
        Remove a previously allocated request from the network.

        :param lightpath_id: Specific lightpath ID to release (for grooming)
        :type lightpath_id: int | None
        :param slicing_flag: If True, only release spectrum, not transponders
        :type slicing_flag: bool
        """
        if lightpath_id is None:
            raise ValueError("Lightpath ID is none")
        if self.sdn_props.path_list is None:
            raise ValueError("Path list is not initialized")

        for source, dest in zip(self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False):
            for band in self.engine_props["band_list"]:
                for core_num in range(self.engine_props["cores_per_link"]):
                    if self.sdn_props.network_spectrum_dict is None:
                        continue
                    core_array = self.sdn_props.network_spectrum_dict[(source, dest)]["cores_matrix"][band][core_num]

                    request_id_indices = np.where(core_array == lightpath_id)
                    guard_band_indices = np.where(core_array == (lightpath_id * -1))

                    for request_index in request_id_indices[0]:
                        self.sdn_props.network_spectrum_dict[(source, dest)]["cores_matrix"][band][core_num][request_index] = 0
                        self.sdn_props.network_spectrum_dict[(dest, source)]["cores_matrix"][band][core_num][request_index] = 0

                    for guard_band_index in guard_band_indices[0]:
                        self.sdn_props.network_spectrum_dict[(source, dest)]["cores_matrix"][band][core_num][guard_band_index] = 0
                        self.sdn_props.network_spectrum_dict[(dest, source)]["cores_matrix"][band][core_num][guard_band_index] = 0

        # Handle grooming-specific cleanup and dynamic slicing bandwidth tracking
        if lightpath_id is not None:
            self._release_lightpath_resources(lightpath_id, skip_validation=skip_validation)

    def _release_lightpath_resources(self, lightpath_id: int, skip_validation: bool = False) -> None:
        """
        Release transponders and update lightpath status dict.

        :param lightpath_id: ID of lightpath to release
        :type lightpath_id: int
        :param skip_validation: Skip validation checks (for rollback scenarios)
        :type skip_validation: bool
        """
        # Early return only if path_list or lightpath_status_dict are None
        # (transponder_usage_dict is optional and only needed for transponder tracking)
        if self.sdn_props.path_list is None or self.sdn_props.lightpath_status_dict is None:
            return

        # Always update transponders
        # for node in [self.sdn_props.source, self.sdn_props.destination]:
        #     if node not in self.sdn_props.transponder_usage_dict:
        #         logger.warning("Node %s not in transponder usage dict", node)
        #         continue
        #     self.sdn_props.transponder_usage_dict[node]["available_transponder"] += 1
        # Update transponders if tracking is enabled and dict is available
        if self.engine_props.get("transponder_usage_per_node", None) and self.sdn_props.transponder_usage_dict is not None:
            for node in [self.sdn_props.source, self.sdn_props.destination]:
                if node not in self.sdn_props.transponder_usage_dict:
                    raise KeyError(f"Node '{node}' not found in transponder usage dictionary.")
                self.sdn_props.transponder_usage_dict[node]["available_transponder"] += 1

        light_id = tuple(sorted([self.sdn_props.path_list[0], self.sdn_props.path_list[-1]]))

        # Check if light_id exists
        if light_id not in self.sdn_props.lightpath_status_dict:
            return

        # Handle lightpath status dict
        if light_id in self.sdn_props.lightpath_status_dict and lightpath_id in self.sdn_props.lightpath_status_dict[light_id]:
            # Calculate bandwidth utilization stats
            try:
                if self.sdn_props.lp_bw_utilization_dict is None:
                    return

                lp_status = self.sdn_props.lightpath_status_dict[light_id][lightpath_id]

                average_bw_usage = 0.0
                # Note: average_bandwidth_usage may not exist yet
                # Skip if not available
                try:
                    from fusion.utils.network import (  # type: ignore[attr-defined]
                        average_bandwidth_usage,
                    )

                    if self.sdn_props.depart is not None:
                        average_bw_usage = average_bandwidth_usage(
                            bw_dict=lp_status["time_bw_usage"],
                            departure_time=self.sdn_props.depart,
                        )
                except (ImportError, AttributeError):
                    pass

                # Only record utilization for normal releases, not SNR rollbacks
                # (rolled-back lightpaths never served traffic, so no utilization stats)
                if not skip_validation:
                    self.sdn_props.lp_bw_utilization_dict.update(
                        {
                            lightpath_id: {
                                "band": lp_status["band"],
                                "core": lp_status["core"],
                                "bit_rate": lp_status["lightpath_bandwidth"],
                                "utilization": average_bw_usage,
                            }
                        }
                    )
            except (TypeError, ValueError, KeyError) as e:
                logger.warning("Average BW update skipped: %s", e)

            # Grooming validation - ensure no active requests (skip during rollback)
            if (
                not skip_validation
                and self.sdn_props.lightpath_status_dict[light_id][lightpath_id]["requests_dict"]
                and self.engine_props["is_grooming_enabled"]
            ):
                raise ValueError(f"Lightpath {lightpath_id} still has active requests")

            # Remove from status dict
            self.sdn_props.lightpath_status_dict[light_id].pop(lightpath_id)

    def _allocate_guard_band(
        self,
        band: str,
        core_matrix: dict[str, Any],
        reverse_core_matrix: dict[str, Any],
        core_num: int,
        end_slot: int,
        lightpath_id: int,
    ) -> None:
        """
        Allocate guard band slots for spectrum isolation.

        :param band: Spectral band identifier
        :type band: str
        :param core_matrix: Core matrix for forward direction
        :type core_matrix: list[Any]
        :param reverse_core_matrix: Core matrix for reverse direction
        :type reverse_core_matrix: list[Any]
        :param core_num: Core number to allocate on
        :type core_num: int
        :param end_slot: End slot position for guard band
        :type end_slot: int
        :param lightpath_id: Lightpath ID to use for guard band marking
        :type lightpath_id: int
        :raises BufferError: If attempting to allocate already taken spectrum
        """
        if core_matrix[band][core_num][end_slot] != 0.0 or reverse_core_matrix[band][core_num][end_slot] != 0.0:
            raise BufferError("Attempted to allocate a taken spectrum.")

        # Use lightpath_id with negative sign for guard bands
        core_matrix[band][core_num][end_slot] = lightpath_id * -1
        reverse_core_matrix[band][core_num][end_slot] = lightpath_id * -1

    def _allocate_on_path(self, path: list[int]) -> None:
        """
        Allocate spectrum on a specific path (bidirectionally).

        This helper method contains the core allocation logic for a single path.
        It validates spectrum availability, updates usage counts, allocates slots
        on both forward and reverse links, and handles guard bands.

        :param path: List of node IDs representing the path
        :type path: list[int]
        :raises BufferError: If attempting to allocate already taken spectrum
        :raises ValueError: If no spectrum is detected during allocation
        """
        # Get allocation parameters from spectrum_props
        start_slot = self.spectrum_obj.spectrum_props.start_slot
        end_slot = self.spectrum_obj.spectrum_props.end_slot
        core_num = self.spectrum_obj.spectrum_props.core_number
        band = self.spectrum_obj.spectrum_props.current_band
        lightpath_id = self.spectrum_obj.spectrum_props.lightpath_id

        # Lightpath ID must always be set before allocation
        if lightpath_id is None:
            raise ValueError(
                "lightpath_id must be set in spectrum_props before calling allocate(). "
                "This is a programming error - ensure get_lightpath_id() is called "
                "and assigned to spectrum_props.lightpath_id before allocation."
            )

        # Validate all required parameters are present
        if start_slot is None or end_slot is None or core_num is None or band is None:
            raise ValueError("Missing required spectrum allocation parameters")
        if self.sdn_props.network_spectrum_dict is None:
            raise ValueError("Network spectrum dictionary is None")

        logger.debug(f"Attempting allocation on path {path}: slots [{start_slot}:{end_slot}], band={band}, core={core_num}")

        # Guard slot adjustment
        if self.engine_props["guard_slots"] != 0:
            end_slot = end_slot - 1
        else:
            end_slot = end_slot + 1

        # Allocate on each link in the path
        for link_tuple in zip(path, path[1:], strict=False):
            link_dict = self.sdn_props.network_spectrum_dict[(link_tuple[0], link_tuple[1])]
            reverse_link_dict = self.sdn_props.network_spectrum_dict[(link_tuple[1], link_tuple[0])]

            # Validate spectrum is free on both directions
            spectrum_slots_set = set(link_dict["cores_matrix"][band][core_num][start_slot:end_slot])
            reverse_spectrum_slots_set = set(reverse_link_dict["cores_matrix"][band][core_num][start_slot:end_slot])

            if spectrum_slots_set == {} or reverse_spectrum_slots_set == {}:
                raise ValueError("Nothing detected on the spectrum when allocating.")

            if spectrum_slots_set != {0.0} or reverse_spectrum_slots_set != {0.0}:
                logger.error(
                    f"Spectrum already taken on link {link_tuple}: "
                    f"forward={spectrum_slots_set}, reverse={reverse_spectrum_slots_set}. "
                    f"Path: {path}, slots [{start_slot}:{end_slot}], band={band}, core={core_num}"
                )
                raise BufferError("Attempted to allocate a taken spectrum.")

            # Update usage counts
            if link_tuple in self.sdn_props.network_spectrum_dict:
                self.sdn_props.network_spectrum_dict[link_tuple]["usage_count"] += 1
            self.sdn_props.network_spectrum_dict[(link_tuple[1], link_tuple[0])]["usage_count"] += 1

            # Allocate spectrum on both directions
            core_matrix = link_dict["cores_matrix"]
            reverse_core_matrix = reverse_link_dict["cores_matrix"]

            # Use lightpath_id instead of request_id
            core_matrix[band][core_num][start_slot:end_slot] = lightpath_id
            reverse_core_matrix[band][core_num][start_slot:end_slot] = lightpath_id

            # Handle guard bands
            if self.engine_props["guard_slots"]:
                self._allocate_guard_band(
                    band=band,
                    core_matrix=core_matrix,
                    reverse_core_matrix=reverse_core_matrix,
                    core_num=core_num,
                    end_slot=end_slot,
                    lightpath_id=lightpath_id,
                )

    def allocate(self) -> None:
        """
        Allocate spectrum resources for a network request.

        Assigns spectrum slots to the current request across all links in the path,
        including guard bands if configured. For 1+1 protected requests, allocates
        on both primary and backup paths.

        :raises BufferError: If attempting to allocate already taken spectrum
        :raises ValueError: If no spectrum is detected during allocation
        """
        if (
            self.spectrum_obj.spectrum_props.start_slot is None
            or self.spectrum_obj.spectrum_props.end_slot is None
            or self.spectrum_obj.spectrum_props.core_number is None
            or self.spectrum_obj.spectrum_props.current_band is None
            or self.sdn_props.path_list is None
            or self.sdn_props.network_spectrum_dict is None
        ):
            raise ValueError("Missing required spectrum or path information")

        # Allocate on primary path
        self._allocate_on_path(self.sdn_props.path_list)

        # If protected, also allocate on backup path
        backup_path = self.spectrum_obj.spectrum_props.backup_path
        if backup_path is not None:
            logger.debug(f"1+1 protection: Allocating spectrum on backup path {backup_path}")
            self._allocate_on_path(backup_path)

        # Track allocated request for failure handling
        self._track_allocated_request()

    def _track_allocated_request(self) -> None:
        """
        Track an allocated request for failure handling.

        Records request information including paths, protection status, and timing
        to enable proper handling when failures occur.
        """
        if self.sdn_props.request_id is None:
            return

        # Get backup path if this is a protected request
        backup_path = self.spectrum_obj.spectrum_props.backup_path

        # Store request information
        self.sdn_props.allocated_requests[self.sdn_props.request_id] = {
            "request_id": self.sdn_props.request_id,
            "primary_path": self.sdn_props.path_list.copy() if self.sdn_props.path_list else None,
            "backup_path": backup_path.copy() if backup_path else None,
            "is_protected": self.sdn_props.is_protected,
            "active_path": self.sdn_props.active_path,
            "arrive_time": self.sdn_props.arrive,
            "depart_time": self.sdn_props.depart,
            "bandwidth": self.sdn_props.bandwidth,
            "source": self.sdn_props.source,
            "destination": self.sdn_props.destination,
            "start_slot": self.spectrum_obj.spectrum_props.start_slot,
            "end_slot": self.spectrum_obj.spectrum_props.end_slot,
            "core_number": self.spectrum_obj.spectrum_props.core_number,
            "current_band": self.spectrum_obj.spectrum_props.current_band,
        }

        logger.debug(
            f"Tracked allocated request {self.sdn_props.request_id}: "
            f"protected={self.sdn_props.is_protected}, "
            f"primary={self.sdn_props.path_list}, backup={backup_path}"
        )

    def _update_request_statistics(self, bandwidth: float | None) -> None:
        """
        Update request statistics with allocation results.

        :param bandwidth: Allocated bandwidth for the request
        :type bandwidth: str
        """
        if bandwidth is not None:
            self.sdn_props.bandwidth_list.append(bandwidth)
        for stat_key in self.sdn_props.stat_key_list:
            # Skip remaining_bw - it's tracked separately for grooming
            if stat_key == "remaining_bw":
                continue

            # Map stat_key to corresponding spectrum_props attribute
            spectrum_key = stat_key.split("_", maxsplit=1)[0]
            if spectrum_key == "crosstalk":
                spectrum_key = "crosstalk_cost"
            elif spectrum_key == "snr":
                spectrum_key = "crosstalk_cost"  # SNR cost is same as crosstalk_cost
            elif spectrum_key == "xt":
                spectrum_key = "crosstalk_cost"  # XT cost is same as crosstalk_cost
            elif spectrum_key == "core":
                spectrum_key = "core_number"
            elif spectrum_key == "band":
                spectrum_key = "current_band"
            elif spectrum_key == "start":
                spectrum_key = "start_slot"
            elif spectrum_key == "end":
                spectrum_key = "end_slot"
            elif spectrum_key == "modulation":
                spectrum_key = "modulation"
            elif stat_key == "lightpath_id_list":
                spectrum_key = "lightpath_id"
            elif stat_key == "lightpath_bandwidth_list":
                spectrum_key = "lightpath_bandwidth"

            self.sdn_props.update_params(
                key=stat_key,
                spectrum_key=spectrum_key,
                spectrum_obj=self.spectrum_obj.spectrum_props,
            )

    # Backward compatibility alias for tests
    def _update_req_stats(self, bandwidth: float | None = None, remaining: str | None = None) -> None:
        """Legacy method name for _update_request_statistics."""
        # Note: remaining parameter is accepted for compatibility but not currently used
        self._update_request_statistics(bandwidth)

    def _update_grooming_stats(self) -> None:
        """Update grooming statistics after request processing."""
        if not hasattr(self, "stats") or self.stats is None:
            return

        if not hasattr(self.stats, "grooming_stats") or self.stats.grooming_stats is None:
            return

        # Determine grooming outcome
        was_groomed = getattr(self.sdn_props, "was_groomed", False) or False
        was_partially_groomed = getattr(self.sdn_props, "was_partially_groomed", False)

        # Get bandwidth
        bandwidth = float(self.sdn_props.bandwidth) if self.sdn_props.bandwidth else 0.0

        # Count new lightpaths established
        was_new_lps = getattr(self.sdn_props, "was_new_lp_established", [])
        new_lightpaths = len(was_new_lps) if isinstance(was_new_lps, list) else 0

        self.stats.grooming_stats.update_grooming_outcome(
            was_groomed=was_groomed,
            was_partially_groomed=was_partially_groomed,
            bandwidth=bandwidth,
            new_lightpaths=new_lightpaths,
        )

    def _handle_slicing_request(
        self,
        path_list: list[Any],
        path_index: int,
        forced_segments: int,
        force_slicing: bool,
    ) -> bool:
        """
        Handle slicing request using the dedicated slicing manager.

        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param path_index: Index of the current path being processed
        :type path_index: int
        :param forced_segments: Number of segments to force (-1 for auto)
        :type forced_segments: int
        :param force_slicing: Whether slicing is forced
        :type force_slicing: bool
        :return: True if slicing was successful
        :rtype: bool
        """
        if self.engine_props["dynamic_lps"]:
            return self.slicing_manager.handle_dynamic_slicing_direct(
                path_list=path_list,
                path_index=path_index,
                forced_segments=forced_segments,
                sdn_controller=self,
            )
        return self.slicing_manager.handle_static_slicing_direct(path_list=path_list, forced_segments=forced_segments, sdn_controller=self)

    def _check_snr_after_allocation(self, lightpath_id: int) -> bool:
        """
        Recheck SNR after spectrum allocation (for grooming).

        :param lightpath_id: ID of newly allocated lightpath
        :type lightpath_id: int
        :return: True if SNR is acceptable, False otherwise
        :rtype: bool
        """
        if not self.engine_props.get("snr_recheck", False):
            return True

        # Build lightpath info for SNR recheck
        new_lp_info = {
            "id": lightpath_id,
            "path": self.sdn_props.path_list,
            "spectrum": (
                self.spectrum_obj.spectrum_props.start_slot,
                self.spectrum_obj.spectrum_props.end_slot,
            ),
            "core": self.spectrum_obj.spectrum_props.core_number,
            "band": self.spectrum_obj.spectrum_props.current_band,
            "mod_format": self.spectrum_obj.spectrum_props.modulation,
        }

        # Create SNR checker instance
        snr_checker = SnrMeasurements(
            engine_props_dict=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=copy.deepcopy(self.spectrum_obj.spectrum_props),
            route_props=self.route_obj.route_props,
        )

        # Perform SNR recheck
        recheck_enable, violations = snr_checker.snr_recheck_after_allocation(new_lp_info)

        if recheck_enable:
            return True

        # SNR recheck failed - rollback the allocation
        logger.warning(f"SNR recheck failed for lightpath {lightpath_id} - rolling back allocation. Violations: {violations}")

        # Release the lightpath from the network (skip validation for rollback)
        self.release(lightpath_id=lightpath_id, slicing_flag=True, skip_validation=True)

        # Remove from tracking lists
        if lightpath_id in self.sdn_props.lightpath_id_list:
            idx = self.sdn_props.lightpath_id_list.index(lightpath_id)

            # Only restore remaining bandwidth for sliced/partial requests (v5 behavior)
            # For non-sliced requests, remaining_bw should stay None/0 since no bandwidth was served
            if self.sdn_props.is_sliced or self.sdn_props.was_partially_groomed:
                bw_to_restore = parse_bandwidth(float(self.sdn_props.bandwidth_list[idx]))
                if self.sdn_props.remaining_bw is None or self.sdn_props.remaining_bw == 0:
                    self.sdn_props.remaining_bw = bw_to_restore
                else:
                    # Handle remaining_bw being str/float type
                    current_remaining = parse_bandwidth(float(self.sdn_props.remaining_bw))
                    self.sdn_props.remaining_bw = current_remaining + bw_to_restore

            # Remove from all tracking lists (with safety check for list length)
            tracking_lists: list[list] = [
                self.sdn_props.lightpath_id_list,
                self.sdn_props.lightpath_bandwidth_list,
                self.sdn_props.start_slot_list,
                self.sdn_props.band_list,
                self.sdn_props.core_list,
                self.sdn_props.end_slot_list,
                self.sdn_props.xt_list,
                self.sdn_props.crosstalk_list,
                self.sdn_props.snr_list,
                self.sdn_props.bandwidth_list,
                self.sdn_props.modulation_list,
            ]
            for tracking_list in tracking_lists:
                if idx < len(tracking_list):
                    tracking_list.pop(idx)

        # Remove from new lightpath list
        if lightpath_id in self.sdn_props.was_new_lp_established:
            self.sdn_props.was_new_lp_established.remove(lightpath_id)

        # Mark request as blocked
        self.sdn_props.was_routed = False
        self.sdn_props.block_reason = "snr_recheck_failed"

        return False

    def _handle_congestion_with_grooming(self, remaining_bw: int) -> None:
        """
        Handle allocation failure with grooming rollback.

        If partial grooming occurred but remaining bandwidth cannot be allocated,
        rollback the newly created lightpaths but keep the groomed portion.

        :param remaining_bw: Remaining bandwidth that could not be allocated
        :type remaining_bw: int
        """
        if self.sdn_props.bandwidth is not None and remaining_bw != parse_bandwidth(self.sdn_props.bandwidth):
            # Rollback newly established lightpaths (skip_validation since never served traffic)
            was_new_lps = getattr(self.sdn_props, "was_new_lp_established", [])
            if isinstance(was_new_lps, list):
                for lpid in list(was_new_lps):
                    self.release(lightpath_id=lpid, slicing_flag=True, skip_validation=True)
                    was_new_lps.remove(lpid)

                    # Remove from tracking lists
                    lp_idx = self.sdn_props.lightpath_id_list.index(lpid)
                    self.sdn_props.lightpath_id_list.pop(lp_idx)
                    self.sdn_props.lightpath_bandwidth_list.pop(lp_idx)
                    self.sdn_props.start_slot_list.pop(lp_idx)
                    self.sdn_props.band_list.pop(lp_idx)
                    self.sdn_props.core_list.pop(lp_idx)
                    self.sdn_props.end_slot_list.pop(lp_idx)
                    self.sdn_props.xt_list.pop(lp_idx)
                    self.sdn_props.crosstalk_list.pop(lp_idx)
                    self.sdn_props.snr_list.pop(lp_idx)
                    self.sdn_props.bandwidth_list.pop(lp_idx)
                    self.sdn_props.modulation_list.pop(lp_idx)

        self.sdn_props.number_of_transponders = 1
        self.sdn_props.is_sliced = False
        self.sdn_props.was_partially_routed = False

        if getattr(self.sdn_props, "was_partially_groomed", False):
            if self.sdn_props.bandwidth is not None:
                self.sdn_props.remaining_bw = parse_bandwidth(self.sdn_props.bandwidth) - sum(
                    map(parse_bandwidth, self.sdn_props.bandwidth_list)
                )
        else:
            if self.sdn_props.bandwidth is not None:
                self.sdn_props.remaining_bw = parse_bandwidth(self.sdn_props.bandwidth)

        self.sdn_props.was_new_lp_established = []

    def _handle_congestion(self, remaining_bandwidth: int | None = None, remaining_bw: int | None = None) -> None:
        """
        Handle allocation failure due to network congestion.

        :param remaining_bandwidth: Remaining bandwidth that could not be allocated
        :type remaining_bandwidth: int
        :param remaining_bw: Legacy parameter name for remaining_bandwidth
        :type remaining_bw: int
        """
        # Handle backward compatibility
        if remaining_bw is not None:
            remaining_bandwidth = remaining_bw

        if remaining_bandwidth is None:
            raise ValueError("Must provide remaining_bandwidth")
        self.sdn_props.was_routed = False
        self.sdn_props.block_reason = "congestion"
        # Reset to 0 (not 1) - when congestion happens, no transponders are committed
        # Previously = 1 caused over-counting when slicing recovered from SNR recheck failures
        self.sdn_props.number_of_transponders = 0

        if self.sdn_props.bandwidth is not None and remaining_bandwidth != parse_bandwidth(self.sdn_props.bandwidth):
            # Release all newly established lightpaths for this request (skip_validation since never served traffic)
            was_new_lps = getattr(self.sdn_props, "was_new_lp_established", [])
            if isinstance(was_new_lps, list) and self.sdn_props.path_list:
                for lpid in list(was_new_lps):
                    # Remove current request from lightpath's requests_dict before releasing
                    light_id = tuple(sorted([self.sdn_props.path_list[0], self.sdn_props.path_list[-1]]))
                    if (
                        self.sdn_props.lightpath_status_dict is not None
                        and light_id in self.sdn_props.lightpath_status_dict
                        and lpid in self.sdn_props.lightpath_status_dict[light_id]
                    ):
                        lp_info = self.sdn_props.lightpath_status_dict[light_id][lpid]
                        if self.sdn_props.request_id in lp_info["requests_dict"]:
                            # Restore remaining bandwidth before removing request
                            allocated_bw = lp_info["requests_dict"][self.sdn_props.request_id]
                            lp_info["remaining_bandwidth"] += allocated_bw
                            lp_info["requests_dict"].pop(self.sdn_props.request_id)

                    self.release(lightpath_id=lpid, slicing_flag=True, skip_validation=True)
                    was_new_lps.remove(lpid)

                    # Remove from tracking lists
                    if lpid in self.sdn_props.lightpath_id_list:
                        lp_idx = self.sdn_props.lightpath_id_list.index(lpid)
                        self.sdn_props.lightpath_id_list.pop(lp_idx)
                        self.sdn_props.lightpath_bandwidth_list.pop(lp_idx)
                        self.sdn_props.start_slot_list.pop(lp_idx)
                        self.sdn_props.band_list.pop(lp_idx)
                        self.sdn_props.core_list.pop(lp_idx)
                        self.sdn_props.end_slot_list.pop(lp_idx)
                        self.sdn_props.xt_list.pop(lp_idx)
                        self.sdn_props.crosstalk_list.pop(lp_idx)
                        self.sdn_props.snr_list.pop(lp_idx)
                        self.sdn_props.bandwidth_list.pop(lp_idx)
                        self.sdn_props.modulation_list.pop(lp_idx)

        self.sdn_props.is_sliced = False
        self.sdn_props.was_partially_routed = False

        # Reset remaining_bw to match v5 behavior
        if self.sdn_props.bandwidth is not None:
            if self.sdn_props.was_partially_groomed:
                self.sdn_props.remaining_bw = parse_bandwidth(self.sdn_props.bandwidth) - sum(
                    map(parse_bandwidth, self.sdn_props.bandwidth_list)
                )
            else:
                self.sdn_props.remaining_bw = parse_bandwidth(self.sdn_props.bandwidth)

        self.sdn_props.was_new_lp_established = []

    def _initialize_request_statistics(self) -> None:
        """Initialize request statistics for a new request."""
        self.sdn_props.bandwidth_list = []
        self.sdn_props.reset_params()

    def _setup_routing(self, force_route_matrix: list[Any] | None, force_mod_format: str | list[str] | None) -> tuple[list[Any], float]:
        """
        Setup routing for the request.

        :param force_route_matrix: Optional forced routing matrix
        :type force_route_matrix: list[Any] | None
        :param force_mod_format: Optional forced modulation format (single or list)
        :type force_mod_format: str | list[str] | None
        :return: Tuple of (route_matrix, route_time)
        :rtype: tuple[list[Any], float]
        """
        start_time = time.time()
        if force_route_matrix is None:
            self.route_obj.get_route()
            route_matrix = self.route_obj.route_props.paths_matrix
        else:
            route_matrix = force_route_matrix
            if force_mod_format:
                # Handle both string and list formats for force_mod_format
                formats_matrix: list[list[str | bool]]
                if isinstance(force_mod_format, list):
                    formats_matrix = [force_mod_format]  # type: ignore[list-item]
                else:
                    formats_matrix = [[force_mod_format]]
            else:
                formats_matrix = [[]]
            self.route_obj.route_props.modulation_formats_matrix = formats_matrix

            # For partially groomed requests, use path_weight from groomed lightpath
            # This ensures new lightpaths on the same path get the correct weight
            if getattr(self.sdn_props, "was_partially_groomed", False) and self.sdn_props.path_weight is not None:
                self.route_obj.route_props.weights_list = [self.sdn_props.path_weight]
            else:
                self.route_obj.route_props.weights_list = [0.0]
        route_time = time.time() - start_time
        return route_matrix, route_time

    def _get_ml_prediction(self, ml_model: Any | None, request_dict: dict[str, Any]) -> float:
        """
        Get ML model prediction for forced segments.

        :param ml_model: Optional machine learning model
        :type ml_model: Any | None
        :param request_dict: Request dictionary
        :type request_dict: dict[str, Any]
        :return: Forced segments prediction (-1 for auto)
        :rtype: float
        """
        if ml_model is not None:
            input_df = get_ml_obs(
                request_dict=request_dict,
                engine_properties=self.engine_props,
                sdn_properties=self.sdn_props,
            )
            prediction = ml_model.predict(input_df)[0]
            return float(prediction)
        return -1.0

    def _process_single_path(
        self,
        path_list: list[Any],
        path_index: int,
        mod_format_list: list[str],
        forced_segments: float,
        force_slicing: bool,
        segment_slicing: bool,
        forced_index: int | None,
        force_core: int | None,
        forced_band: str | None,
        backup_mod_format_list: list[str] | None = None,
    ) -> bool:
        """
        Process allocation for a single path.

        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param path_index: Index of the current path
        :type path_index: int
        :param mod_format_list: List of modulation formats for primary path
        :type mod_format_list: list[str]
        :param forced_segments: Number of forced segments
        :type forced_segments: float
        :param force_slicing: Whether to force slicing
        :type force_slicing: bool
        :param segment_slicing: Whether segment slicing is enabled
        :type segment_slicing: bool
        :param forced_index: Optional forced spectrum index
        :type forced_index: int | None
        :param force_core: Optional forced core number
        :type force_core: int | None
        :param forced_band: Optional forced band
        :type forced_band: str | None
        :param backup_mod_format_list: Optional modulation formats for backup path (1+1)
        :type backup_mod_format_list: list[str] | None
        :return: True if path processing was successful
        :rtype: bool
        """
        self.sdn_props.path_list = path_list
        self.sdn_props.path_index = path_index

        # Handle slicing scenarios
        # Match v5 behavior: set force_slicing=True when segment_slicing is enabled
        if segment_slicing or force_slicing or forced_segments > 1:
            force_slicing = True
            success = self._handle_slicing_request(path_list, path_index, int(forced_segments), force_slicing)
            if not success:
                self.sdn_props.number_of_transponders = 1
                return False
        else:
            # Handle standard allocation
            self.spectrum_obj.spectrum_props.forced_index = forced_index
            self.spectrum_obj.spectrum_props.forced_core = force_core
            self.spectrum_obj.spectrum_props.path_list = path_list
            self.spectrum_obj.spectrum_props.forced_band = forced_band

            self.spectrum_obj.get_spectrum(
                mod_format_list=mod_format_list,
                backup_mod_format_list=backup_mod_format_list,
            )

            if self.spectrum_obj.spectrum_props.is_free is not True:
                self.sdn_props.block_reason = "congestion"
                return False

            # Generate and assign unique lightpath ID for this allocation
            lp_id = self.sdn_props.get_lightpath_id()
            self.spectrum_obj.spectrum_props.lightpath_id = lp_id
            self.sdn_props.was_new_lp_established.append(lp_id)

            # Determine bandwidth for statistics and lightpath (handle partial grooming)
            allocate_bandwidth: float | None
            if self.sdn_props.was_partially_groomed:
                # For partial grooming, allocate_bandwidth is the actual bandwidth used for statistics
                allocate_bandwidth = float(self.sdn_props.remaining_bw) if self.sdn_props.remaining_bw else None
            else:
                allocate_bandwidth = self.sdn_props.bandwidth

            # Always use original request bandwidth for lightpath (matches v5 behavior)
            # This ensures lightpath is tracked in the correct bandwidth tier for modulation statistics
            lightpath_bandwidth = self.sdn_props.bandwidth

            # Set lightpath bandwidth only if not already set by slicing code
            if (
                not hasattr(self.spectrum_obj.spectrum_props, "lightpath_bandwidth")
                or self.spectrum_obj.spectrum_props.lightpath_bandwidth is None
            ):
                self.spectrum_obj.spectrum_props.lightpath_bandwidth = lightpath_bandwidth

            self._update_request_statistics(bandwidth=allocate_bandwidth)

            # Create lightpath entry in status dict for grooming/tracking
            self.spectrum_obj._update_lightpath_status()

        return True

    def _finalize_successful_allocation(
        self,
        path_index: int,
        route_time: float,
        force_slicing: bool,
        segment_slicing: bool,
    ) -> bool:
        """
        Finalize a successful allocation.

        :param path_index: Index of the successful path
        :type path_index: int
        :param route_time: Time taken for routing
        :type route_time: float
        :param force_slicing: Whether slicing was forced
        :type force_slicing: bool
        :param segment_slicing: Whether segment slicing was used
        :type segment_slicing: bool
        :return: True if finalization succeeded (including SNR recheck), False otherwise
        :rtype: bool
        """
        self.sdn_props.was_routed = True
        self.sdn_props.route_time = route_time
        self.sdn_props.path_weight = self.route_obj.route_props.weights_list[path_index]
        self.sdn_props.spectrum_object = self.spectrum_obj.spectrum_props

        if not segment_slicing and not force_slicing:
            # Set is_sliced=True for partial grooming (v5 behavior)
            if self.sdn_props.was_partially_groomed:
                self.sdn_props.is_sliced = True
                self.sdn_props.remaining_bw = 0  # Request fully served
            else:
                self.sdn_props.is_sliced = False
            self.allocate()

        # Check SNR after allocation for newly created lightpaths
        # For sliced requests, SNR recheck is done in the slicing loop (v5 behavior)
        # to prevent orphaned allocations. Only do finalize SNR recheck for non-sliced.
        if not segment_slicing and not force_slicing:
            if self.sdn_props.was_new_lp_established:
                for lp_id in self.sdn_props.was_new_lp_established:
                    if not self._check_snr_after_allocation(lp_id):
                        # SNR recheck failed - allocation was rolled back
                        return False

        # Update grooming statistics
        if self.engine_props.get("is_grooming_enabled", False):
            self._update_grooming_stats()

        return True

    def handle_event(
        self,
        request_dict: dict[str, Any],
        request_type: str,
        force_slicing: bool = False,
        force_route_matrix: list[Any] | None = None,
        forced_index: int | None = None,
        force_core: int | None = None,
        ml_model: Any | None = None,
        force_mod_format: str | list[str] | None = None,
        forced_band: str | None = None,
    ) -> None:
        """
        Handle any event that occurs in the simulation.

        Controls the main flow of request processing including routing, spectrum
        allocation, and various slicing strategies.

        :param request_dict: Request dictionary containing request parameters
        :type request_dict: dict[str, Any]
        :param request_type: Type of request ('arrival' or 'release')
        :type request_type: str
        :param force_slicing: Whether to force light path segment slicing
        :type force_slicing: bool
        :param force_route_matrix: Optional forced routing matrix
        :type force_route_matrix: list[Any] | None
        :param forced_index: Optional forced start index for spectrum allocation
        :type forced_index: int | None
        :param force_mod_format: Optional forced modulation format
        :type force_mod_format: str | None
        :param force_core: Optional forced core number
        :type force_core: int | None
        :param ml_model: Optional machine learning model for predictions
        :type ml_model: Any | None
        :param forced_band: Optional forced spectral band
        :type forced_band: str | None
        """
        # Handle release requests
        if request_type == "release":
            lightpath_id_list: list[int | None] = []
            if self.engine_props.get("is_grooming_enabled", False):
                groom_result = self.grooming_obj.handle_grooming(request_type)
                if isinstance(groom_result, list):
                    # Convert list[int] to list[int | None]
                    lightpath_id_list = list(groom_result)
            else:
                # Convert list[int] to list[int | None]
                lightpath_id_list = list(self.sdn_props.lightpath_id_list)

            # Update throughput once per request, then release each lightpath
            self._update_throughput()
            for lightpath_id in lightpath_id_list:
                self.release(lightpath_id=lightpath_id)
            return

        self._initialize_request_statistics()
        self.sdn_props.number_of_transponders = 1

        # Try grooming first if enabled
        if self.engine_props.get("is_grooming_enabled", False):
            # Set lightpath status dict for grooming object
            if hasattr(self.grooming_obj, "lightpath_status_dict"):
                self.grooming_obj.lightpath_status_dict = self.sdn_props.lightpath_status_dict

            groom_result = self.grooming_obj.handle_grooming(request_type)

            if groom_result:
                # Fully groomed - done!
                self._update_grooming_stats()
                return

            # Not groomed or partially groomed
            self.sdn_props.was_new_lp_established = []

            if getattr(self.sdn_props, "was_partially_groomed", False):
                # Force route on same path as groomed portion
                force_route_matrix = [self.sdn_props.path_list]

                # Get modulation formats for remaining bandwidth
                from fusion.utils.data import sort_nested_dict_values

                if self.sdn_props.modulation_formats_dict is not None:
                    mod_formats_dict = sort_nested_dict_values(
                        original_dict=self.sdn_props.modulation_formats_dict,
                        nested_key="max_length",
                    )
                    # Use all modulation formats sorted by max_length (v5 behavior)
                    force_mod_format = list(mod_formats_dict.keys())

        # Setup routing
        route_matrix, route_time = self._setup_routing(force_route_matrix, force_mod_format)

        # Get ML prediction if available
        forced_segments = self._get_ml_prediction(ml_model, request_dict)

        # Try allocation with different strategies
        segment_slicing = False
        while True:
            for path_index, path_list in enumerate(route_matrix):
                if path_list is not False:
                    # Check path feasibility if failures are active
                    if self.failure_manager and not self.failure_manager.is_path_feasible(path_list):
                        logger.debug(f"Path {path_list} (index {path_index}) is infeasible due to active failures")
                        continue  # Skip this path and try next one

                    # Check backup path feasibility for protected requests
                    # Get corresponding backup path from backup_paths_matrix if available
                    backup_path = None
                    if (
                        hasattr(self.route_obj.route_props, "backup_paths_matrix")
                        and len(self.route_obj.route_props.backup_paths_matrix) > path_index
                    ):
                        backup_path = self.route_obj.route_props.backup_paths_matrix[path_index]
                    elif self.sdn_props.backup_path is not None:
                        # Fallback to sdn_props.backup_path for backward compatibility
                        backup_path = self.sdn_props.backup_path

                    if self.failure_manager and backup_path is not None and not self.failure_manager.is_path_feasible(backup_path):
                        logger.debug(f"Backup path {backup_path} (for primary {path_list}) is infeasible due to active failures")
                        continue  # Skip this path pair and try next one

                    # Set backup path in sdn_props for spectrum assignment
                    backup_mod_format_list = None
                    if backup_path is not None:
                        self.sdn_props.backup_path = backup_path
                        self.sdn_props.is_protected = True

                        # Get backup modulation formats for 1+1 protection validation
                        if (
                            hasattr(
                                self.route_obj.route_props,
                                "backup_modulation_formats_matrix",
                            )
                            and len(self.route_obj.route_props.backup_modulation_formats_matrix) > path_index
                        ):
                            backup_mod_format_list = self.route_obj.route_props.backup_modulation_formats_matrix[path_index]

                    mod_format_list = self.route_obj.route_props.modulation_formats_matrix[path_index]

                    # Match v5: set force_slicing=True when segment_slicing or forced_segments > 1
                    if segment_slicing or force_slicing or forced_segments > 1:
                        force_slicing = True

                    # Set path_weight for this path attempt BEFORE processing
                    # This ensures any lightpaths created during this attempt (even if it fails
                    # but is accepted via partial serving) will store the correct path_weight
                    self.sdn_props.path_weight = self.route_obj.route_props.weights_list[path_index]

                    # Process the path
                    success = self._process_single_path(
                        path_list,
                        path_index,
                        mod_format_list,  # type: ignore[arg-type]
                        forced_segments,
                        force_slicing,
                        segment_slicing,
                        forced_index,
                        force_core,
                        forced_band,
                        backup_mod_format_list,  # type: ignore[arg-type]
                    )

                    if success:
                        # Try to finalize - includes SNR recheck
                        finalize_success = self._finalize_successful_allocation(path_index, route_time, force_slicing, segment_slicing)
                        if finalize_success:
                            return
                        # If finalize failed (SNR recheck), continue to next path

            # Try segment slicing if not already tried
            if self.engine_props["max_segments"] > 1 and self.sdn_props.bandwidth != "25" and not segment_slicing:
                segment_slicing = True
                continue

            # All paths exhausted
            self.sdn_props.block_reason = "distance"
            self.sdn_props.was_routed = False

            # FEATURE: Support partial serving (v5 behavior)
            # If can_partially_serve is enabled and SOME bandwidth was allocated, accept partial service
            if self.engine_props.get("can_partially_serve", False):
                # Check if any bandwidth was allocated (either through grooming or new lightpaths)
                has_allocated_bandwidth = (
                    len(self.sdn_props.lightpath_id_list) > 0 or len(getattr(self.sdn_props, "was_new_lp_established", [])) > 0
                )

                if has_allocated_bandwidth:
                    # Check if this is a partially groomed request OR the last path attempt
                    if (
                        getattr(self.sdn_props, "was_partially_groomed", False)
                        or self.sdn_props.path_index >= self.engine_props.get("k_paths", 1) - 1
                    ):
                        # Accept partial service - mark as successfully routed
                        self.sdn_props.was_routed = True
                        self.sdn_props.was_partially_routed = True
                        self.sdn_props.is_sliced = True

                        return  # Success!

            # CRITICAL FIX: Release groomed bandwidth if request was partially groomed but blocked
            if getattr(self.sdn_props, "was_partially_groomed", False):
                if self.sdn_props.lightpath_id_list and self.sdn_props.source is not None and self.sdn_props.destination is not None:
                    # Release groomed bandwidth from all lightpaths
                    light_id = tuple(sorted([self.sdn_props.source, self.sdn_props.destination]))
                    lp_status = self.sdn_props.lightpath_status_dict
                    for lp_id in self.sdn_props.lightpath_id_list[:]:
                        if lp_status is not None and light_id in lp_status:
                            if lp_id in lp_status[light_id]:
                                lp_info = lp_status[light_id][lp_id]
                                if self.sdn_props.request_id in lp_info["requests_dict"]:
                                    # Get allocated bandwidth and release it
                                    req_bw = lp_info["requests_dict"][self.sdn_props.request_id]
                                    lp_info["requests_dict"].pop(self.sdn_props.request_id)
                                    lp_info["remaining_bandwidth"] += req_bw

                    # Clear the lightpath tracking lists
                    self.sdn_props.lightpath_id_list = []
                    self.sdn_props.lightpath_bandwidth_list = []

            return

    # Backward compatibility methods for tests
    def _allocate_slicing(self, num_segments: int, mod_format: str, path_list: list[Any], bandwidth: str) -> None:
        """
        Backward compatibility wrapper for allocate_slicing method.

        :param num_segments: Number of segments to allocate
        :type num_segments: int
        :param mod_format: Modulation format to use
        :type mod_format: str
        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param bandwidth: Bandwidth requirement for each segment
        :type bandwidth: str
        """
        self.slicing_manager.allocate_slicing_direct(
            num_segments=num_segments,
            mod_format=mod_format,
            path_list=path_list,
            bandwidth=bandwidth,
            sdn_controller=self,
        )

    def _handle_dynamic_slicing(self, path_list: list[Any], path_index: int, forced_segments: int) -> None:
        """
        Backward compatibility wrapper for handle_dynamic_slicing method.

        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param path_index: Index of the current path being processed
        :type path_index: int
        :param forced_segments: Number of forced segments (unused)
        :type forced_segments: int
        """
        self.slicing_manager.handle_dynamic_slicing_direct(
            path_list=path_list,
            path_index=path_index,
            forced_segments=forced_segments,
            sdn_controller=self,
        )
