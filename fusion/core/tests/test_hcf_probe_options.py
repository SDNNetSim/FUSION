"""Compact probe options must reproduce candidate-based opportunity scoring."""

import copy
import math
from types import MethodType
from unittest.mock import patch

import numpy as np
import pytest

from fusion.core.hcf import HCFMode
from fusion.core.tests.test_hcf_controller import arrive, prepare_request
from fusion.core.tests.test_hcf_controller import controller as controller_fixture

controller = controller_fixture


def candidate_reference(self, modes, powers, power_used, get_probe_paths, virtual=None):
    """Original enumeration: full candidate objects and every amplifier demand."""
    value = 0.0
    for paths, rate, weight in self._hcf_probes(get_probe_paths):
        count = 0
        for path in paths:
            for band in self.engine_props_dict["band_list"]:
                for channel in self.get_hcf_free_channels(path, band, virtual):
                    choices = self.get_hcf_candidates(path, band, channel, modes, powers)
                    if any(c.mode.net_rate_gbps >= rate and self.hcf_power_fits(c, power_used, virtual) for c in choices):
                        count += 1
        value += weight * math.log1p(count)
    return value


def configure(ctrl, objective, multiband=False):
    props = ctrl.engine_props
    props.update(hcf_power_objective=objective, hcf_selection_policy="first_channel_ranked",
                 hcf_power_weight=0.5, hcf_amplifier_budget_dbm={"c": 38, "l": 38})
    props["hcf_modes"].append({"name": "16-QAM", "net_rate_gbps": 200, "baud_rate_gbaud": 64, "threshold_db": 13.24})
    props["hcf_probes"] = [
        {"source": "A", "destination": "C", "bandwidth": 100, "weight": 0.3},
        {"source": "B", "destination": "C", "bandwidth": 200, "weight": 0.7},
    ]
    # Different reverse-direction physical parameters must stay separate.
    props["topology_info"]["links"][9] = copy.deepcopy(props["topology_info"]["links"][1])
    props["topology_info"]["links"][9]["fiber"]["hcf"]["noise_figure"] = 6.0
    ctrl.sdn_props.network_spectrum_dict[("B", "A")]["link_num"] = 9
    if multiband:
        props.update(band_list=["c", "l"], l_band=3)
        for link in props["topology_info"]["links"].values():
            link["fiber"]["frequency_start_l"] = 185e12
        for link in ctrl.sdn_props.network_spectrum_dict.values():
            link["cores_matrix"]["l"] = np.zeros((1, 3))


@pytest.mark.parametrize("objective", ["discrete", "fixed", "max_gsnr", "minimum_required"])
@pytest.mark.parametrize("multiband", [False, True])
def test_compact_opportunity_matches_candidate_reference_with_live_and_virtual_resources(controller, objective, multiband):
    configure(controller, objective, multiband)
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    candidate = spectrum.get_hcf_candidates(["A", "B", "C"], "c", 1, modes, powers)[0]
    controller.sdn_props.network_spectrum_dict[("A", "C")]["cores_matrix"]["c"][0, 0] = 99
    budgets = {key: budget for key, _, budget in candidate.demands}
    forward = ("A", "B", 0, "c")
    reverse = ("B", "A", 0, "c")
    states = [
        {},
        {forward: budgets[forward]},
        {reverse: budgets[reverse] - 0.001},
        {("B", "C", 1, "c"): budgets[("B", "C", 1, "c")] - 0.001},
    ]
    network_before = copy.deepcopy(controller.sdn_props.network_spectrum_dict)
    for used in states:
        for virtual in (None, candidate):
            expected = candidate_reference(spectrum, modes, powers, used, controller._hcf_paths, virtual)
            assert spectrum._hcf_opportunity(modes, powers, used, controller._hcf_paths, virtual) == expected
    for edge, item in network_before.items():
        for band in controller.engine_props["band_list"]:
            np.testing.assert_array_equal(item["cores_matrix"][band],
                                          controller.sdn_props.network_spectrum_dict[edge]["cores_matrix"][band])
    assert not controller.hcf_allocations and not controller.hcf_power_used


def test_probe_scoring_does_not_construct_candidates_or_repeat_gsnr(controller):
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    snr = spectrum.snr_measurements
    with (
        patch.object(spectrum, "get_hcf_candidates", side_effect=AssertionError("No probe candidate construction")),
        patch.object(spectrum, "hcf_power_fits", side_effect=AssertionError("Use live directional minima")),
        patch.object(snr, "check_gsnr_hcf", wraps=snr.check_gsnr_hcf) as gsnr,
    ):
        first = spectrum._hcf_opportunity(modes, powers, {}, controller._hcf_paths)
        calls = gsnr.call_count
        assert calls > 0
        assert spectrum._hcf_opportunity(modes, powers, {}, controller._hcf_paths) == first
        assert gsnr.call_count == calls
    assert spectrum._hcf_transmission_cache.current_bytes <= spectrum._hcf_transmission_cache.max_bytes


@pytest.mark.parametrize("change", ["threshold", "rate", "baud", "rolloff", "margin", "trx", "grid", "fixed", "reverse_physics", "budget"])
def test_option_keys_and_live_budget_updates(controller, change):
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    first = spectrum.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)
    mode = modes[0]
    if change in ("threshold", "rate", "baud", "rolloff"):
        fields = {"name": mode.name, "net_rate_gbps": mode.net_rate_gbps, "baud_rate_gbaud": mode.baud_rate_gbaud,
                  "threshold_db": mode.threshold_db, "roll_off": mode.roll_off}
        key, value = {"threshold": ("threshold_db", 90), "rate": ("net_rate_gbps", 200),
                      "baud": ("baud_rate_gbaud", 64), "rolloff": ("roll_off", 0.8)}[change]
        fields[key] = value
        modes = [HCFMode(**fields)]
    elif change == "margin":
        controller.engine_props["hcf_margin_db"] = 90
    elif change == "trx":
        controller.engine_props["hcf_transceiver_snr_db"] = 2
    elif change == "grid":
        powers = [0.003]
    elif change == "fixed":
        controller.engine_props.update(hcf_power_objective="fixed", input_power=0.003)
    elif change == "reverse_physics":
        props = controller.engine_props
        props["topology_info"]["links"][9] = copy.deepcopy(props["topology_info"]["links"][1])
        props["topology_info"]["links"][9]["fiber"]["hcf"]["attenuation"] = 1.0
        controller.sdn_props.network_spectrum_dict[("B", "A")]["link_num"] = 9
    elif change == "budget":
        controller.engine_props["hcf_amplifier_budget_dbm"] = {"c": -20}
    updated = spectrum.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)
    spectrum.clear_hcf_transmission_cache()
    assert spectrum.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers) == updated
    assert updated != first


def test_headroom_matches_per_amplifier_checks_at_tolerance_boundary(controller):
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    candidate = spectrum.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)[0]
    key, power, budget = candidate.demands[0]
    for factor in (1.0, 1.0 + 0.5e-12, 1.0 + 2e-12):
        used = {key: budget - power / factor}
        forward, reverse = spectrum._hcf_probe_headroom(candidate.path, "c", used, {})
        compact = forward > 0 and reverse > 0 and power <= forward * (1 + 1e-12) and power <= reverse * (1 + 1e-12)
        assert compact == spectrum.hcf_power_fits(candidate, used)


def test_option_cache_eviction_and_iteration_reset(controller):
    spectrum = controller.spectrum_obj
    spectrum._hcf_transmission_cache.max_entries = 2
    modes, powers = controller._hcf_settings()
    initial = spectrum.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)
    for channel in (1, 2):
        spectrum.get_hcf_candidates(["A", "B", "C"], "c", channel, modes, powers)
    assert len(spectrum._hcf_transmission_cache) == 2
    assert spectrum.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers) == initial
    controller.reset_hcf_state()
    assert len(spectrum._hcf_transmission_cache) == spectrum._hcf_transmission_cache.current_bytes == 0


@pytest.mark.parametrize("objective", ["discrete", "fixed", "max_gsnr", "minimum_required"])
def test_arrivals_and_departures_preserve_original_probe_ranking(controller, objective):
    reference = controller_fixture.__wrapped__()
    reference.spectrum_obj._hcf_opportunity = MethodType(candidate_reference, reference.spectrum_obj)
    traces = []
    for ctrl in (controller, reference):
        configure(ctrl, objective, multiband=True)
        active = []
        trace = []
        for request_id in range(1, 13):
            prepare_request(ctrl, request_id=request_id)
            arrive(ctrl)
            decision = copy.deepcopy(ctrl.hcf_last_result)
            decision.pop("allocation_time_seconds")
            trace.append(decision)
            if ctrl.sdn_props.was_routed:
                active.append((request_id, ctrl.sdn_props.lightpath_id_list[0]))
            if request_id % 3 == 0 and active:
                old_request, lightpath = active.pop(0)
                prepare_request(ctrl, request_id=old_request)
                ctrl.sdn_props.lightpath_id_list = [lightpath]
                ctrl.handle_hcf_event({}, "release")
        traces.append(trace)
    assert traces[0] == traces[1]
    assert controller.hcf_power_used == reference.hcf_power_used

