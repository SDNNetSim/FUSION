"""Bounded physical reuse must preserve QoT, live resources and ranking."""

import copy
from unittest.mock import patch

import pytest

from fusion.core.hcf import HCFPhysicalCache
from fusion.core.tests.test_hcf_controller import arrive, prepare_request
from fusion.core.tests.test_hcf_controller import controller as controller_fixture
from fusion.core.tests.test_hcf_minimum_power import physical_args

controller = controller_fixture


def test_cache_lru_limit_and_cached_gain_rejection():
    cache = HCFPhysicalCache(max_entries=2)
    a, b, c = (cache.key(name) for name in ("a", "b", "c"))
    cache.put(a, (1.0, 2.0, 3.0))
    cache.put(b, None)
    assert cache.get(a) == (True, (1.0, 2.0, 3.0))
    cache.put(c, (4.0,))
    assert len(cache) == 2
    assert cache.get(b) == (False, None)
    assert cache.get(a)[0]
    cache.put(b, None)
    assert cache.get(b) == (True, None)
    cache.clear()
    assert len(cache) == cache.current_bytes == cache.hits == cache.misses == 0


def test_cache_byte_limit_oversized_keys_and_disabled_storage():
    cache = HCFPhysicalCache()
    cache.put(cache.key("a"), (1.0,))
    size = cache.current_bytes
    cache = HCFPhysicalCache(max_entries=100, max_bytes=2 * size)
    for name in ("a", "b", "c"):
        cache.put(cache.key(name), (1.0,))
        assert cache.current_bytes <= cache.max_bytes
    assert len(cache) == 2
    cache.put(cache.key("x" * 10000), (1.0,))
    assert len(cache) == 2 and cache.current_bytes <= cache.max_bytes
    disabled = HCFPhysicalCache(max_entries=0)
    disabled.put(disabled.key("a"), (1.0,))
    assert len(disabled) == disabled.current_bytes == 0


def test_fixed_gsnr_reuses_noise_but_checks_new_threshold_and_rate(controller):
    snr = controller.spectrum_obj.snr_measurements
    args = physical_args()
    with patch.object(snr, "_compute_hcf_noise_coefficients", wraps=snr._compute_hcf_noise_coefficients) as compute:
        first = snr.check_gsnr_hcf(**args, power_w=0.001)
        calls = compute.call_count
        assert calls > 0
        assert snr.check_gsnr_hcf(**args, power_w=0.001) == first
        assert compute.call_count == calls
        args.update(modulation="64-QAM", bit_rate_gbps=600, threshold_db=90)
        rejected = snr.check_gsnr_hcf(**args, power_w=0.001)
        assert rejected == (False, first[1], 600.0)
        assert compute.call_count == calls


@pytest.mark.parametrize("method", ["check_gsnr_hcf_optimized_power", "check_gsnr_hcf_minimum_power"])
def test_power_optimizers_reuse_path_coefficients(controller, method):
    snr = controller.spectrum_obj.snr_measurements
    check = getattr(snr, method)
    with patch.object(snr, "_compute_hcf_noise_coefficients", wraps=snr._compute_hcf_noise_coefficients) as compute:
        first = check(**physical_args())
        count = compute.call_count
        assert check(**physical_args()) == first
        assert count > 0 and compute.call_count == count
        rejected = check(**physical_args(threshold_db=90))
        assert not rejected[0]
        assert compute.call_count == count


@pytest.mark.parametrize("change", [
    "noise_figure", "dispersion", "non_linearity", "inter_modal_interference",
    "attenuation", "max_amplifier_gain", "length", "span_length", "frequency",
    "baud", "spacing", "comb", "power", "transceiver", "channel_profile", "mapping", "planck",
])
def test_changed_physical_inputs_cannot_hit_stale_gsnr(controller, change):
    snr = controller.spectrum_obj.snr_measurements
    props = controller.engine_props
    link = props["topology_info"]["links"][1]
    profile = link["fiber"]["hcf"]
    args = {**physical_args(), "power_w": 0.001}
    before = snr.check_gsnr_hcf(**args)
    if change in ("noise_figure", "dispersion", "non_linearity", "inter_modal_interference", "attenuation"):
        profile[change] *= 1.2
    elif change == "max_amplifier_gain":
        profile[change] = 0
    elif change in ("length", "span_length"):
        link[change] *= 0.7
    elif change == "frequency":
        for item in props["topology_info"]["links"].values():
            item["fiber"]["frequency_start_c"] += 1e12
    elif change == "baud":
        args["baud_rate_gbaud"] = 64
    elif change == "spacing":
        props["bw_per_slot"] = 100
    elif change == "comb":
        props.update(band_list=["c", "l"], l_band=3)
    elif change == "power":
        args["power_w"] = 0.002
    elif change == "transceiver":
        args["transceiver_snr_db"] = 20
    elif change == "channel_profile":
        profile["band_profiles"] = {"c": {"attenuation": [0.22, 0.11, 0.11]}}
    elif change == "mapping":
        props["topology_info"]["links"][9] = copy.deepcopy(link)
        props["topology_info"]["links"][9]["length"] = 90
        controller.sdn_props.network_spectrum_dict[("A", "B")]["link_num"] = 9
    elif change == "planck":
        snr.snr_props.planck_constant *= 2
    after = snr.check_gsnr_hcf(**args)
    snr._hcf_physical_cache = HCFPhysicalCache(max_entries=0)
    uncached = snr.check_gsnr_hcf(**args)
    assert after == uncached
    assert after[1] != before[1]


def test_cached_path_coefficients_observe_changed_link_and_band(controller):
    snr = controller.spectrum_obj.snr_measurements
    first = snr.check_gsnr_hcf_optimized_power(**physical_args())
    controller.engine_props["topology_info"]["links"][1]["length"] = 90
    changed = snr.check_gsnr_hcf_optimized_power(**physical_args())
    snr.clear_hcf_physical_cache()
    assert snr.check_gsnr_hcf_optimized_power(**physical_args()) == changed
    assert changed != first
    props = controller.engine_props
    props.update(band_list=["c", "l"], l_band=3)
    for link in props["topology_info"]["links"].values():
        link["fiber"]["frequency_start_l"] = 185e12
    l_result = snr.check_gsnr_hcf_optimized_power(**physical_args(band="l"))
    snr.clear_hcf_physical_cache()
    assert snr.check_gsnr_hcf_optimized_power(**physical_args(band="l")) == l_result


def test_legacy_implicit_power_settings_are_part_of_key(controller):
    snr = controller.spectrum_obj.snr_measurements
    first = snr.check_gsnr_hcf(**physical_args())
    controller.engine_props["input_power"] = 0.002
    second = snr.check_gsnr_hcf(**physical_args())
    assert first != second
    controller.engine_props["optimized_power"] = True
    optimized = snr.check_gsnr_hcf(**physical_args())
    snr.clear_hcf_physical_cache()
    assert snr.check_gsnr_hcf(**physical_args()) == optimized
    assert optimized != second


def test_iteration_reset_clears_physical_reuse_and_demands_stay_live(controller):
    spectrum = controller.spectrum_obj
    snr = spectrum.snr_measurements
    modes, powers = controller._hcf_settings()
    first = spectrum.get_hcf_candidates(["A", "B"], "c", 0, modes, powers)
    assert len(snr._hcf_physical_cache) > 0
    controller.engine_props["hcf_amplifier_budget_dbm"] = {"c": -20}
    second = spectrum.get_hcf_candidates(["A", "B"], "c", 0, modes, powers)
    assert second[0].gsnr_db == first[0].gsnr_db
    assert not spectrum.hcf_power_fits(second[0], {})
    assert second[0].demands != first[0].demands
    controller.reset_hcf_state()
    assert len(snr._hcf_physical_cache) == snr._hcf_physical_cache.current_bytes == 0


@pytest.mark.parametrize("objective", ["discrete", "fixed", "max_gsnr", "minimum_required"])
@pytest.mark.parametrize("policy", ["first_channel_ranked", "power_spectrum"])
def test_cached_and_uncached_arrivals_have_identical_decisions(controller, objective, policy):
    other = controller_fixture.__wrapped__()
    traces = []
    for instance, reuse in ((controller, True), (other, False)):
        instance.engine_props.update(hcf_power_objective=objective, hcf_selection_policy=policy,
                                     hcf_amplifier_budget_dbm={"c": 38}, hcf_power_weight=0.5)
        if not reuse:
            instance.spectrum_obj.snr_measurements._hcf_physical_cache = HCFPhysicalCache(max_entries=0)
        trace = []
        for request_id in range(1, 9):
            prepare_request(instance, request_id=request_id)
            arrive(instance)
            result = copy.deepcopy(instance.hcf_last_result)
            result.pop("allocation_time_seconds")
            trace.append(result)
        traces.append(trace)
    assert traces[0] == traces[1]
    assert controller.hcf_power_used == other.hcf_power_used

