"""Minimum-power physics and probe-free dynamic HCF allocation regressions."""

import copy
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fusion.configs.config import ConfigManager
from fusion.core.hcf import HCFCandidate, hcf_power_objective
from fusion.core.tests.test_hcf_controller import arrive, prepare_request
from fusion.core.tests.test_hcf_controller import controller as hcf_controller_fixture

controller = hcf_controller_fixture


def physical_args(**overrides):
    args = {"path": ["A", "B", "C"], "band": "c", "channel": 0, "modulation": "QPSK",
            "bit_rate_gbps": 100, "baud_rate_gbaud": 50, "threshold_db": 6.72,
            "margin_db": 1.0, "transceiver_snr_db": 25.0}
    args.update(overrides)
    return args


@pytest.mark.parametrize("channel", [0, 2])
@pytest.mark.parametrize("margin", [0.0, 1.0, 3.0])
def test_minimum_power_hits_threshold_and_lower_power_fails(controller, channel, margin):
    snr = controller.spectrum_obj.snr_measurements
    args = physical_args(channel=channel, margin_db=margin)
    with patch.object(snr, "_hcf_noise_coefficients", wraps=snr._hcf_noise_coefficients) as coefficients:
        ok, value, rate, power = snr.check_gsnr_hcf_minimum_power(**args)
    assert coefficients.call_count == 3  # 80 km + 100 km + 20 km, once each.
    assert ok and rate == 100 and power > 0
    assert value == pytest.approx(args["threshold_db"] + margin, abs=1e-9)
    _, peak, _, optimum = snr.check_gsnr_hcf_optimized_power(**args)
    assert power < optimum
    _, independently_checked, _ = snr.check_gsnr_hcf(**args, power_w=power)
    assert independently_checked >= args["threshold_db"] + margin - 1e-9
    assert not snr.check_gsnr_hcf(**args, power_w=power * 0.999)[0]
    assert peak == pytest.approx(snr.check_gsnr_hcf(**args, power_w=optimum)[1], abs=1e-10)
    assert snr.check_gsnr_hcf(**args, power_w=optimum * 0.9)[1] < peak
    assert snr.check_gsnr_hcf(**args, power_w=optimum * 1.1)[1] < peak


def test_noise_floor_and_unreachable_peak_are_rejected(controller):
    snr = controller.spectrum_obj.snr_measurements
    assert not snr.check_gsnr_hcf_minimum_power(**physical_args(threshold_db=25))[0]
    peak = snr.check_gsnr_hcf_optimized_power(**physical_args())[1]
    result = snr.check_gsnr_hcf_minimum_power(**physical_args(threshold_db=peak + 0.1, margin_db=0))
    assert not result[0] and result[3] == 0


def test_zero_nli_uses_required_power_even_above_configured_fallback(controller):
    for link in controller.engine_props["topology_info"]["links"].values():
        link["fiber"]["hcf"]["non_linearity"] = 0.0
    controller.engine_props["input_power"] = 1e-12
    snr = controller.spectrum_obj.snr_measurements
    args = physical_args()
    ok, value, _, power = snr.check_gsnr_hcf_minimum_power(**args)
    assert ok and power > controller.engine_props["input_power"]
    assert value == pytest.approx(7.72, abs=1e-8)
    assert not snr.check_gsnr_hcf(**args, power_w=power * 0.999)[0]


def test_gain_failure_returns_no_minimum_power(controller):
    controller.engine_props["topology_info"]["links"][1]["fiber"]["hcf"]["attenuation"] = 1.0
    result = controller.spectrum_obj.snr_measurements.check_gsnr_hcf_minimum_power(**physical_args())
    assert result == (False, -math.inf, 100.0, 0.0)


@pytest.mark.parametrize("objective", ["fixed", "max_gsnr", "minimum_required"])
def test_objectives_allocate_and_release_exact_directional_demands(controller, objective):
    controller.engine_props.update(hcf_power_objective=objective, hcf_power_levels_dbm=[],
                                   hcf_amplifier_budget_dbm={"c": 38}, input_power=0.002)
    with patch.object(controller, "_check_snr_after_allocation", side_effect=AssertionError("No rechecks")):
        arrive(controller)
        assert controller.sdn_props.was_routed
        candidate = controller.hcf_allocations[1]
        assert controller.hcf_power_used == {key: power for key, power, _ in candidate.demands}
        if objective == "fixed":
            assert {power for _, power, _ in candidate.demands} == {0.002}
        controller.handle_hcf_event({}, "release")
    assert not controller.hcf_allocations and not controller.hcf_power_used
    assert all(np.all(link["cores_matrix"]["c"] == 0) for link in controller.sdn_props.network_spectrum_dict.values())


def test_explicit_fixed_objective_overrides_optimized_boolean_and_grid(controller):
    controller.engine_props.update(hcf_power_objective="fixed", optimized_power=True, input_power=0.002)
    modes, powers = controller._hcf_settings()
    snr = controller.spectrum_obj.snr_measurements
    with patch.object(snr, "check_gsnr_hcf_optimized_power", side_effect=AssertionError("Fixed only")):
        choices = controller.spectrum_obj.get_hcf_candidates(["A", "B"], "c", 0, modes, powers)
    assert len(choices) == 1
    assert {p for _, p, _ in choices[0].demands} == {0.002}


def test_minimum_power_can_fit_budget_when_maximum_gsnr_cannot(controller):
    controller.engine_props.update(hcf_power_objective="max_gsnr", hcf_amplifier_budget_dbm={"c": 0})
    arrive(controller)
    assert not controller.sdn_props.was_routed
    assert controller.sdn_props.block_reason == "hcf_power"
    controller.engine_props["hcf_power_objective"] = "minimum_required"
    prepare_request(controller, request_id=2)
    arrive(controller)
    assert controller.sdn_props.was_routed
    assert all(p <= budget for _, p, budget in next(iter(controller.hcf_allocations.values())).demands)


@pytest.mark.parametrize("failure", ["gsnr", "power"])
def test_ranked_checks_first_available_only_and_never_calls_probes(controller, failure):
    controller.engine_props.update(hcf_power_objective="minimum_required", hcf_selection_policy="power_spectrum",
                                   hcf_amplifier_budget_dbm={"c": 38})
    spectrum = controller.spectrum_obj
    if failure == "gsnr":
        controller.engine_props["topology_info"]["links"][1]["fiber"]["hcf"]["band_profiles"] = {
            "c": {"attenuation": [1.0, 0.11, 0.11]}}
    else:
        controller.hcf_power_used[("A", "B", 0, "c")] = 10.0
    with (
        patch.object(spectrum, "_hcf_probes", side_effect=AssertionError("No probes")),
        patch.object(spectrum, "_hcf_opportunity", side_effect=AssertionError("No opportunity scan")),
        patch.object(spectrum, "get_hcf_candidates", wraps=spectrum.get_hcf_candidates) as evaluate,
    ):
        arrive(controller)
    assert [(call.args[0], call.args[2]) for call in evaluate.call_args_list] == [
        (["A", "B", "C"], 0), (["A", "C"], 0)]
    assert controller.sdn_props.was_routed
    assert controller.sdn_props.path_list == ["A", "C"]


@pytest.mark.parametrize("objective", ["fixed", "max_gsnr", "minimum_required"])
def test_c_full_falls_back_to_l_and_releases_only_new_reservations(controller, objective):
    controller.engine_props.update(hcf_power_objective=objective, hcf_selection_policy="power_spectrum",
                                   band_list=["c", "l"], l_band=3, hcf_amplifier_budget_dbm={"c": 38, "l": 38})
    for link in controller.engine_props["topology_info"]["links"].values():
        link["fiber"]["frequency_start_l"] = 185e12
    for link in controller.sdn_props.network_spectrum_dict.values():
        link["cores_matrix"]["c"][:] = 99
        link["cores_matrix"]["l"] = np.zeros((1, 3))
        link["cores_matrix"]["l"][0, 0] = 99
    with patch.object(controller.spectrum_obj, "_hcf_opportunity", side_effect=AssertionError("No probes")):
        arrive(controller)
        assert controller.sdn_props.was_routed
        candidate = controller.hcf_allocations[1]
        assert candidate.band == "l" and candidate.channel == 1
        assert all(key[-1] == "l" for key in controller.hcf_power_used)
        controller.handle_hcf_event({}, "release")
    assert not controller.hcf_power_used
    for link in controller.sdn_props.network_spectrum_dict.values():
        assert np.all(link["cores_matrix"]["c"] == 99)
        np.testing.assert_array_equal(link["cores_matrix"]["l"], [[99, 0, 0]])


@pytest.mark.parametrize("weight,expected_path", [(0.0, ["A", "B", "C"]), (1.0, ["A", "C"])])
def test_scarcity_weight_changes_path_without_mutating_search_state(controller, weight, expected_path):
    controller.engine_props.update(hcf_power_objective="fixed", hcf_selection_policy="power_spectrum",
                                   hcf_power_weight=weight, input_power=0.001)
    # Direct path has one free wavelength, alternate has three on each link.
    controller.sdn_props.network_spectrum_dict[("A", "C")]["cores_matrix"]["c"][0, 1:] = 99
    # Alternate has almost no power headroom; direct has all its power budget.
    controller.hcf_power_used[("A", "B", 0, "c")] = 0.0089
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    before = copy.deepcopy(controller.sdn_props.network_spectrum_dict)
    used = dict(controller.hcf_power_used)
    results = spectrum.rank_hcf_first_channels(
        [(0, ["A", "B", "C"]), (1, ["A", "C"])], modes, powers,
        power_used=used, get_probe_paths=controller._hcf_paths, probe_modes=modes)
    assert list(results[0][1].candidate.path) == expected_path
    assert controller.hcf_power_used == used
    assert not controller.hcf_allocations
    for edge, link in before.items():
        np.testing.assert_array_equal(link["cores_matrix"]["c"],
                                      controller.sdn_props.network_spectrum_dict[edge]["cores_matrix"]["c"])


def test_ranking_uses_each_direction_power_instead_of_reverse_scalar(controller):
    controller.engine_props.update(hcf_selection_policy="power_spectrum", hcf_power_objective="minimum_required",
                                   hcf_power_weight=1.0)
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    forward, reverse = ("A", "B", 0, "c"), ("B", "A", 0, "c")
    a = HCFCandidate(("A", "B"), "c", 0, modes[0], 0.001, 20,
                     ((forward, 0.008, 0.01), (reverse, 0.001, 0.01)))
    b = HCFCandidate(("A", "B"), "c", 0, modes[0], 0.002, 20,
                     ((forward, 0.002, 0.01), (reverse, 0.002, 0.01)))
    with patch.object(spectrum, "get_hcf_candidates", return_value=(a, b)):
        result = spectrum.get_spectrum_hcf(["A", "B"], modes, powers, power_used={},
                                          get_probe_paths=controller._hcf_paths, probe_modes=modes)
    assert result.candidate == b
    assert result.score == pytest.approx(0.4)


def test_objective_validation_and_legacy_defaults():
    assert hcf_power_objective({}) == "discrete"
    assert hcf_power_objective({"optimized_power": True}) == "max_gsnr"
    with pytest.raises(ValueError, match="hcf_power_objective"):
        hcf_power_objective({"hcf_power_objective": "typo"})


def test_four_experiment_configs_have_identical_traffic_and_physics():
    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    configurations = [
        ("fixed_first_fit", "fixed", "first_feasible"),
        ("max_gsnr_first_fit", "max_gsnr", "first_feasible"),
        ("minimum_power_first_fit", "minimum_required", "first_feasible"),
        ("minimum_power_ranked", "minimum_required", "power_spectrum"),
    ]
    common = None
    for name, objective, policy in configurations:
        config = ConfigManager(str(root / f"hcf_{name}.ini")).get_config()
        assert config.general["hcf_power_objective"] == objective
        assert config.general["hcf_selection_policy"] == policy
        assert config.general["thread_erlangs"] is False
        general, snr = dict(config.general), dict(config.snr)
        for key in ("hcf_power_objective", "hcf_selection_policy"):
            general.pop(key)
        snr.pop("optimized_power")
        same = (general, snr, config.topology, config.spectrum)
        if common is None:
            common = same
        assert same == common



@pytest.mark.parametrize("mode,rate,threshold", [
    ("BPSK", 100, 3.71), ("QPSK", 200, 6.72), ("8-QAM", 300, 10.84),
    ("16-QAM", 400, 13.24), ("32-QAM", 500, 16.16), ("64-QAM", 600, 19.01),
])
def test_64_gbaud_modes_meet_their_individual_thresholds(controller, mode, rate, threshold):
    snr = controller.spectrum_obj.snr_measurements
    args = physical_args(modulation=mode, bit_rate_gbps=rate, baud_rate_gbaud=64, threshold_db=threshold)
    ok, gsnr, supported_rate, power = snr.check_gsnr_hcf_minimum_power(**args)
    assert ok and supported_rate == rate
    assert gsnr == pytest.approx(threshold + 1.0, abs=1e-9)
    assert snr.check_gsnr_hcf(**args, power_w=power)[1] >= threshold + 1.0 - 1e-9
    assert not snr.check_gsnr_hcf(**args, power_w=power * 0.999)[0]

