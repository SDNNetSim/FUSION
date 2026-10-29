"""HCF path/wavelength allocation, independent QoT, and resource lifecycle tests."""

import copy
import math
from types import MappingProxyType
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
import pytest

from fusion.core.sdn_controller import SDNController


@pytest.fixture
def controller():
    topology = nx.Graph()
    network, links = {}, {}
    for index, (source, destination, length) in enumerate((("A", "B", 80), ("B", "C", 120), ("A", "C", 300)), 1):
        topology.add_edge(source, destination, length=length, weight=length)
        matrix = {"c": np.zeros((1, 3))}
        for edge in ((source, destination), (destination, source)):
            network[edge] = {"link_num": index, "cores_matrix": matrix, "usage_count": 0, "throughput": 0}
        links[index] = {
            "source": source,
            "destination": destination,
            "length": length,
            "span_length": 100,
            "fiber": {
                "num_cores": 1,
                "frequency_start_c": 190e12,
                "hcf": {
                    "attenuation": 0.11, "noise_figure": 5, "inter_modal_interference": -60, "max_amplifier_gain": 32,
                    "dispersion": -2.55e-27, "non_linearity": 5e-7,
                },
            },
        }
    props = {
        "fixed_grid": True,
        "hollow_core_fiber": True,
        "bi_directional": True,
        "cores_per_link": 1,
        "guard_slots": 0,
        "max_segments": 1,
        "is_grooming_enabled": False,
        "snr_recheck": False,
        "optimized_power": False,
        "input_power": 0.001,
        "route_method": "k_shortest_path",
        "k_paths": 2,
        "band_list": ["c"],
        "c_band": 3,
        "bw_per_slot": 75,
        "topology": topology,
        "topology_info": {"links": links, "nodes": {node: {} for node in topology}},
        "hcf_modes": [{"name": "QPSK", "net_rate_gbps": 100, "baud_rate_gbaud": 50, "threshold_db": 6.72}],
        "hcf_power_levels_dbm": [0, 3],
        "hcf_amplifier_budget_dbm": {"c": 10},
        "hcf_selection_policy": "first_feasible",
        "request_distribution": {"100": 1.0},
        "hcf_probes": [{"source": "A", "destination": "C", "bandwidth": 100}],
    }
    result = SDNController(props)
    result.sdn_props.network_spectrum_dict = network
    result.sdn_props.topology = topology
    prepare_request(result)
    return result


def prepare_request(controller, request_id=1, source="A", destination="C", bandwidth=100):
    props = controller.sdn_props
    props.request_id, props.source, props.destination = request_id, source, destination
    props.bandwidth, props.arrive, props.depart = bandwidth, 0, 10
    props.modulation_formats_dict = {"QPSK": {"max_length": 1, "bandwidth": 100}}


def arrive(controller, **kwargs):
    controller.handle_hcf_event({}, "arrival", **kwargs)


def test_next_wavelength_on_same_path_after_gsnr_failure(controller):
    # Channel 0 exceeds amplifier gain, while channel 1 is physically feasible.
    controller.engine_props["topology_info"]["links"][1]["fiber"]["hcf"]["band_profiles"] = {
        "c": {"attenuation": [1.0, 0.11, 0.11]}
    }
    with patch.object(controller, "_check_snr_after_allocation", side_effect=AssertionError("No rechecks")):
        arrive(controller)
    assert controller.sdn_props.path_list == ["A", "B", "C"]
    assert controller.sdn_props.start_slot_list == [1]
    assert controller.sdn_props.lightpath_id_list == [1]
    assert controller.sdn_props.was_routed


def test_exhaust_current_path_before_next_path(controller):
    controller.engine_props["topology_info"]["links"][1]["fiber"]["hcf"]["attenuation"] = 1.0
    with patch.object(controller.spectrum_obj, "get_hcf_candidates", wraps=controller.spectrum_obj.get_hcf_candidates) as evaluator:
        arrive(controller)
    assert [(call.args[0], call.args[2]) for call in evaluator.call_args_list] == [
        (["A", "B", "C"], 0), (["A", "B", "C"], 1), (["A", "B", "C"], 2), (["A", "C"], 0)
    ]
    assert controller.sdn_props.path_list == ["A", "C"]


@pytest.mark.parametrize("policy", ["first_feasible", "first_channel_ranked"])
def test_round_trip_power_spectrum_and_no_established_channel_rechecks(controller, policy):
    controller.engine_props["hcf_selection_policy"] = policy
    with patch.object(controller, "_check_snr_after_allocation", side_effect=AssertionError("No rechecks")):
        arrive(controller)
        candidate = controller.hcf_allocations[1]
        assert len(candidate.demands) == 6  # AB: 1 span; BC: 2 spans; both directions.
        assert all(value == pytest.approx(0.001) for value in controller.hcf_power_used.values())
        controller.handle_hcf_event({}, "release")
    assert not controller.hcf_power_used
    assert not controller.hcf_allocations
    assert not controller.sdn_props.allocated_requests
    assert all(np.all(link["cores_matrix"]["c"] == 0) for link in controller.sdn_props.network_spectrum_dict.values())
    throughput = controller.sdn_props.network_spectrum_dict[("A", "B")]["throughput"]
    controller.handle_hcf_event({}, "release")
    assert controller.sdn_props.network_spectrum_dict[("A", "B")]["throughput"] == throughput


def test_power_shortage_blocks_without_disturbing_existing_channel(controller):
    controller.engine_props["hcf_amplifier_budget_dbm"] = {"c": 0}
    arrive(controller, force_route_matrix=[["A", "B", "C"]])
    first = controller.hcf_allocations[1]
    prepare_request(controller, request_id=2)
    arrive(controller, force_route_matrix=[["A", "B", "C"]])
    assert not controller.sdn_props.was_routed
    assert controller.sdn_props.block_reason == "hcf_power"
    assert controller.hcf_allocations == {1: first}
    assert all(value == pytest.approx(0.001) for value in controller.hcf_power_used.values())
    assert controller.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"]["c"][0].tolist() == [1, 0, 0]


def test_oversized_request_is_blocked_without_slicing(controller):
    controller.sdn_props.bandwidth = 200
    arrive(controller)
    assert not controller.sdn_props.was_routed
    assert controller.sdn_props.block_reason == "distance"
    assert not controller.hcf_allocations


@pytest.mark.parametrize("changes", [{"snr_recheck": True}, {"max_segments": 2}, {"is_grooming_enabled": True}, {"bi_directional": False}])
def test_incompatible_settings_rejected(controller, changes):
    controller.engine_props.update(changes)
    with pytest.raises(ValueError):
        arrive(controller)
    assert not controller.hcf_allocations


def test_read_only_search_and_virtual_opportunity_count(controller):
    modes, powers = controller._hcf_settings()
    spectrum = controller.spectrum_obj
    before = spectrum._hcf_opportunity(modes, powers, controller.hcf_power_used, controller._hcf_paths)
    assert before == pytest.approx(math.log1p(6))  # 2 paths x 3 wavelengths; not x powers.
    candidate = controller.spectrum_obj.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)[0]
    after = spectrum._hcf_opportunity(modes, powers, controller.hcf_power_used, controller._hcf_paths, candidate)
    assert after == pytest.approx(math.log1p(5))
    assert not controller.hcf_power_used
    assert controller.sdn_props.lightpath_counter == 0
    assert all(np.all(link["cores_matrix"]["c"] == 0) for link in controller.sdn_props.network_spectrum_dict.values())


@pytest.mark.parametrize("request_rate", [100, 200])
def test_joint_score_preserves_scarce_wavelength_for_probe(controller, request_rate):
    controller.engine_props.update({
        "hcf_selection_policy": "joint",
        "hcf_power_weight": 0.0,
        "hcf_probes": [{"source": "A", "destination": "C", "bandwidth": 100}],
    })
    controller.engine_props["hcf_modes"].append(
        {"name": "16-QAM", "net_rate_gbps": 200, "baud_rate_gbaud": 50, "threshold_db": 20}
    )
    # Probe's direct AC path is occupied; ABC can use only channel 0 and QPSK.
    # The 200-Gbps arrival cannot use QPSK, but probes must retain that option.
    for edge in (("A", "C"), ("C", "A")):
        controller.sdn_props.network_spectrum_dict[edge]["cores_matrix"]["c"][0][:] = 99
    controller.engine_props["topology_info"]["links"][1]["fiber"]["hcf"]["band_profiles"] = {
        "c": {"attenuation": [0.39, 1.0, 1.0]}
    }
    prepare_request(controller, source="B", destination="C", bandwidth=request_rate)
    arrive(controller, force_route_matrix=[["B", "C"]])
    assert controller.sdn_props.was_routed
    assert controller.sdn_props.start_slot_list == [1]


def test_scoring_can_choose_later_wavelength_on_first_path(controller):
    controller.engine_props.update({"hcf_selection_policy": "joint", "hcf_power_weight": 0})
    # A controlled scorer isolates selection/order from the physical-model test.
    def opportunity(modes, powers, power_used, get_probe_paths, virtual=None):
        return 10 if virtual is None else 9 if virtual.channel == 2 else 5

    with patch.object(controller.spectrum_obj, "_hcf_opportunity", side_effect=opportunity):
        arrive(controller)
    assert controller.sdn_props.path_list == ["A", "B", "C"]
    assert controller.sdn_props.start_slot_list == [2]


def test_rollback_restores_partial_spectrum_write_and_request_state(controller):
    original = controller._allocate_on_path

    def interrupted(path):
        original(path[:2])
        raise RuntimeError("injected mid-path failure")

    with patch.object(controller, "_allocate_on_path", side_effect=interrupted), pytest.raises(RuntimeError, match="mid-path"):
        arrive(controller)
    assert not controller.hcf_allocations
    assert not controller.hcf_power_used
    assert controller.sdn_props.lightpath_counter == 0
    assert not controller.sdn_props.lightpath_id_list
    assert not controller.sdn_props.allocated_requests
    assert not controller.sdn_props.was_routed
    for link in controller.sdn_props.network_spectrum_dict.values():
        assert np.all(link["cores_matrix"]["c"] == 0)
        assert link["usage_count"] == 0


def test_reset_before_id_reuse_removes_active_hcf_state(controller):
    arrive(controller)
    controller.reset_hcf_state()
    controller.sdn_props.reset_lightpath_id_counter()
    prepare_request(controller, request_id=2)
    arrive(controller)
    assert controller.sdn_props.lightpath_id_list == [1]
    assert all(value == pytest.approx(0.001) for value in controller.hcf_power_used.values())


def test_failed_release_restores_spectrum_tracking_and_power(controller):
    arrive(controller)
    powers = dict(controller.hcf_power_used)
    original = controller.release

    def interrupted(**kwargs):
        original(**kwargs)
        raise RuntimeError("injected release failure")

    with patch.object(controller, "release", side_effect=interrupted), pytest.raises(RuntimeError, match="release failure"):
        controller.handle_hcf_event({}, "release")
    assert controller.hcf_power_used == powers
    assert 1 in controller.hcf_allocations
    assert 1 in controller.sdn_props.allocated_requests
    assert 1 in controller.sdn_props.lightpath_status_dict[("A", "C")]
    assert controller.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"]["c"][0][0] == 1
    assert controller.sdn_props.network_spectrum_dict[("A", "B")]["throughput"] == 0


def test_nonmonotone_gsnr_does_not_assume_highest_power_is_best(controller):
    controller.engine_props["hcf_power_levels_dbm"] = [-10, 0, 10]
    controller.engine_props["hcf_modes"][0]["threshold_db"] = 15
    for link in controller.engine_props["topology_info"]["links"].values():
        link["fiber"]["hcf"]["non_linearity"] = 0.003
    modes, powers = controller._hcf_settings()
    choices = controller.spectrum_obj.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)
    assert choices
    assert all(c.power_w < 0.01 for c in choices)
    arrive(controller)
    assert controller.sdn_props.was_routed
    assert controller.hcf_allocations[1].power_w < 0.01


def test_physical_gsnr_independent_of_other_occupancy(controller):
    modes, powers = controller._hcf_settings()
    before = controller.spectrum_obj.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)
    controller.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"]["c"][0][1] = 99
    after = controller.spectrum_obj.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)
    assert before == after


def test_existing_hcf_link_calculator_ase_imi_reference(controller):
    profile = {"attenuation": 0.1, "noise_figure": 3, "inter_modal_interference": -60, "dispersion": -2.55e-27, "non_linearity": 0}
    snr = controller.spectrum_obj.snr_measurements
    _, actual, power = snr.calculate_hcf_snr_link(3, 1, 100, 0.05, 75e9, 190e12, power_w=0.001, fiber={"hcf": profile})
    assert power == 0.001
    expected_ase = 6.62607015e-34 * 190e12 * 10**0.3 * 50e9 * 9
    assert actual == pytest.approx(expected_ase / 0.001 + 100e-6)


def test_hcf_link_gain_rejection_returns_three_values(controller):
    snr = controller.spectrum_obj.snr_measurements
    profile = dict(controller.engine_props["topology_info"]["links"][1]["fiber"]["hcf"])
    profile["max_amplifier_gain"] = 0
    linear, inverse, power = snr.calculate_hcf_snr_link(
        3, 1, 80, 0.05, 75e9, 190e12, power_w=0.001, fiber={"hcf": profile}
    )
    assert (linear, inverse, power) == (0.0, math.inf, 0.0)


def test_heterogeneous_links_and_partial_spans(controller):
    controller.engine_props["topology_info"]["links"][2]["fiber"]["hcf"]["noise_figure"] = 8
    modes, powers = controller._hcf_settings()
    candidate = controller.spectrum_obj.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)[0]
    links = controller.engine_props["topology_info"]["links"]
    inverse = sum(
        controller.spectrum_obj.snr_measurements.calculate_hcf_snr_link(
            3, 1, length, 0.05, 75e9, 190e12 + 0.5 * 75e9, power_w=candidate.power_w, fiber=links[link_id]["fiber"]
        )[1]
        for link_id, length in ((1, 80), (2, 100), (2, 20))
    )
    assert candidate.gsnr_db == pytest.approx(-10 * math.log10(inverse))


def test_handler_reuses_existing_routing_spectrum_and_hcf_gsnr(controller):
    from fusion.utils.spectrum import find_common_channels_on_paths

    snr = controller.spectrum_obj.snr_measurements
    with (
        patch.object(controller.route_obj, "get_route", wraps=controller.route_obj.get_route) as routing,
        patch.object(controller.spectrum_obj, "get_spectrum_hcf", wraps=controller.spectrum_obj.get_spectrum_hcf) as search,
        patch("fusion.core.spectrum_assignment.find_common_channels_on_paths", wraps=find_common_channels_on_paths) as spectrum,
        patch.object(snr, "check_gsnr_hcf", wraps=snr.check_gsnr_hcf) as gsnr,
        patch.object(snr, "calculate_hcf_snr_link", wraps=snr.calculate_hcf_snr_link) as link_gsnr,
    ):
        arrive(controller)
    routing.assert_called_once()
    search.assert_called_once()
    assert spectrum.called and gsnr.called and link_gsnr.called
    assert gsnr.call_args_list[0].kwargs["path"] == ("A", "B", "C")
    assert gsnr.call_args_list[0].kwargs["channel"] == 0


def test_spectrum_assignment_selects_candidate_without_reserving_resources(controller):
    controller.engine_props["hcf_selection_policy"] = "joint"
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    before = copy.deepcopy(vars(spectrum.spectrum_props))
    with patch.object(spectrum, "_calculate_slots_needed", wraps=spectrum._calculate_slots_needed) as slot_count:
        result = spectrum.get_spectrum_hcf(
            ["A", "B", "C"], modes, powers, power_used=MappingProxyType(controller.hcf_power_used),
            get_probe_paths=controller._hcf_paths, probe_modes=modes,
        )
    assert slot_count.called
    assert result.candidate.channel == 0
    expected_loss = (math.log1p(6) - math.log1p(5)) / math.log1p(6)
    assert result.score == pytest.approx(0.5 * 0.1 + 0.5 * expected_loss)
    assert result.free_channels == result.qot_feasible_channels == 3
    assert vars(spectrum.spectrum_props) == before
    assert controller.sdn_props.lightpath_counter == 0
    assert not controller.hcf_allocations and not controller.hcf_power_used
    assert all(np.all(link["cores_matrix"]["c"] == 0) for link in controller.sdn_props.network_spectrum_dict.values())


def test_spectrum_assignment_continues_after_power_rejection(controller):
    from dataclasses import replace

    modes, powers = controller._hcf_settings()
    spectrum = controller.spectrum_obj
    physical_candidates = spectrum.get_hcf_candidates

    def candidates(path, band, channel, modes, powers):
        choices = physical_candidates(path, band, channel, modes, powers)
        # Channel 0 has insufficient amplifier budget; later channels still fit.
        if channel == 0:
            return tuple(replace(c, demands=tuple((key, power, power / 2) for key, power, _ in c.demands)) for c in choices)
        return choices

    with patch.object(spectrum, "get_hcf_candidates", side_effect=candidates):
        result = spectrum.get_spectrum_hcf(
            ["A", "B", "C"], modes, powers, power_used=controller.hcf_power_used,
            get_probe_paths=controller._hcf_paths, probe_modes=modes,
        )
    assert result.candidate.channel == 1
    assert result.power_rejections == 1
    assert result.free_channels == 2


def test_transmission_options_and_probe_paths_are_reused(controller):
    modes, powers = controller._hcf_settings()
    spectrum = controller.spectrum_obj
    snr = spectrum.snr_measurements
    with patch.object(snr, "check_gsnr_hcf", wraps=snr.check_gsnr_hcf) as check:
        first = spectrum.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)
        calls = check.call_count
        second = spectrum.get_hcf_candidates(["A", "B", "C"], "c", 0, modes, powers)
        assert calls > 0 and check.call_count == calls
        assert first == second
    with patch.object(controller, "_hcf_paths", wraps=controller._hcf_paths) as paths:
        first = spectrum._hcf_probes(paths)
        calls = paths.call_count
        assert spectrum._hcf_probes(paths) == first
        assert calls > 0 and paths.call_count == calls


def test_default_modes_reuse_existing_snr_mappings(controller):
    controller.engine_props.pop("hcf_modes")
    modes, _ = controller._hcf_settings()
    snr_props = controller.spectrum_obj.snr_measurements.snr_props
    assert {m.name: m.net_rate_gbps for m in modes} == snr_props.bandwidth_mapping_dict[75]
    assert {m.name: m.threshold_db for m in modes} == snr_props.req_snr


def test_existing_no_argument_hcf_api_remains_supported(controller):
    snr = controller.spectrum_obj.snr_measurements
    spectrum = controller.spectrum_obj.spectrum_props
    spectrum.path_list, spectrum.current_band, spectrum.start_slot, spectrum.modulation = ["A", "B", "C"], "c", 0, "QPSK"
    state_before = copy.deepcopy(vars(spectrum))
    legacy = snr.check_gsnr_hcf()
    explicit = snr.check_gsnr_hcf(path=["A", "B", "C"], band="c", channel=0, modulation="QPSK", power_w=0.001)
    assert legacy == explicit
    assert legacy[2] == 200
    assert vars(spectrum) == state_before
    assert controller.engine_props["input_power"] == 0.001


def test_probe_routing_reuses_ksp_without_mutating_request(controller):
    from fusion.modules.routing.k_shortest_path import KShortestPath

    source, destination = controller.sdn_props.source, controller.sdn_props.destination
    with patch.object(KShortestPath, "get_paths", autospec=True, return_value=[["B", "C"]]) as ksp:
        paths = controller._hcf_paths("B", "C")
    assert paths == [["B", "C"]]
    assert ksp.call_args.args[1:] == ("B", "C")
    assert (controller.sdn_props.source, controller.sdn_props.destination) == (source, destination)


def test_reverse_direction_limits_bidirectional_channel(controller):
    controller.engine_props["topology_info"]["links"][4] = copy.deepcopy(controller.engine_props["topology_info"]["links"][1])
    controller.engine_props["topology_info"]["links"][4]["fiber"]["hcf"]["attenuation"] = 1
    controller.sdn_props.network_spectrum_dict[("B", "A")]["link_num"] = 4
    arrive(controller, force_route_matrix=[["A", "B", "C"]])
    assert not controller.sdn_props.was_routed
    assert controller.sdn_props.block_reason == "xt_threshold"


def test_invalid_mode_width_does_not_allocate(controller):
    controller.engine_props["hcf_modes"][0]["baud_rate_gbaud"] = 100
    arrive(controller)
    assert not controller.sdn_props.was_routed


def test_forced_channel_and_mode_are_honored(controller):
    arrive(controller, forced_index=2, force_mod_format="QPSK", force_route_matrix=[["A", "C"]])
    assert controller.sdn_props.start_slot_list == [2]
    assert controller.sdn_props.modulation_list == ["QPSK"]


def test_existing_spectrum_helper_finds_single_free_channels_after_gaps(controller):
    controller.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"]["c"][0][:] = [0, 99, 0]
    assert controller.spectrum_obj.get_hcf_free_channels(["A", "B", "C"], "c") == [0, 2]
    arrive(controller, forced_index=2, force_route_matrix=[["A", "B", "C"]])
    assert controller.sdn_props.was_routed
    assert controller.sdn_props.start_slot_list == [2]


def test_probe_rng_does_not_consume_global_random_state(controller):
    import random

    controller.engine_props.pop("hcf_probes")
    state = random.getstate()
    probes = controller.spectrum_obj._hcf_probes(controller._hcf_paths)
    assert random.getstate() == state
    assert controller.spectrum_obj._hcf_probes(controller._hcf_paths) == probes


def test_simulation_dispatches_both_arrival_and_release(controller):
    from fusion.core.simulation import SimulationEngine

    engine = SimulationEngine.__new__(SimulationEngine)
    engine.engine_props = {**controller.engine_props, "resource_allocation_mode": "hcf_power_opportunity"}
    engine.use_orchestrator = False
    engine.sdn_obj = controller
    engine.ml_model = None
    engine.dataset_logger = None
    engine.stats_obj = Mock()
    engine.stats_obj.stats_props.hcf_decisions_list = []
    arrival, departure = (1, 0.0), (1, 10.0)
    request = {"req_id": 1, "source": "A", "destination": "C", "bandwidth": 100, "arrive": 0, "depart": 10}
    engine.reqs_dict = {arrival: request, departure: request}
    engine.reqs_status_dict = {}
    engine.network_spectrum_dict = controller.sdn_props.network_spectrum_dict
    engine.grooming_stats = {}
    engine.handle_arrival(arrival)
    assert engine.reqs_status_dict[1]["lightpath_id_list"] == [1]
    assert len(engine.stats_obj.stats_props.hcf_decisions_list) == 1
    engine.handle_release(departure)
    assert not controller.hcf_power_used
    assert not controller.hcf_allocations


@pytest.mark.parametrize("explicit_catalog", [True, False])
def test_loaded_template_runs_real_event_loop_and_persists_hcf_metrics(controller, monkeypatch, explicit_catalog):
    import json
    from pathlib import Path

    from fusion.cli.config_setup import load_config
    from fusion.core.simulation import SimulationEngine

    monkeypatch.setenv("FUSION_USE_ORCHESTRATOR", "0")
    config_path = Path(__file__).parents[2] / "configs/templates/hcf_dynamic_power.ini"
    props = load_config(str(config_path))["s1"]
    props.update({
        "network": "hcf-test", "date": "test", "sim_start": "test", "erlang": 1,
        "topology_info": controller.engine_props["topology_info"], "c_band": 3, "save_snapshots": False,
        "hcf_amplifier_budget_dbm": {"c": -3}, "hcf_probe_count": 2,
        "mod_per_bw": {"150": {"QPSK": {"max_length": 1}}},
    })
    if explicit_catalog:
        props["hcf_modes"] = [{"name": "QPSK", "net_rate_gbps": 200, "baud_rate_gbaud": 50, "threshold_db": 6.72}]
    engine = SimulationEngine(props)
    engine.create_topology()
    engine.stats_obj.init_iter_stats()
    events = {}
    for index in range(3):
        request = {"req_id": index + 1, "source": "A", "destination": "C", "bandwidth": "150", "arrive": index, "depart": 10 + index}
        events[(index + 1, float(index))] = {**request, "request_type": "arrival"}
        events[(index + 1, float(10 + index))] = {**request, "request_type": "release"}
    engine.reqs_dict = dict(sorted(events.items(), key=lambda item: item[0][1]))
    for index, event in enumerate(engine.reqs_dict):
        engine.handle_request(event, index + 1)
    records = engine.stats_obj.stats_props.hcf_decisions_list
    assert len(records) == 3
    assert [record["accepted"] for record in records] == [True, True, False]
    assert engine.stats_obj.stats_props.block_reasons_dict["hcf_power"] == 1
    assert records[0]["requested_rate_gbps"] == 150
    assert records[0]["carrier_rate_gbps"] == 200
    assert engine.stats_obj.stats_props.lp_bw_utilization_dict["200"]["c"][0] == [75, 75]
    assert not engine.sdn_obj.hcf_power_used
    assert all(np.all(link["cores_matrix"]["c"] == 0) for link in engine.network_spectrum_dict.values())
    persisted = engine.persistence._prepare_iteration_stats(engine.stats_obj.stats_props)
    assert len(persisted["hcf_decisions_list"]) == 3
    json.dumps(persisted)
    engine.reset()
    engine.stats_obj.init_iter_stats()
    assert not engine.stats_obj.stats_props.hcf_decisions_list
    assert engine.sdn_obj.sdn_props.lightpath_counter == 0


def test_orchestrator_override_rejects_new_mode(controller, monkeypatch):
    from fusion.core.simulation import SimulationEngine

    monkeypatch.setenv("FUSION_USE_ORCHESTRATOR", "1")
    props = {
        **controller.engine_props, "resource_allocation_mode": "hcf_power_opportunity",
        "network": "test", "date": "test", "sim_start": "test",
    }
    with pytest.raises(ValueError, match="legacy engine"):
        SimulationEngine(props)


def test_first_channel_ranks_paths_and_shares_opportunity_baseline(controller):
    controller.engine_props.update({"hcf_selection_policy": "first_channel_ranked", "hcf_power_weight": 0.0})
    spectrum = controller.spectrum_obj

    def opportunity(modes, powers, power_used, get_probe_paths, virtual=None):
        return 10 if virtual is None else 9 if virtual.path == ("A", "C") else 5

    with (
        patch.object(spectrum, "_hcf_opportunity", side_effect=opportunity) as scorer,
        patch.object(spectrum, "get_hcf_candidates", wraps=spectrum.get_hcf_candidates) as candidates,
    ):
        arrive(controller)
    assert controller.sdn_props.path_list == ["A", "C"]
    assert controller.sdn_props.path_index == 1
    assert controller.hcf_last_result["score"] == pytest.approx(0.1)
    assert controller.hcf_last_result["ranked_candidates"] == 2
    assert controller.hcf_last_result["selected_rank"] == 1
    assert scorer.call_count == 3  # One baseline plus one virtual score per path.
    assert [(call.args[0], call.args[2]) for call in candidates.call_args_list] == [
        (["A", "B", "C"], 0), (["A", "C"], 0),
    ]


def test_first_channel_does_not_scan_later_channels_after_qot_failure(controller):
    controller.engine_props.update({"hcf_selection_policy": "first_channel_ranked", "hcf_power_weight": 1.0})
    controller.engine_props["topology_info"]["links"][1]["fiber"]["hcf"]["band_profiles"] = {
        "c": {"attenuation": [1.0, 0.11, 0.11]}
    }
    spectrum = controller.spectrum_obj
    with patch.object(spectrum, "get_hcf_candidates", wraps=spectrum.get_hcf_candidates) as candidates:
        arrive(controller, force_route_matrix=[["A", "B", "C"]])
    assert not controller.sdn_props.was_routed
    assert controller.sdn_props.block_reason == "xt_threshold"
    assert candidates.call_count == 1
    assert candidates.call_args.args[2] == 0
    assert not controller.hcf_power_used


@pytest.mark.parametrize("forced_index,expected", [(None, 1), (2, 2)])
def test_first_channel_skips_occupied_and_respects_forced_channel(controller, forced_index, expected):
    controller.engine_props.update({"hcf_selection_policy": "first_channel_ranked", "hcf_power_weight": 1.0})
    controller.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"]["c"][0, 0] = 99
    with patch.object(controller.spectrum_obj, "_hcf_opportunity", side_effect=AssertionError("Weight one needs no probes")):
        arrive(controller, force_route_matrix=[["A", "B", "C"]], forced_index=forced_index, forced_band="c")
    assert controller.sdn_props.was_routed
    assert controller.sdn_props.start_slot_list == [expected]
    assert controller.hcf_last_result["free_channels"] == 1


def test_first_channel_ties_preserve_routing_order(controller):
    controller.engine_props.update({"hcf_selection_policy": "first_channel_ranked", "hcf_power_weight": 0.0})
    with patch.object(controller.spectrum_obj, "_hcf_opportunity", return_value=0.0):
        arrive(controller)
    assert controller.sdn_props.path_list == ["A", "B", "C"]
    assert controller.hcf_last_result["score"] == 0.0


def test_first_channel_tries_next_rank_if_resources_changed(controller):
    controller.engine_props.update({"hcf_selection_policy": "first_channel_ranked", "hcf_power_weight": 0.0})
    spectrum = controller.spectrum_obj
    original = spectrum.rank_hcf_first_channels

    def rank_then_occupy(*args, **kwargs):
        ranked = original(*args, **kwargs)
        first = ranked[0][1].candidate
        controller.sdn_props.network_spectrum_dict[first.path[:2]]["cores_matrix"][first.band][0, first.channel] = 99
        return ranked

    with (
        patch.object(spectrum, "_hcf_opportunity", return_value=0.0),
        patch.object(spectrum, "rank_hcf_first_channels", side_effect=rank_then_occupy),
    ):
        arrive(controller)
    assert controller.sdn_props.path_list == ["A", "C"]
    assert controller.hcf_last_result["selected_rank"] == 2
    assert len(controller.hcf_allocations) == 1


def test_first_channel_power_failure_does_not_scan_next_channel(controller):
    controller.engine_props.update({"hcf_selection_policy": "first_channel_ranked", "hcf_power_weight": 1.0})
    controller.engine_props["hcf_amplifier_budget_dbm"] = {"c": -10}
    spectrum = controller.spectrum_obj
    with patch.object(spectrum, "get_hcf_candidates", wraps=spectrum.get_hcf_candidates) as candidates:
        arrive(controller)
    assert not controller.sdn_props.was_routed
    assert controller.sdn_props.block_reason == "hcf_power"
    assert candidates.call_count == 2
    assert controller.hcf_last_result["power_rejections"] == 2
    assert not controller.hcf_power_used


def test_first_channel_search_is_read_only_and_stops_across_bands(controller):
    controller.engine_props.update({
        "hcf_selection_policy": "first_channel_ranked", "hcf_power_weight": 1.0, "band_list": ["c", "l"], "l_band": 3,
    })
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    before = copy.deepcopy(controller.sdn_props.network_spectrum_dict)
    with patch.object(spectrum, "get_hcf_free_channels", return_value=[0, 1, 2]) as channels:
        results = spectrum.rank_hcf_first_channels(
            [(0, ["A", "B", "C"])], modes, powers, power_used=MappingProxyType(controller.hcf_power_used),
            get_probe_paths=controller._hcf_paths, probe_modes=modes,
        )
    assert results[0][1].candidate.band == "c"
    assert channels.call_count == 1
    assert not controller.hcf_power_used
    assert not controller.hcf_allocations
    assert controller.sdn_props.lightpath_counter == 0
    for edge, link in before.items():
        np.testing.assert_array_equal(link["cores_matrix"]["c"], controller.sdn_props.network_spectrum_dict[edge]["cores_matrix"]["c"])


@pytest.mark.parametrize("optimized", [False, True])
def test_power_search_uses_one_optimized_evaluation_or_full_discrete_grid(controller, optimized):
    from fusion.core.hcf import HCFMode

    controller.engine_props["optimized_power"] = optimized
    modes = [HCFMode("QPSK", 200, 64, 6.72), HCFMode("16-QAM", 400, 64, 13.24)]
    powers = [0.0005, 0.001, 0.002]
    snr = controller.spectrum_obj.snr_measurements
    with (
        patch.object(snr, "check_gsnr_hcf_optimized_power", return_value=(True, 20.0, 200.0, 0.003)) as opt,
        patch.object(snr, "check_gsnr_hcf", return_value=(True, 20.0, 200.0)) as discrete,
    ):
        choices = controller.spectrum_obj.get_hcf_candidates(["A", "B", "C"], "c", 1, modes, powers)
    if optimized:
        assert opt.call_count == len(modes) * 2
        discrete.assert_not_called()
        assert len(choices) == len(modes)
        assert all(c.power_w == 0.003 for c in choices)
        calls = opt.call_args_list
    else:
        opt.assert_not_called()
        assert discrete.call_count == len(modes) * len(powers) * 2
        assert len(choices) == len(modes) * len(powers)
        assert {c.power_w for c in choices} == set(powers)
        calls = discrete.call_args_list
    for mode in modes:
        for direction in (("A", "B", "C"), ("C", "B", "A")):
            matching = [call for call in calls if call.kwargs["modulation"] == mode.name
                        and call.kwargs["path"] == direction]
            assert len(matching) == (1 if optimized else len(powers))


@pytest.mark.parametrize("policy", ["first_feasible", "first_channel_ranked"])
@pytest.mark.parametrize("alternative_fits", [True, False])
def test_optimized_selection_checks_all_directional_budgets(controller, policy, alternative_fits):
    from fusion.core.hcf import HCFCandidate, HCFMode

    controller.engine_props.update(optimized_power=True, hcf_selection_policy=policy, hcf_power_weight=1.0)
    spectrum = controller.spectrum_obj
    modes = [HCFMode("QPSK", 100, 50, 6.72), HCFMode("QPSK", 200, 64, 6.72)]
    forward, reverse = ("A", "B", 0, "c"), ("B", "A", 0, "c")
    used = {forward: 0.005, reverse: 0.005}
    # Only 5 mW remains at each amplifier. A has the smaller scalar but
    # exceeds the forward budget; B can fit despite a larger reverse power.
    a = HCFCandidate(("A", "B"), "c", 0, modes[0], 0.002, 20.0,
                     ((forward, 0.006, 0.01), (reverse, 0.002, 0.01)))
    b = HCFCandidate(("A", "B"), "c", 0, modes[1], 0.003, 20.0,
                     ((forward, 0.004 if alternative_fits else 0.006, 0.01), (reverse, 0.003, 0.01)))
    with patch.object(spectrum, "get_hcf_candidates", return_value=(a, b)):
        result = spectrum.get_spectrum_hcf(
            ["A", "B"], modes, [0.001], power_used=MappingProxyType(used),
            get_probe_paths=controller._hcf_paths, probe_modes=modes, forced_index=0,
        )
    assert result.candidate == (b if alternative_fits else None)
    assert result.power_rejections == (0 if alternative_fits else 1)
    assert result.free_channels == result.qot_feasible_channels == 1
    assert used == {forward: 0.005, reverse: 0.005}
    assert not controller.hcf_allocations
    assert all(np.all(link["cores_matrix"]["c"] == 0) for link in controller.sdn_props.network_spectrum_dict.values())


@pytest.mark.parametrize("automatic", [False, True])
def test_probe_reuse_is_cleared_at_iteration_reset(controller, automatic):
    spectrum = controller.spectrum_obj
    if automatic:
        controller.engine_props.update(hcf_probes=None, hcf_probe_count=3, hcf_probe_seed=7)
    with patch.object(controller, "_hcf_paths", wraps=controller._hcf_paths) as paths:
        first = spectrum._hcf_probes(paths)
        calls = paths.call_count
        assert calls == (3 if automatic else 1)
        assert spectrum._hcf_probes(paths) is first
        assert paths.call_count == calls
        controller.reset_hcf_state()
        assert spectrum._hcf_probe_cache is None
        rebuilt = spectrum._hcf_probes(paths)
        assert rebuilt == first
        assert rebuilt is not first
        assert paths.call_count == 2 * calls


def test_probe_reset_picks_up_new_routing_and_probe_configuration(controller):
    spectrum = controller.spectrum_obj
    assert len(spectrum._hcf_probes(controller._hcf_paths)[0][0]) == 2
    controller.engine_props.update(k_paths=1, hcf_probes=None, hcf_probe_count=3, hcf_probe_seed=9)
    controller.reset_hcf_state()
    probes = spectrum._hcf_probes(controller._hcf_paths)
    assert len(probes) == 3
    assert all(len(paths) == 1 for paths, _, _ in probes)
    assert sum(weight for _, _, weight in probes) == pytest.approx(1.0)


def test_cached_probe_paths_still_check_live_spectrum_and_power(controller):
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    paths = Mock(return_value=[["A", "B", "C"]])
    initial = spectrum._hcf_opportunity(modes, powers, {}, paths)
    assert initial == pytest.approx(math.log1p(3))
    controller.sdn_props.network_spectrum_dict[("A", "B")]["cores_matrix"]["c"][0, 0] = 99
    occupied = spectrum._hcf_opportunity(modes, powers, {}, paths)
    assert occupied == pytest.approx(math.log1p(2))
    exhausted = spectrum._hcf_opportunity(modes, powers, {("A", "B", 0, "c"): 0.01}, paths)
    assert exhausted == 0.0
    assert paths.call_count == 1
    # Clearing only reuse reproduces the same score for the current resources.
    spectrum.clear_hcf_probe_cache()
    assert spectrum._hcf_opportunity(modes, powers, {}, paths) == occupied
    assert paths.call_count == 2


def test_probe_reuse_preserves_first_channel_ranking(controller):
    controller.engine_props.update(hcf_selection_policy="first_channel_ranked", hcf_power_weight=0.5)
    spectrum = controller.spectrum_obj
    modes, powers = controller._hcf_settings()
    args = {"power_used": {}, "get_probe_paths": controller._hcf_paths, "probe_modes": modes}
    candidate_paths = [(0, ["A", "B", "C"]), (1, ["A", "C"])]
    cached = spectrum.rank_hcf_first_channels(candidate_paths, modes, powers, **args)
    prepare_probes = spectrum._hcf_probes

    def without_reuse(get_paths):
        spectrum.clear_hcf_probe_cache()
        return prepare_probes(get_paths)

    with patch.object(spectrum, "_hcf_probes", side_effect=without_reuse):
        recomputed = spectrum.rank_hcf_first_channels(candidate_paths, modes, powers, **args)
    assert cached == recomputed
