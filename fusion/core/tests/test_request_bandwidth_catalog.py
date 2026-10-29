"""Traffic mixes can use a subset of a fixed-grid modulation catalog."""

from collections import Counter
from copy import deepcopy

import pytest

from fusion.core.request import generate_simulation_requests


@pytest.fixture
def properties():
    return {
        "is_only_core_node": True,
        "topology_info": {"nodes": {"A": {}, "B": {}, "C": {}}},
        "arrival_rate": 1.0,
        "holding_time": 10.0,
        "num_requests": 10,
        "fixed_grid": True,
        "bw_per_slot": 75,
        "c_band": 80,
        "request_distribution": {"400": 0.1, "100": 0.5, "200": 0.4},
        "mod_per_bw": {
            rate: {"64-QAM": {"max_length": 400, "slots_needed": 1}}
            for rate in ("100", "200", "300", "400", "500", "600")
        },
    }


def test_extra_catalog_rates_do_not_change_requested_mix_or_seeded_trace(properties):
    subset = deepcopy(properties)
    subset["mod_per_bw"] = {rate: modes for rate, modes in subset["mod_per_bw"].items() if rate in subset["request_distribution"]}
    expected = generate_simulation_requests(7, subset)
    before = deepcopy(properties)
    actual = generate_simulation_requests(7, properties)

    assert actual == expected
    assert properties == before
    arrivals = [event for event in actual.values() if event["request_type"] == "arrival"]
    assert Counter(event["bandwidth"] for event in arrivals) == {"100": 5, "200": 4, "400": 1}
    assert len(actual) == 20
    for arrival in arrivals:
        release = actual[(arrival["req_id"], arrival["depart"])]
        assert release["request_type"] == "release"
        assert release["bandwidth"] == arrival["bandwidth"]
        assert release["mod_formats"] == properties["mod_per_bw"][arrival["bandwidth"]]


def test_requested_rate_missing_from_catalog_reports_configuration_error(properties):
    properties["mod_per_bw"].pop("200")
    with pytest.raises(ValueError, match=r"request_distribution bandwidths missing from mod_per_bw: \['200'\]"):
        generate_simulation_requests(7, properties)


def test_explicit_zero_probability_rates_generate_no_requests(properties):
    properties["request_distribution"].update({"300": 0.0, "500": 0.0, "600": 0.0})
    events = generate_simulation_requests(7, properties)
    arrivals = [event for event in events.values() if event["request_type"] == "arrival"]
    assert Counter(event["bandwidth"] for event in arrivals) == {"100": 5, "200": 4, "400": 1}
