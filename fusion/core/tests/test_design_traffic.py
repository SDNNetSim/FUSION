"""Design traffic behavior and compatibility with the event consumers."""

import json
import math
from unittest.mock import Mock

import networkx as nx
import pytest

from fusion.core.request import generate_simulation_requests
from fusion.core.simulation import SimulationEngine
from fusion.domain.request import Request


@pytest.fixture
def props():
    topology = nx.Graph()
    topology.add_edge("A", "B", length=5)
    topology.add_edge("B", "C", length=10)
    topology.add_edge("A", "C", length=100)
    return {
        "traffic_type": "static",
        "num_requests": 9,
        "request_distribution": {"100": 0.5, "400": 0.5},
        "mod_per_bw": {"100": {}, "400": {}},
        "is_only_core_node": True,
        "topology_info": {"nodes": {"A": {}, "B": {}, "C": {}}},
        "topology": topology,
        "arrival_rate": 2.0,
        "holding_time": 10,
    }


def test_static_keeps_all_simultaneous_demands_without_releases(props):
    events = generate_simulation_requests(7, props)
    assert len(events) == 9
    assert len({key[0] for key in events}) == 9
    assert all(key[1] == 0 for key in events)
    assert all(row["depart"] == math.inf and row["request_type"] == "arrival" for row in events.values())
    assert sum(row["bandwidth"] == "100" for row in events.values()) == 5
    assert events == generate_simulation_requests(7, props)
    assert events != generate_simulation_requests(8, props)
    # The orchestrator's typed request adapter accepts permanent requests.
    assert all(Request.from_legacy_dict(key, row).depart_time == math.inf for key, row in events.items())


def test_semi_static_has_increasing_arrivals_and_no_releases(props):
    props.update(traffic_type="semi_static", request_sort="bandwidth", request_sort_order="descending")
    events = generate_simulation_requests(7, props)
    arrivals = [row["arrive"] for row in events.values()]
    assert all(a < b for a, b in zip(arrivals, arrivals[1:], strict=False))
    props["request_sort"] = "none"
    assert events == generate_simulation_requests(7, props)
    assert all(math.isinf(row["depart"]) for row in events.values())


@pytest.mark.parametrize("metric", ["bandwidth", "shortest_path", "hop_count", "bandwidth_distance"])
@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_sorting_metrics(props, metric, order):
    props.update(request_sort=metric, request_sort_order=order)
    rows = list(generate_simulation_requests(7, props).values())

    def score(row):
        bw = float(row["bandwidth"])
        if metric == "bandwidth":
            return bw
        distance = nx.shortest_path_length(
            props["topology"], row["source"], row["destination"], weight=None if metric == "hop_count" else "length"
        )
        return distance * bw if metric == "bandwidth_distance" else distance

    scores = [score(row) for row in rows]
    assert scores == sorted(scores, reverse=order == "descending")


def test_imported_arrivals_preserved_and_ties_sorted(props, tmp_path):
    path = tmp_path / "demands.json"
    path.write_text(
        json.dumps(
            [
                {"source": "A", "destination": "B", "bandwidth": 100, "arrival": 9},
                {"source": "A", "destination": "C", "bandwidth": 400, "arrival": 9},
                {"source": "B", "destination": "C", "bandwidth": 100, "arrival": 2},
            ]
        )
    )
    props.update(traffic_type="semi_static", traffic_matrix_file=str(path), request_sort="bandwidth", request_sort_order="descending")
    rows = list(generate_simulation_requests(7, props).values())
    assert [(row["arrive"], row["bandwidth"]) for row in rows] == [(2, "100"), (9, "400"), (9, "100")]
    assert props["num_requests"] == 3
    assert props["request_distribution"] == {"100": 2 / 3, "400": 1 / 3}
    props["traffic_type"] = "static"
    with pytest.raises(ValueError, match="Static traffic"):
        generate_simulation_requests(7, props)


def test_default_and_explicit_dynamic_are_identical(props):
    props["num_requests"] = 10
    props.pop("traffic_type")
    implicit = generate_simulation_requests(7, props)
    props["traffic_type"] = "dynamic"
    assert implicit == generate_simulation_requests(7, props)
    assert len(implicit) == 20
    assert sum(row["request_type"] == "release" for row in implicit.values()) == 10


def test_dynamic_matrix_has_finite_releases(props, tmp_path):
    path = tmp_path / "demands.csv"
    path.write_text("source,destination,bandwidth,count\nA,B,100,2\n")
    props.update(traffic_type="dynamic", traffic_matrix_file=str(path))
    events = generate_simulation_requests(7, props)
    assert len(events) == 4
    assert all(math.isfinite(row["depart"]) and row["depart"] > row["arrive"] for row in events.values())


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"traffic_type": "bad"}, "traffic_type"),
        ({"request_sort": "bad"}, "request_sort"),
        ({"request_sort_order": "bad"}, "request_sort_order"),
        ({"traffic_type": "dynamic", "request_sort": "bandwidth"}, "sorting"),
        ({"traffic_type": "semi_static", "arrival_rate": 0}, "arrival_rate"),
        ({"num_requests": 0}, "num_requests"),
        ({"request_distribution": {"100": -1, "400": 2}}, "non-negative"),
        ({"failure_type": "link"}, "Time-based failures"),
    ],
)
def test_reject_invalid_design_settings(props, updates, match):
    props.update(updates)
    with pytest.raises(ValueError, match=match):
        generate_simulation_requests(7, props)


def test_engine_and_rl_keep_metric_order(props):
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.engine_props = dict(props, request_sort="bandwidth", request_sort_order="descending")
    engine.generate_requests(7)
    engine.use_orchestrator = True
    engine._get_or_create_v5_request = lambda key: engine.reqs_dict[key]
    engine._rl_request_index = 0
    assert engine.get_next_request()["bandwidth"] == "400"
    engine.failure_manager = None
    engine.handle_request = Mock()
    engine._process_all_requests()
    assert engine.handle_request.call_count == props["num_requests"]
    assert [call.kwargs["current_time"] for call in engine.handle_request.call_args_list] == list(engine.reqs_dict)


def test_static_final_utilization_uses_occupied_capacity():
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.network_spectrum_dict = {}
    engine.use_orchestrator = False
    engine.sdn_obj = Mock()
    engine.sdn_obj.sdn_props.lightpath_status_dict = {
        ("A", "B"): {
            1: {
                "band": "c",
                "core": 0,
                "lightpath_bandwidth": 400,
                "remaining_bandwidth": 100,
            }
        }
    }
    engine.stats_obj = Mock()
    engine._record_design_utilization()
    engine.stats_obj.update_utilization_dict.assert_called_once_with(
        {
            1: {
                "band": "c",
                "core": 0,
                "bit_rate": 400,
                "utilization": 75,
            }
        }
    )


def test_failure_schedule_uses_float_times(props):
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.engine_props = dict(
        props,
        traffic_type="semi_static",
        failure_type="link",
        failed_link_src="A",
        failed_link_dst="B",
        t_fail_arrival_index=1,
        t_repair_after_arrivals=2,
    )
    engine.topology = props["topology"]
    engine.generate_requests(7)
    engine.failure_manager = Mock()
    engine.failure_manager.inject_failure.return_value = {"failed_links": [("A", "B")]}
    engine._schedule_failure()
    kwargs = engine.failure_manager.inject_failure.call_args.kwargs
    assert isinstance(kwargs["t_fail"], float)
    assert kwargs["t_fail"] < kwargs["t_repair"]
