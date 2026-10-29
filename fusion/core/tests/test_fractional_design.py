"""Real allocation and accounting regressions for exact imported bandwidths."""

import math
from pathlib import Path

import numpy as np
import pytest

from fusion.cli.config_setup import load_config
from fusion.core.simulation import SimulationEngine
from fusion.domain.request import Request
from fusion.io.generate import create_pt
from fusion.utils.bandwidth import bandwidth_key, parse_bandwidth


@pytest.mark.parametrize("mode", ["static", "semi_static", "dynamic"])
@pytest.mark.parametrize("orchestrator", [False, True])
def test_fractional_requests_allocate_and_remain_or_release(mode, orchestrator):
    template = Path(__file__).resolve().parents[2] / "configs/templates/static_design.ini"
    props = load_config(str(template))["s1"]
    props.update(
        traffic_type=mode,
        request_sort="none",
        date="test",
        sim_start="fractional-design",
        thread_num="s1",
        erlang=1,
        arrival_rate=1,
        num_requests=3,
        request_distribution={"44.5": 1.0},
        mod_per_bw={"44.5": {"QPSK": {"max_length": 1000, "slots_needed": 2}}},
        snr_type=None,
        save_snapshots=False,
        use_orchestrator=orchestrator,
        holding_time=1,
        topology_info=create_pt(1, {("A", "B"): 10}),
        core_nodes=["A", "B"],
    )
    engine = SimulationEngine(props)
    engine.create_topology()
    engine.stats_obj.init_iter_stats()
    engine.generate_requests(1)
    engine._process_all_requests()
    assert engine.stats_obj.bit_rate_request == 133.5
    assert engine.stats_obj.blocked_requests == 0
    if mode != "dynamic":
        engine._record_design_utilization()
        assert len(engine.reqs_status_dict) == 3
        assert all(link["throughput"] == 133.5 for link in engine.network_spectrum_dict.values())
        assert engine.stats_obj.stats_props.lp_bw_utilization_dict["overall"] == [100.0] * 3
    else:
        assert all(not np.any(link["cores_matrix"]["c"]) for link in engine.network_spectrum_dict.values())


def test_fractional_typed_request_roundtrip():
    data = {"req_id": 1, "source": "A", "destination": "B", "arrive": 0, "depart": math.inf, "bandwidth": "44.5"}
    request = Request.from_legacy_dict((1, 0), data)
    assert request.bandwidth_gbps == 44.5
    assert request.to_legacy_dict()["bandwidth"] == "44.5Gbps"
    assert parse_bandwidth("100.0") == 100
    assert bandwidth_key(44.5) == "44.5"


def test_fractional_grooming_preserves_remaining_capacity():
    from fusion.core.grooming import Grooming
    from fusion.core.properties import SDNProps

    sdn = SDNProps()
    sdn.source, sdn.destination, sdn.bandwidth, sdn.request_id, sdn.arrive = "A", "B", "44.5", 2, 0
    lp = {
        "path": ["A", "B"],
        "remaining_bandwidth": 55.5,
        "lightpath_bandwidth": 100,
        "requests_dict": {1: 44.5},
        "time_bw_usage": {0: 44.5},
        "core": 0,
        "band": "c",
        "start_slot": 0,
        "end_slot": 2,
        "mod_format": "QPSK",
        "snr_cost": 0,
        "xt_cost": 0,
        "path_weight": 10,
    }
    sdn.lightpath_status_dict = {("A", "B"): {1: lp}}
    grooming = Grooming({"is_grooming_enabled": True}, sdn)
    assert grooming._end_to_end_grooming()
    assert lp["remaining_bandwidth"] == 11
    assert lp["requests_dict"][2] == 44.5
