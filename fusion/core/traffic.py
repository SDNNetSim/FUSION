"""Permanent-demand generation and ordering for network design experiments."""

import math
from typing import Any

import networkx as nx
import numpy as np

from fusion.io.traffic_matrix import prepare_traffic_matrix

TRAFFIC_TYPES = ("dynamic", "static", "semi_static")
SORT_METRICS = ("none", "bandwidth", "shortest_path", "hop_count", "bandwidth_distance")


def validate_traffic_options(properties: dict[str, Any]) -> None:
    """Reject invalid modes/orderings without changing legacy dynamic defaults."""
    mode = properties.get("traffic_type", "dynamic")
    metric = properties.get("request_sort", "none")
    if mode not in TRAFFIC_TYPES:
        raise ValueError(f"traffic_type must be one of {TRAFFIC_TYPES}")
    if metric not in SORT_METRICS:
        raise ValueError(f"request_sort must be one of {SORT_METRICS}")
    if properties.get("request_sort_order", "ascending") not in ("ascending", "descending"):
        raise ValueError("request_sort_order must be ascending or descending")
    _positive(properties.get("traffic_matrix_scale", 1.0), "traffic_matrix_scale")
    if properties.get("traffic_matrix_direction", "directed") not in ("directed", "undirected"):
        raise ValueError("traffic_matrix_direction must be directed or undirected")
    if mode == "dynamic" and metric != "none":
        raise ValueError("Request sorting is supported only for static and semi_static traffic")
    if mode == "static" and properties.get("failure_type", "none") != "none":
        raise ValueError("Time-based failures require dynamic or semi_static traffic; all static arrivals occur at time zero")


def _positive(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _generated_demands(properties: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    nodes = list(properties["topology_info"]["nodes"]) if properties["is_only_core_node"] else list(properties["core_nodes"])
    if len(set(nodes)) < 2:
        raise ValueError("Traffic generation requires at least two distinct eligible nodes")
    count = properties["num_requests"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("num_requests must be a positive integer")
    distribution = properties["request_distribution"]
    if not distribution or any(not math.isfinite(float(p)) or p < 0 for p in distribution.values()):
        raise ValueError("request_distribution must contain finite, non-negative fractions")
    if not math.isclose(sum(distribution.values()), 1.0):
        raise ValueError("request_distribution fractions must sum to one")
    missing = set(distribution) - set(properties["mod_per_bw"])
    if missing:
        raise ValueError(f"request_distribution bandwidths missing from mod_per_bw: {sorted(missing)}")
    # Largest-remainder allocation supports small design workloads without losing demands.
    bands = list(distribution)
    exact = [distribution[bw] * count for bw in bands]
    counts = [math.floor(value) for value in exact]
    remainder_order = sorted(range(len(bands)), key=lambda i: exact[i] - counts[i], reverse=True)
    for index in remainder_order[: count - sum(counts)]:
        counts[index] += 1
    bandwidths = [bw for bw, amount in zip(bands, counts, strict=True) for _ in range(amount)]
    rng.shuffle(bandwidths)
    rows = []
    for bandwidth in bandwidths:
        source, destination = rng.choice(len(nodes), 2, replace=False)
        rows.append({"source": nodes[source], "destination": nodes[destination], "bandwidth": bandwidth})
    return rows


def _sort_demands(rows: list[dict[str, Any]], properties: dict[str, Any], timed: bool) -> list[dict[str, Any]]:
    metric = properties.get("request_sort", "none")
    if metric != "none":
        graph = properties.get("topology")
        if metric != "bandwidth" and graph is None:
            raise ValueError("Path-based request sorting requires the initialized topology")
        distances: dict[tuple[Any, Any], float] = {}

        def score(row: dict[str, Any]) -> float:
            bandwidth = _positive(row["bandwidth"], "Request bandwidth")
            if metric == "bandwidth":
                return bandwidth
            pair = (row["source"], row["destination"])
            if pair not in distances:
                try:
                    distances[pair] = nx.shortest_path_length(graph, *pair, weight=None if metric == "hop_count" else "length")
                except nx.NetworkXNoPath:
                    # Still offer disconnected demands to the simulator; they count as blocks.
                    distances[pair] = math.inf
            distance = distances[pair]
            return distance * bandwidth if metric == "bandwidth_distance" else distance

        rows = sorted(rows, key=score, reverse=properties.get("request_sort_order", "ascending") == "descending")
    # Stable sorting retains metric order within a timestamp while preserving chronology.
    if timed:
        rows = sorted(rows, key=lambda row: row["arrival"])
    return rows


def generate_design_requests(seed: int, properties: dict[str, Any]) -> dict[tuple[int, float], dict[str, Any]]:
    """Generate static/semi-static events, or dynamic events from an imported workload.

    Permanent requests have depart=inf and no release event. Generate timestamps
    before sorting so ordering never changes semi-static demand arrival times.
    Metrics only break ties between demands arriving at the same time.
    """
    from fusion.core.request import _create_request_entry

    validate_traffic_options(properties)
    prepare_traffic_matrix(properties)
    rng = np.random.default_rng(seed)
    rows = (
        [dict(row) for row in properties["traffic_demands"]]
        if properties.get("traffic_matrix_file")
        else _generated_demands(properties, rng)
    )
    mode = properties.get("traffic_type", "dynamic")
    timed = "arrival" in rows[0]
    if mode == "static" and timed and any(row["arrival"] != 0 for row in rows):
        raise ValueError("Static traffic requires all imported arrival times to be zero (or omitted)")
    rate = _positive(properties["arrival_rate"], "arrival_rate") if mode != "static" and not timed else None
    holding = _positive(properties["holding_time"], "holding_time") if mode == "dynamic" else None
    current_time = 0.0
    for row in rows:
        if mode == "static":
            arrival = 0.0
        elif timed:
            arrival = row["arrival"]
        else:
            arrival = max(current_time + float(rng.exponential(1 / rate)), math.nextafter(current_time, math.inf))
        if not math.isfinite(arrival):
            raise ValueError("Arrival time overflow; increase arrival_rate")
        current_time = arrival
        row["arrival"] = arrival
    rows = _sort_demands(rows, properties, timed=True)
    properties["request_counts"] = {bw: sum(row["bandwidth"] == bw for row in rows) for bw in properties["mod_per_bw"]}
    events = {}
    for request_id, row in enumerate(rows, 1):
        arrival = row["arrival"]
        departure = max(arrival + float(rng.exponential(holding)), math.nextafter(arrival, math.inf)) if holding else math.inf
        if holding and not math.isfinite(departure):
            raise ValueError("Departure time overflow; decrease holding_time")
        event = _create_request_entry(
            request_id,
            row["source"],
            row["destination"],
            arrival,
            departure,
            "arrival",
            row["bandwidth"],
            properties["mod_per_bw"][row["bandwidth"]],
        )
        events[(request_id, arrival)] = event
        if mode == "dynamic":
            events[(request_id, departure)] = dict(event, request_type="release")
    return events
