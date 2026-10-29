"""Import the real-network Excel dataset published by TopologyBench.

Workbooks can be read directly or selected inside the original ZIP, without
extracting files. Published computed fibre lengths are already in kilometres.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import networkx as nx

DATASET_URL = "https://zenodo.org/records/13921775/files/real_topologies.zip"
DATASET_SHA256 = "f2ad31ecc53378dd8992beedfe93141454bc1915257bd78e4e0dca3484ed45c8"
MAX_FILE_BYTES = 20 * 1024 * 1024


@dataclass
class ImportedTopology:
    """Source node attributes, undirected edges in km, and input provenance."""

    nodes: dict[str, dict[str, Any]]
    links: dict[tuple[str, str], float]
    summary: dict[str, Any]


def _members(archive: ZipFile) -> list[str]:
    return sorted(name for name in archive.namelist() if name.lower().endswith(".xlsx") and not name.startswith("__MACOSX/"))


def list_topologies(path: str | Path) -> list[str]:
    """Return exact workbook member names for unambiguous selection."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return [path.name]
    if path.suffix.lower() != ".zip":
        raise ValueError("TopologyBench import supports real-network .xlsx or .zip files; pickle is not supported")
    with ZipFile(path) as archive:
        return _members(archive)


def _workbook_bytes(path: Path, name: str) -> tuple[bytes, str]:
    if path.suffix.lower() == ".xlsx":
        if name and name not in (path.name, path.stem):
            raise ValueError(f"topology_name {name!r} does not match workbook {path.name!r}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Topology workbook exceeds 20 MiB")
        return path.read_bytes(), path.name
    if path.suffix.lower() != ".zip":
        raise ValueError("TopologyBench import supports real-network .xlsx or .zip files; pickle is not supported")
    with ZipFile(path) as archive:
        names = _members(archive)
        matches = [member for member in names if name in (member, Path(member).name, Path(member).stem)] if name else names
        if len(matches) != 1:
            raise ValueError(
                "Select exactly one workbook using topology_name; list choices with python -m fusion.io.topologybench list PATH"
            )
        member = matches[0]
        if archive.getinfo(member).file_size > MAX_FILE_BYTES:
            raise ValueError("Topology workbook exceeds 20 MiB")
        return archive.read(member), member


def _node_id(value: Any) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError("Missing or invalid topology node ID")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Topology node ID must be finite")
        value = int(value) if float(value).is_integer() else value
    result = str(value).strip()
    if not result:
        raise ValueError("Topology node ID cannot be empty")
    return result


def _read_nodes(sheet: Any) -> dict[str, dict[str, Any]]:
    rows = iter(sheet.values)
    headers = list(next(rows))
    if headers[:5] != ["Node_ID", "Latitude", "Longitude", "Location Name", "Country"]:
        raise ValueError(f"Unexpected node columns in {sheet.title!r}")
    nodes = {}
    for number, row in enumerate(rows, 2):
        if all(value is None for value in row[:5]):
            continue
        node = _node_id(row[0])
        if node in nodes:
            raise ValueError(f"Duplicate node ID {node!r} on row {number}")
        attributes = {}
        for key, value in zip(("latitude", "longitude", "location_name", "country"), row[1:5], strict=True):
            if value is None:
                continue
            if key in ("latitude", "longitude"):
                limit = 90 if key == "latitude" else 180
                value = float(value)
                if not math.isfinite(value) or abs(value) > limit:
                    raise ValueError(f"Invalid {key} for node {node!r}")
            else:
                value = str(value)
            attributes[key] = value
        nodes[node] = attributes
    return nodes


def _read_links(sheet: Any, nodes: dict[str, dict[str, Any]]) -> dict[tuple[str, str], float]:
    rows = iter(sheet.values)
    headers = list(next(rows))
    if headers[:4] != ["Edge_ID", "Source", "Destination", "Computed Length (km)"]:
        raise ValueError(f"Unexpected edge columns in {sheet.title!r}")
    links = {}
    seen = set()
    for number, row in enumerate(rows, 2):
        # Extra location lookup columns in some published sheets are not edges.
        if all(value is None for value in row[:4]):
            continue
        source, destination = _node_id(row[1]), _node_id(row[2])
        if source not in nodes or destination not in nodes:
            raise ValueError(f"Unknown edge endpoint on row {number}: {source!r}, {destination!r}")
        if source == destination:
            raise ValueError(f"Self-link on row {number} is not supported")
        pair = frozenset((source, destination))
        if pair in seen:
            raise ValueError(f"Duplicate/parallel undirected edge on row {number}: {source!r}, {destination!r}")
        try:
            length = float(row[3])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Missing or invalid computed length on edge row {number}") from error
        if isinstance(row[3], bool) or not math.isfinite(length) or length <= 0:
            raise ValueError(f"Computed length must be positive and finite on edge row {number}")
        links[(source, destination)] = length
        seen.add(pair)
    return links


def load_topologybench(path: str | Path, name: str = "", constant_weight: bool = False) -> ImportedTopology:
    """Load one connected, simple undirected topology, preserving source IDs.

    Only Nodes_* and Edges_* sheets are interpreted. Auxiliary sheets/columns are
    ignored. Missing formula caches and invalid/parallel edges are rejected rather
    than silently changing the network. No pickle objects are deserialized.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as error:
        raise ImportError("TopologyBench Excel import requires openpyxl; install it with pip install openpyxl") from error
    path = Path(path).expanduser().resolve()
    data, member = _workbook_bytes(path, name)
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        node_sheets = [sheet for sheet in workbook if sheet.title.startswith("Nodes_")]
        edge_sheets = [sheet for sheet in workbook if sheet.title.startswith("Edges_")]
        if len(node_sheets) != 1 or len(edge_sheets) != 1:
            raise ValueError("Expected exactly one Nodes_* sheet and one Edges_* sheet")
        if node_sheets[0].title[6:] != edge_sheets[0].title[6:]:
            raise ValueError("Node and edge sheets name different networks")
        nodes = _read_nodes(node_sheets[0])
        links = _read_links(edge_sheets[0], nodes)
    finally:
        workbook.close()
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(links)
    if len(nodes) < 2 or not nx.is_connected(graph):
        raise ValueError("FUSION requires a connected topology with at least two nodes; isolated/disconnected nodes are not removed")
    summary = {
        "source": "TopologyBench",
        "dataset_url": "https://zenodo.org/records/13921775",
        "file": str(path),
        "workbook": member,
        "workbook_sha256": hashlib.sha256(data).hexdigest(),
        "num_nodes": len(nodes),
        "num_links": len(links),
        "source_length_unit": "km",
        "source_total_length_km": sum(links.values()),
        "constant_weight": constant_weight,
    }
    if constant_weight:
        links = dict.fromkeys(links, 1.0)
    return ImportedTopology(nodes, links, summary)


def download_topologybench(destination: str | Path) -> Path:
    """Download the pinned real-network archive and verify it before saving."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with urllib.request.urlopen(DATASET_URL, timeout=30) as response:  # nosec B310: fixed HTTPS URL
        data = response.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES or hashlib.sha256(data).hexdigest() != DATASET_SHA256:
        raise ValueError("TopologyBench download failed its size/checksum check; no file was saved")
    with destination.open("xb") as output:
        output.write(data)
    return destination


def main() -> None:
    """Download, list or inspect dataset files without starting a simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "list", "inspect"))
    parser.add_argument("path", help="Local workbook/ZIP, or new destination for download")
    parser.add_argument("--name", default="", help="Exact workbook filename or stem inside the archive")
    args = parser.parse_args()
    if args.command == "download":
        print(download_topologybench(args.path))
    elif args.command == "list":
        print("\n".join(list_topologies(args.path)))
    else:
        print(json.dumps(load_topologybench(args.path, args.name).summary, indent=2))


if __name__ == "__main__":
    main()
