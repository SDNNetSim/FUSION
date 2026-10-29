# TopologyBench topology import

FUSION can load the real-network Excel workbooks in
[TopologyBench v4](https://zenodo.org/records/13921775), either individually or
directly from `real_topologies.zip`. This works with dynamic, static and
semi-static traffic. Built-in network selection is unchanged when
`topology_file` is empty.

## Download and run

From the repository root, using the Python environment installed for FUSION:

```powershell
python -m fusion.io.topologybench download data/raw/topologybench-real.zip
python -m fusion.io.topologybench list data/raw/topologybench-real.zip
python -m fusion.io.topologybench inspect data/raw/topologybench-real.zip --name TOP_1_ABILENE
python -m fusion.cli.run_sim run_sim --config_path fusion/configs/templates/topologybench_static.ini --run_id topologybench-static
```

The download command uses the pinned v4 record, verifies SHA-256, and never
overwrites an existing file. Download once; simulations read local files and
make no network requests. The archive is not bundled with FUSION.

Use `--traffic_type semi_static` or `--traffic_type dynamic` to change the
traffic model. Dynamic traffic also needs `--request_sort none`, since the
example enables bandwidth ordering for permanent demands.

## Configuration

```ini
[topology_settings]
network = TopologyBench-ABILENE
topology_file = ../../../data/raw/topologybench-real.zip
topology_name = TOP_1_ABILENE
const_link_weight = False
is_only_core_node = True
```

Add these settings to a complete FUSION config; the supplied
`fusion/configs/templates/topologybench_static.ini` is ready after downloading.
`topology_file` is relative to the **INI file's directory**, including CLI
overrides. Absolute paths also work. `network` names the simulation output
directory and may be a custom label; it does not select an archive member.

`topology_name` selects the exact workbook filename or stem inside a ZIP (for
example `TOP_1_ABILENE.xlsx` or `TOP_1_ABILENE`). Bare names like `CANARIE` are
not accepted because the dataset includes multiple versions of some networks.
Omit the selector for a single `.xlsx` or an archive with one workbook. For a
workbook path, a supplied selector must match its filename or stem. Both
options support CLI and per-experiment `[s2]` overrides.

## Data interpretation

- The `Nodes_*` and `Edges_*` sheets define the graph. Auxiliary location
  sheets and extra lookup columns are ignored.
- Node IDs are preserved as strings. Numeric Excel IDs become integer strings
  when applicable, so numeric `1` matches text `"1"`; text `"03"` stays `"03"`.
  Latitude, longitude, location name and country are retained in topology info.
- `Computed Length (km)` is used directly, without rounding, rescaling or
  recalculation from coordinates. These published lengths are estimates of
  fibre distance; see the paper for their derivation.
- Links are simple undirected physical connections. FUSION's existing
  directional spectrum handling and configured fibre parameters still apply.
  This dataset does not supply FUSION transponder/modulation definitions or a
  traffic matrix. All imported nodes are eligible core nodes.
- Keep `const_link_weight = False` for physical-distance experiments. The
  legacy `True` setting replaces each link length with `1.0`, including lengths
  used by physical-layer calculations, not just routing weights.
- The input properties include `topology_import_summary`: selected file and
  member, workbook SHA-256, node/link counts, original total length and whether
  constant weights were requested. A `topology_import_s1.json` file containing
  this summary and the complete physical topology is also saved under
  `data/input/<network>/<date>/<run timestamp>/` (with the corresponding thread
  suffix). This snapshot is written by both the batch CLI and legacy runner.

To combine an imported topology with a traffic matrix, use the workbook's
`Node_ID` values for demand endpoints, or set `traffic_node_map_file` to an
explicit mapping. Matching city names alone does not establish a mapping.
Fractional traffic bandwidths remain unchanged and require matching modulation
definitions. See [traffic-design.md](traffic-design.md).

## Validation and scope

The importer rejects missing/invalid lengths, unknown endpoints, duplicate node
IDs, self-links, parallel/repeated undirected edges, and disconnected networks.
It does not remove nodes or collapse links automatically. Formula cells need
cached numeric values, as in the published workbooks.

All 105 v4 real workbooks were checked: 98 load directly. Seven require an
explicitly reviewed, edited copy before simulation:

| Workbook | Source-data issue |
| --- | --- |
| TOP_13_CERNET | Repeated edges, including different lengths for one pair |
| TOP_27_DT17 | Two lengths for the same node pair |
| TOP_36_GEANT2 | Repeated undirected edge |
| TOP_62_LONI | Repeated undirected edge |
| TOP_77_OXFORD | Zero-length self-link |
| TOP_80_PIONIER | Repeated undirected edge |
| TOP_86_REDCLARA | Repeated undirected edge |

Synthetic pickle/CSV collections are not supported by this importer. It never
deserializes pickle files. `openpyxl` is required and already appears in
FUSION's `requirements.txt`.

## Reference

Robin Matzner et al., *Topology Bench: systematic graph-based benchmarking for
core optical networks*, Journal of Optical Communications and Networking 17(1),
7–27. [Dataset and citation](https://zenodo.org/records/13921775),
[project repository](https://github.com/TopologyBench/topologybench.github.io).
