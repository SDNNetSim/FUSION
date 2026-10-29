# Static and semi-static traffic design

Add these optional settings to `[general_settings]` in any existing simulation INI:

```ini
traffic_type = static
request_sort = bandwidth
request_sort_order = descending
```

Omitting the settings retains the existing seeded dynamic generator. The new generator
uses a separate reproducible random stream. Changing the traffic mode does not promise
the same random source/destination pairs as a dynamic run.

| Traffic type | Arrival times | Departures |
| --- | --- | --- |
| `dynamic` (default) | Existing exponential arrivals | Existing exponential holding times |
| `static` | All at time zero | Infinite lifetime; no release events |
| `semi_static` | Exponential arrivals, or times in an imported file | Infinite lifetime; no release events |

The run ends after processing its finite event set. Permanent connections remain allocated
until the iteration resets. Do **not** set `holding_time = inf`: keep the existing positive
finite setting because the runner uses `cores_per_link * erlang / holding_time` to calculate
the arrival rate. For semi-static runs it controls spacing, not connection lifetime.
For static runs Erlang and holding time do not change the offered demand set. Use
`num_requests` or an imported matrix to change design load. A single Erlang point avoids
repeating identical static experiments. Existing max-iteration and confidence settings apply.

## Ordering

`request_sort` supports the following metrics. `request_sort_order` is `ascending` or
`descending`; ties retain input/generated order. Sorting is rejected for dynamic traffic.

| Metric | Meaning |
| --- | --- |
| `none` (default) | Keep original order within each arrival time |
| `bandwidth` | Numeric demand bandwidth, small to large or large to small |
| `shortest_path` | Shortest path weighted by the topology's `length` attribute |
| `hop_count` | Fewest hops in the physical topology |
| `bandwidth_distance` | Bandwidth multiplied by shortest-path length |

**Arrival times always take precedence.** Static demands all share time zero, so the
metric orders the entire workload. Semi-static sorting only orders simultaneous arrivals.
Generated exponential arrivals are distinct, so selecting a sort metric does not reorder
them. Imported timestamps remain attached to their original demands. Request IDs are
assigned in processing order. Sorting uses the initial physical topology, not congestion
or a topology modified by subsequent failures. Unreachable pairs remain in the workload
and receive an infinite path score; the simulator can then count their blocking.

Generated design workloads use largest-remainder allocation for bandwidth percentages,
so small demand sets can use distributions that do not divide evenly. Fractions must sum
to one. Counts, node choices, and stable ties are reproducible for a fixed seed.

## Importing traffic

```ini
traffic_matrix_file = demands.csv
traffic_matrix_format = auto
traffic_matrix_scale = 1.0
traffic_node_map_file =
```

The file replaces random node/bandwidth generation. Its expanded demand count overrides
`num_requests`, and its bandwidth distribution feeds the statistics. The prepared demand
list is cached in the engine properties and saved by the input preparation path. The importer
does not fetch changing remote data during simulation.

Paths in the CLI INI loader are relative to the INI's directory; absolute paths also work.
Direct Python callers should provide absolute paths. CLI overrides exist for every setting,
including when an older INI does not contain the option. `[s2]`, `[s3]`, etc. can override them.

Supported formats:

* `csv`: headers `source,destination,bandwidth`; optional `count` and `arrival` columns.
* `json`: an array of equivalent records, or an object containing a `demands` array.
* `matrix_csv`: square matrix with row/column node labels. Zero cells produce no demand.
* `matrix_xlsx`: labelled Excel matrix, with `traffic_matrix_sheet` selecting a worksheet.
  Blank cells mean zero; trailing empty formatting rows are ignored.
* `sndlib`: SNDlib network XML, using `demandValue * routingUnit`.
* `auto`: infer CSV demand records, JSON, SNDlib XML, or Excel from the file extension.
  Square CSV matrices must explicitly select `matrix_csv`.

Demand records, including repeated pairs, are independent requests:

```csv
source,destination,bandwidth,count,arrival
1,2,100,2,0
2,3,44.5,1,10
```

```json
{"demands": [
  {"source": "1", "destination": "2", "bandwidth": 100},
  {"source": "2", "destination": "3", "bandwidth": 44.5}
]}
```

```csv
node,1,2,3
1,0,100,0
2,100,0,44.5
3,0,44.5,0
```

For matrix formats, `traffic_matrix_direction = directed` (default) reads every nonzero
cell. `undirected` requires symmetry and reads each pair once. Choose this explicitly for
a bidirectional demand represented twice in a symmetric matrix; FUSION allocates resources
in both physical directions. Record-based formats use each record exactly as provided.

Every record must either supply a non-negative finite `arrival`, or omit it. Static imports
reject nonzero arrivals. Dynamic imports use finite sampled holding times. Both static and
semi-static imports omit departure events entirely.

Bandwidth values use the units of `mod_per_bw` (normally Gbps). Scaling is explicit.
**There is no bandwidth-class rounding or splitting.** Fractional values such as `44.5`
remain fractional through request creation, allocation, grooming and blocking accounting.
Every imported value must have a matching bandwidth entry in the configured modulation
definitions, including its validated reach and slot requirements. The importer fails on
missing definitions rather than inventing physical-layer assumptions. Numerically integral
values use integer keys (`100.0` becomes `100`); fractional values retain their loaded
floating-point precision, including any representation noise already present in the source.

Node names must exist in the simulation topology. A JSON node-map file can translate names:

```json
{"Paris": "1", "London": "2"}
```

Self-demands, missing nodes, unsupported bandwidths, invalid amounts/counts, inconsistent
arrival fields, malformed matrices, and asymmetric undirected matrices produce errors.
Explicit imports select their own endpoints; the core-node filter applies to random generation.

## TEFNET24 from Zenodo

The selected [versioned TEFNET24 record](https://zenodo.org/records/13362934) includes
`Traffic_Matrix_20240822.xlsx`. Download a local copy from the repository root:

```powershell
python -m fusion.io.traffic_matrix "https://zenodo.org/records/13362934/files/Traffic_Matrix_20240822.xlsx?download=1" "TEFNET24-traffic.xlsx"
```

Downloads require HTTPS, have a timeout and a 20 MiB limit, and never overwrite an existing
file. For another open-source repository, use a raw-file URL pinned to a commit.

Select either dataset worksheet in your simulation configuration:

```ini
traffic_type = static
traffic_matrix_file = ../../../TEFNET24-traffic.xlsx
traffic_matrix_format = matrix_xlsx
traffic_matrix_sheet = HL1-HL2-HL3 traffic matrix
traffic_matrix_direction = undirected
traffic_matrix_scale = 1.0
request_sort = bandwidth
request_sort_order = descending
```

The path above assumes the INI lives in `fusion/configs/templates`. For metro traffic use
`traffic_matrix_sheet = HL4-HL3 Traffic Matrix`. The published workbook was checked with
both directed and undirected reads. Its padded rows and empty cells are supported.

These settings import demands; they do **not** substitute TEFNET24 for the configured physical
topology or create modulation definitions. Use the corresponding topology and physical-layer
assumptions for the experiment. The workbook's router IDs can include hierarchy suffixes such
as `_HL3`; retain the appropriate mapping to your chosen topology. Mapping two endpoints onto
one physical site is rejected as a self-demand; explicitly preprocess such local traffic if
your experiment excludes it. Verify the source units before choosing a scale.

The [SNDlib XML schema documentation](https://sndlib.put.poznan.pl/html/docu/io-formats/xml/http___sndlib.zib.de_network/index.html)
describes the other supported public-network format. Dataset topology and units are always
the experiment author's responsibility.

## Running the included example

`fusion/configs/templates/static_design.ini` is a complete NSFNet example using modulation
reach limits and a single load point. SNR checking is disabled in this example; enable a
validated physical-layer configuration for your experiment. From the repository root:

```powershell
python -m fusion.cli.run_sim run_sim --config_path fusion/configs/templates/static_design.ini --run_id static-design
python -m fusion.cli.run_sim run_sim --config_path fusion/configs/templates/static_design.ini --run_id semi-static-design --traffic_type semi_static
```

Design utilization is a final snapshot of occupied lightpath capacity, and link throughput
is final carried Gbps. Dynamic release-based time averages and Gbps-seconds remain unchanged.
Time-based failure injection requires dynamic or semi-static traffic; static arrivals all
occur at time zero and cannot define separate failure and repair times.

The CLI and engine APIs support these settings. The existing graphical configuration editor
does not yet provide dedicated controls for them.
