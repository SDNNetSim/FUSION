# Run Directory Contract

Each simulation run managed by the GUI has a dedicated directory under `data/gui_runs/`. This document defines the exact structure and file formats.

## Directory Structure

```
data/gui_runs/
├── runs.db                     # SQLite database (run metadata)
└── {run_id}/                   # One directory per run
    ├── config.ini              # Frozen configuration (immutable after start)
    ├── logs/
    │   └── sim.log             # Combined stdout/stderr from simulation
    ├── progress.jsonl          # Structured progress events (machine-readable)
    ├── output/                 # Simulation output artifacts
    │   ├── {sim_info}/
    │   │   └── {thread_num}/
    │   │       └── {erlang}_erlang.json
    │   └── ...
    └── plots/                  # Generated visualizations (optional)
        └── ...
```

## File Specifications

### config.ini

The frozen configuration snapshot used for this run. Written at run creation time, never modified afterward.

```ini
[general_settings]
network = nsfnet
erlang_start = 50
erlang_stop = 200
erlang_step = 10
max_iters = 100
num_requests = 1000
holding_time = 60

[topology_settings]
k_paths = 3

[routing_settings]
route_method = shortest_path

[spectrum_settings]
c_band = true
; ... other settings
```

**Contract:**
- Created by: Backend at run creation
- Modified by: Never (immutable)
- Read by: Simulator subprocess, UI (for display)

### logs/sim.log

Combined stdout and stderr from the simulation process. Human-readable log output.

```
2024-01-15 10:00:00 INFO     Starting simulation with config: config.ini
2024-01-15 10:00:01 INFO     Loaded topology: nsfnet (14 nodes, 21 links)
2024-01-15 10:00:01 INFO     Erlang range: 50 to 200, step 10
2024-01-15 10:00:05 INFO     [Erlang=50] Iteration 1/100: BP=0.0234
2024-01-15 10:00:10 INFO     [Erlang=50] Iteration 2/100: BP=0.0215
...
```

**Contract:**
- Created by: Backend (opens file, passes to subprocess)
- Written by: Simulator subprocess (stdout/stderr redirect)
- Read by: SSE log streaming endpoint, UI log viewer

### progress.jsonl

Structured progress events in JSON Lines format. One JSON object per line.

**Event Types:**

```text
{"type":"start","ts":"2024-01-15T10:00:00Z","config":{"total_erlangs":16,"total_iterations":100,"erlang_start":50,"erlang_stop":200,"erlang_step":10}}
{"type":"erlang_start","ts":"2024-01-15T10:00:01Z","erlang":50,"erlang_index":0,"total_erlangs":16}
{"type":"iteration","ts":"2024-01-15T10:00:05Z","erlang":50,"iteration":1,"total_iterations":100,"metrics":{"blocking_prob":0.0234,"mean_hops":2.3}}
{"type":"iteration","ts":"2024-01-15T10:00:10Z","erlang":50,"iteration":2,"total_iterations":100,"metrics":{"blocking_prob":0.0215,"mean_hops":2.4}}
{"type":"erlang_complete","ts":"2024-01-15T10:01:00Z","erlang":50,"metrics":{"mean_blocking":0.022,"ci_lower":0.019,"ci_upper":0.025}}
{"type":"complete","ts":"2024-01-15T10:10:00Z","exit_code":0,"summary":{"total_time_seconds":600}}
{"type":"error","ts":"2024-01-15T10:05:00Z","message":"Out of memory","traceback":"..."}
```

**Schema Definitions:**

```typescript
// Start event (written once at beginning)
interface StartEvent {
  type: "start";
  ts: string;  // ISO 8601 timestamp
  config: {
    total_erlangs: number;
    total_iterations: number;
    erlang_start: number;
    erlang_stop: number;
    erlang_step: number;
  };
}

// Erlang sweep start
interface ErlangStartEvent {
  type: "erlang_start";
  ts: string;
  erlang: number;
  erlang_index: number;
  total_erlangs: number;
}

// Iteration complete
interface IterationEvent {
  type: "iteration";
  ts: string;
  erlang: number;
  iteration: number;
  total_iterations: number;
  metrics: {
    blocking_prob: number;
    mean_hops?: number;
    mean_path_length?: number;
    // ... other optional metrics
  };
}

// Erlang sweep complete
interface ErlangCompleteEvent {
  type: "erlang_complete";
  ts: string;
  erlang: number;
  metrics: {
    mean_blocking: number;
    ci_lower?: number;
    ci_upper?: number;
  };
}

// Simulation complete
interface CompleteEvent {
  type: "complete";
  ts: string;
  exit_code: number;
  summary: {
    total_time_seconds: number;
  };
}

// Error occurred
interface ErrorEvent {
  type: "error";
  ts: string;
  message: string;
  traceback?: string;
}
```

**Contract:**
- Created by: Simulator subprocess (new CLI flag: `--progress-file`)
- Read by: Progress SSE endpoint, UI progress components
- Format: Append-only JSON Lines (one JSON object per line)

### output/

Standard simulation output directory. Structure matches existing FUSION output format.

```
output/
└── {sim_info}/
    └── {thread_num}/
        └── {erlang}_erlang.json
```

Each `{erlang}_erlang.json` contains:
- Blocking probability statistics
- Iteration-level metrics
- Link utilization data
- Path statistics

**Contract:**
- Created by: Simulator (existing output format, unchanged)
- Read by: Artifact browser, plot generation

### plots/ (optional)

Generated visualizations, created on-demand or post-simulation.

```
plots/
├── blocking_vs_erlang.png
├── utilization_heatmap.png
└── ...
```

**Contract:**
- Created by: Backend (on-demand via visualization API) or post-processing
- Read by: UI plot viewer

## Path Security

All artifact access MUST validate that requested paths are within the run directory:

```python
from pathlib import Path

def get_safe_artifact_path(run_id: str, relative_path: str) -> Path:
    """Return absolute path if safe, raise 403 if path traversal attempted."""
    base = Path(f"data/gui_runs/{run_id}").resolve()
    requested = (base / relative_path).resolve()

    # Ensure requested path is under base
    if not requested.is_relative_to(base):
        raise HTTPException(status_code=403, detail="Path traversal not allowed")

    if not requested.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return requested
```

## Directory Cleanup

Runs are NOT automatically deleted. Users can delete via:

1. **UI**: Delete button on run detail page
2. **API**: `DELETE /api/runs/{run_id}`
3. **Manual**: Remove directory from filesystem

When a run is deleted:
1. Kill process if still running
2. Remove database record
3. Remove entire `data/gui_runs/{run_id}/` directory

## Disk Space Considerations

Long simulations can generate significant output:
- `sim.log`: 10MB - 1GB depending on verbosity
- `progress.jsonl`: 1MB - 100MB depending on iterations
- `output/`: 100MB - 10GB depending on configuration

**Recommendations:**
- Implement disk usage display in UI
- Add warning when starting runs if disk is low
- Future: Auto-cleanup of old completed runs (configurable retention)
