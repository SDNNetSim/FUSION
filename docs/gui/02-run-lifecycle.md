# Run Lifecycle

This document describes how simulation runs are created, executed, monitored, and terminated.

## State Machine

```
See: diagrams/run-state-machine.txt for visual representation
```

## States

| State | Description | Transitions To |
|-------|-------------|----------------|
| `PENDING` | Run created, not yet started | `RUNNING`, `CANCELLED` |
| `RUNNING` | Simulation subprocess is active | `COMPLETED`, `FAILED`, `CANCELLED` |
| `COMPLETED` | Simulation finished successfully | (terminal) |
| `FAILED` | Simulation crashed or errored | (terminal) |
| `CANCELLED` | User requested cancellation | (terminal) |

## Lifecycle Flow

### 1. Run Creation (`POST /api/runs`)

```python
# Backend receives request
request = RunCreateRequest(
    name="My Simulation",
    config={"general_settings": {...}, ...},
    template="default"  # optional, loads template then applies overrides
)

# Generate run ID (UUID)
run_id = uuid.uuid4().hex[:12]

# Create run directory structure
# See: 03-run-directory-contract.md

# Write frozen config
config_path = f"data/gui_runs/{run_id}/config.ini"
write_config(request.config, config_path)

# Insert database record (PENDING)
db.insert(Run(id=run_id, status="PENDING", ...))

# Return immediately
return {"id": run_id, "status": "PENDING"}
```

### 2. Run Start (automatic or manual)

By default, runs start immediately after creation. Future: add `autostart=False` option.

```python
def start_run(run_id: str) -> None:
    run = db.get(run_id)

    # Create log file
    log_path = f"data/gui_runs/{run_id}/logs/sim.log"
    log_file = open(log_path, "w")

    # Start subprocess in NEW SESSION (critical for clean kill)
    process = subprocess.Popen(
        [
            sys.executable, "-m", "fusion.cli.run_sim",
            "--config", f"data/gui_runs/{run_id}/config.ini",
            "--output-dir", f"data/gui_runs/{run_id}/output",
            "--progress-file", f"data/gui_runs/{run_id}/progress.jsonl",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # CRITICAL: Creates new process group
    )

    # Update database
    run.status = "RUNNING"
    run.pid = process.pid
    run.pgid = os.getpgid(process.pid)  # Store process group ID
    run.started_at = datetime.utcnow()
    db.commit()

    # Start background watcher (checks for completion)
    start_watcher(run_id, process)
```

**Why `start_new_session=True`?**

The simulator uses `multiprocessing.Pool` internally, spawning child processes. If we only kill the parent PID, children become orphans. By creating a new session, all descendants share a process group ID (PGID) that we can kill atomically.

### 3. Progress Reporting

The simulator writes structured progress to `progress.jsonl`:

```jsonl
{"type":"start","timestamp":"2024-01-15T10:00:00Z","total_erlangs":10,"total_iterations":100}
{"type":"erlang_start","timestamp":"2024-01-15T10:00:01Z","erlang":50,"erlang_index":0}
{"type":"iteration","timestamp":"2024-01-15T10:00:05Z","erlang":50,"iteration":1,"blocking_prob":0.023}
{"type":"iteration","timestamp":"2024-01-15T10:00:10Z","erlang":50,"iteration":2,"blocking_prob":0.021}
{"type":"erlang_complete","timestamp":"2024-01-15T10:01:00Z","erlang":50,"mean_blocking":0.022}
{"type":"complete","timestamp":"2024-01-15T10:10:00Z","exit_code":0}
```

**Why not parse logs?**
- Logs are for humans, progress is for machines
- Log format may change, breaking parsers
- Structured JSON is unambiguous
- Enables rich progress UI (progress bars, charts)

### 4. Log Streaming

Logs stream via SSE by tailing `sim.log`:

```python
@router.get("/runs/{run_id}/logs")
async def stream_logs(run_id: str):
    run = get_run_or_404(run_id)
    log_path = f"data/gui_runs/{run_id}/logs/sim.log"

    async def generate():
        async with aiofiles.open(log_path, mode="r") as f:
            # Send existing content first
            content = await f.read()
            if content:
                yield {"event": "log", "data": content}

            # Then tail for new content
            while True:
                line = await f.readline()
                if line:
                    yield {"event": "log", "data": line}
                else:
                    # Check if run is still active
                    run = db.get(run_id)
                    if run.status not in ("PENDING", "RUNNING"):
                        yield {"event": "end", "data": run.status}
                        break
                    await asyncio.sleep(0.3)

    return EventSourceResponse(generate())
```

**Why not subprocess PIPE?**

Using `stdout=PIPE` with long-running processes causes deadlocks when the pipe buffer fills. Redirecting to a file and tailing is safe regardless of output volume.

### 5. Run Completion

The background watcher detects completion:

```python
async def watch_run(run_id: str, process: subprocess.Popen):
    while True:
        return_code = process.poll()
        if return_code is not None:
            run = db.get(run_id)
            run.completed_at = datetime.utcnow()

            if return_code == 0:
                run.status = "COMPLETED"
            else:
                run.status = "FAILED"
                run.error_message = f"Exit code: {return_code}"

            db.commit()
            break

        await asyncio.sleep(1.0)
```

### 6. Cancellation

```python
def cancel_run(run_id: str) -> bool:
    run = db.get(run_id)

    if run.status not in ("PENDING", "RUNNING"):
        return False  # Cannot cancel completed/failed runs

    if run.status == "PENDING":
        run.status = "CANCELLED"
        db.commit()
        return True

    # RUNNING: Kill entire process group
    try:
        os.killpg(run.pgid, signal.SIGTERM)

        # Wait briefly for graceful shutdown
        time.sleep(2.0)

        # Force kill if still alive
        try:
            os.killpg(run.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Already dead

        run.status = "CANCELLED"
        run.completed_at = datetime.utcnow()
        db.commit()
        return True

    except ProcessLookupError:
        # Process already gone
        run.status = "CANCELLED"
        db.commit()
        return True
```

**Why `os.killpg()` instead of `process.terminate()`?**

`process.terminate()` only kills the direct child. The simulator's multiprocessing workers would become orphans, continuing to consume resources. `os.killpg()` kills the entire process tree.

### 7. Server Restart Recovery

On startup, the server reconciles database state with reality:

```python
def recover_orphaned_runs():
    """Mark stale RUNNING jobs as FAILED."""
    running_runs = db.query(Run).filter(Run.status == "RUNNING").all()

    for run in running_runs:
        if not is_process_alive(run.pgid):
            run.status = "FAILED"
            run.error_message = "Server restarted while run was active"
            run.completed_at = datetime.utcnow()

    db.commit()

def is_process_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)  # Signal 0 = check existence
        return True
    except (ProcessLookupError, PermissionError):
        return False
```

## Concurrency Limits

### MVP: Single Active Run

For simplicity, MVP allows only one `RUNNING` job at a time:

```python
def can_start_run() -> bool:
    active = db.query(Run).filter(Run.status == "RUNNING").count()
    return active == 0
```

### Future: Configurable Concurrency

```python
MAX_CONCURRENT_RUNS = int(os.getenv("FUSION_GUI_MAX_RUNS", "1"))

def can_start_run() -> bool:
    active = db.query(Run).filter(Run.status == "RUNNING").count()
    return active < MAX_CONCURRENT_RUNS
```

## Database Schema

```sql
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    name TEXT,
    status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','COMPLETED','FAILED','CANCELLED')),
    config_json TEXT NOT NULL,

    -- Process tracking
    pid INTEGER,
    pgid INTEGER,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Error info
    error_message TEXT,

    -- Progress cache (updated periodically)
    current_erlang REAL,
    total_erlangs INTEGER,
    current_iteration INTEGER,
    total_iterations INTEGER
);

CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created_at ON runs(created_at DESC);
```
