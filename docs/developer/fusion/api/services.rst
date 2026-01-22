.. _api-services:

========
Services
========

This page documents the service layer in ``fusion/api/services/``. Services contain
the business logic for the API, separating concerns from the route handlers.

----

run_manager.py - Simulation Process Lifecycle
=============================================

:Location: ``fusion/api/services/run_manager.py``
:Class: ``RunManager``

Manages the full lifecycle of simulation runs: creation, process spawning,
monitoring, and cleanup.

Class Overview
--------------

.. code-block:: python

   class RunManager:
       """Manages simulation run lifecycle."""

       def __init__(self, db: Session):
           """Initialize with database session."""

       def create_run(self, data: RunCreate) -> Run:
           """Create a new run and spawn the simulation process."""

       def cancel_or_delete(self, run_id: str) -> Run | None:
           """Cancel running simulation or delete completed run."""

Key Methods
-----------

**create_run(data: RunCreate) -> Run**

Creates a new simulation run:

1. Generates a unique 12-character hex ID
2. Writes the configuration to a temporary INI file
3. Creates the database record with PENDING status
4. Spawns the simulation subprocess with ``fusion-sim``
5. Updates the database with process PID and RUNNING status
6. Starts a background thread to monitor process completion

**cancel_or_delete(run_id: str) -> Run | None**

Handles run cancellation and deletion:

- For RUNNING status: Sends SIGTERM to the process group
- For COMPLETED/FAILED/CANCELLED: Deletes the database record
- Returns None if run not found

**recover_orphaned_runs() -> None**

Module-level function called at startup. Marks any RUNNING/PENDING runs as
FAILED if their process is no longer alive. This handles cases where the API
server crashed or was killed.

Process Management
------------------

Simulations run as subprocesses with these characteristics:

- New process group (``start_new_session=True``) for clean cancellation
- Stdout/stderr redirected to ``logs/{run_id}.log``
- Process tracked by both PID and PGID (process group ID)
- Cancellation sends SIGTERM to entire process group

.. code-block:: python

   # Process spawning
   process = subprocess.Popen(
       ["fusion-sim", "--config_path", config_path, "--run_id", run_id],
       stdout=log_file,
       stderr=subprocess.STDOUT,
       start_new_session=True,  # New process group
   )

   # Cancellation (kills entire process group)
   os.killpg(run.pgid, signal.SIGTERM)

----

progress_watcher.py - Progress Streaming
========================================

:Location: ``fusion/api/services/progress_watcher.py``
:Function: ``stream_progress``

Monitors simulation progress files and streams updates via SSE.

How It Works
------------

The simulation writes progress to a JSONL file (``progress.jsonl``) in the run
output directory. Each line is a JSON object with current progress:

.. code-block:: json

   {"erlang": 300.0, "iteration": 1, "total_iterations": 2, "timestamp": "..."}
   {"erlang": 300.0, "iteration": 2, "total_iterations": 2, "timestamp": "..."}
   {"erlang": 400.0, "iteration": 1, "total_iterations": 2, "timestamp": "..."}

The watcher:

1. Opens the progress file (or waits for it to exist)
2. Reads existing lines and yields them as SSE events
3. Polls for new lines with 1-second intervals
4. Yields a ``done`` event when the run completes

.. code-block:: python

   async def stream_progress(run_id: str) -> AsyncGenerator[dict, None]:
       """Stream progress events for a run."""
       progress_file = get_progress_file_path(run_id)

       # Wait for file to exist
       while not progress_file.exists():
           if is_run_complete(run_id):
               yield {"event": "done", "data": "{}"}
               return
           await asyncio.sleep(1)

       # Stream progress updates
       with open(progress_file) as f:
           while True:
               line = f.readline()
               if line:
                   yield {"event": "progress", "data": line.strip()}
               elif is_run_complete(run_id):
                   yield {"event": "done", "data": "{}"}
                   return
               else:
                   await asyncio.sleep(1)

Usage in Route Handler
----------------------

.. code-block:: python

   from sse_starlette.sse import EventSourceResponse
   from .services.progress_watcher import stream_progress

   @router.get("/{run_id}/progress")
   async def stream_run_progress(run_id: str) -> EventSourceResponse:
       return EventSourceResponse(stream_progress(run_id))

----

artifact_service.py - Output File Handling
==========================================

:Location: ``fusion/api/services/artifact_service.py``
:Functions: ``list_artifacts``, ``get_artifact_path``

Manages access to simulation output files.

Functions
---------

**list_artifacts(run_id: str) -> list[ArtifactInfo]**

Lists all output files for a run:

1. Determines the output directory from run configuration
2. Recursively scans for files (excluding hidden files/directories)
3. Returns file metadata (name, path, size, modified time)

.. code-block:: python

   artifacts = list_artifacts("my_run_id")
   # Returns:
   # [
   #     ArtifactInfo(name="erlang_results.json", path="...", size=4096, ...),
   #     ArtifactInfo(name="blocking_probability.csv", path="...", size=1024, ...),
   # ]

**get_artifact_path(run_id: str, relative_path: str) -> Path | None**

Resolves and validates an artifact path:

1. Joins the output directory with the relative path
2. Validates the path is within the output directory (prevents traversal)
3. Checks the file exists
4. Returns the absolute path or None

.. code-block:: python

   path = get_artifact_path("my_run", "results/erlang.json")
   if path:
       return FileResponse(path)

Security
--------

The ``get_artifact_path`` function includes path traversal protection:

.. code-block:: python

   def get_artifact_path(run_id: str, relative_path: str) -> Path | None:
       output_dir = get_output_directory(run_id)
       full_path = (output_dir / relative_path).resolve()

       # Prevent path traversal attacks
       if not str(full_path).startswith(str(output_dir.resolve())):
           return None

       if not full_path.exists() or not full_path.is_file():
           return None

       return full_path

----

Log Streaming
=============

:Location: ``fusion/api/services/run_manager.py``
:Function: ``stream_run_logs``

Streams simulation log output via SSE. Similar to progress streaming but reads
from the log file (``logs/{run_id}.log``).

.. code-block:: python

   async def stream_run_logs(
       run_id: str,
       from_start: bool = True
   ) -> AsyncGenerator[dict, None]:
       """Stream log lines for a run."""
       log_file = get_log_file_path(run_id)

       with open(log_file) as f:
           if not from_start:
               f.seek(0, 2)  # Seek to end

           while True:
               line = f.readline()
               if line:
                   yield {"event": "log", "data": json.dumps({"line": line.rstrip()})}
               elif is_run_complete(run_id):
                   yield {"event": "done", "data": "{}"}
                   return
               else:
                   await asyncio.sleep(0.1)  # 100ms poll interval
