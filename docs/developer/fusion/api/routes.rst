.. _api-routes:

======
Routes
======

This page documents the API route modules in ``fusion/api/routes/``.

----

runs.py - Simulation Runs
=========================

:Location: ``fusion/api/routes/runs.py``
:Prefix: ``/api/runs``
:Tag: ``runs``

Provides CRUD operations for simulation runs and real-time streaming.

Endpoints
---------

**POST /api/runs**

Create and start a new simulation run.

.. code-block:: python

   # Request body
   {
       "run_id": "my_simulation",       # Optional custom ID
       "name": "Test Run",              # Optional display name
       "template": "minimal.ini",       # Config template name
       "config_overrides": {            # Optional INI overrides
           "general_settings": {
               "num_requests": "1000"
           }
       }
   }

   # Response (201 Created)
   {
       "id": "abc123def456",
       "name": "Test Run",
       "status": "PENDING",
       "template": "minimal.ini",
       "created_at": "2024-01-15T10:30:00Z",
       ...
   }

**GET /api/runs**

List all runs with optional filtering and pagination.

Query parameters:

- ``status`` - Filter by status (comma-separated: ``RUNNING,COMPLETED``)
- ``limit`` - Maximum results (default: 50, max: 100)
- ``offset`` - Pagination offset (default: 0)

**GET /api/runs/{run_id}**

Get details for a specific run including progress information.

**DELETE /api/runs/{run_id}**

Cancel a running job (sends SIGTERM to process group) or delete a completed run.

**GET /api/runs/{run_id}/logs**

Stream log output via Server-Sent Events.

Query parameters:

- ``from_start`` - Whether to send existing log content (default: true)

Event format:

.. code-block:: text

   event: log
   data: {"line": "INFO: Starting simulation..."}

   event: log
   data: {"line": "INFO: Processing request 1/1000"}

   event: done
   data: {}

**GET /api/runs/{run_id}/progress**

Stream progress updates via Server-Sent Events.

Event format:

.. code-block:: text

   event: progress
   data: {"current_erlang": 300.0, "total_erlangs": 3, "current_iteration": 1, "total_iterations": 2}

   event: done
   data: {}

----

configs.py - Configuration Templates
====================================

:Location: ``fusion/api/routes/configs.py``
:Prefix: ``/api/configs``
:Tag: ``configs``

Provides access to INI configuration templates.

Endpoints
---------

**GET /api/configs/templates**

List available configuration templates from ``fusion/configs/templates/``.

.. code-block:: python

   # Response
   {
       "templates": [
           {
               "name": "minimal.ini",
               "path": "fusion/configs/templates/minimal.ini",
               "description": "Minimal configuration for quick tests"
           },
           ...
       ]
   }

**GET /api/configs/templates/{name}**

Get the content of a specific template.

.. code-block:: python

   # Response
   {
       "name": "minimal.ini",
       "content": "[general_settings]\nnetwork = NSFNet\n..."
   }

----

topology.py - Network Topology
==============================

:Location: ``fusion/api/routes/topology.py``
:Prefix: ``/api/topology``
:Tag: ``topology``

Provides network topology data for visualization.

Endpoints
---------

**GET /api/topology**

List available topologies.

.. code-block:: python

   # Response
   {
       "topologies": ["NSFNet", "USbackbone60", "COST239", ...]
   }

**GET /api/topology/{name}**

Get topology data with computed node positions.

.. code-block:: python

   # Response
   {
       "name": "NSFNet",
       "nodes": [
           {"id": 0, "label": "0", "x": 100.5, "y": 200.3},
           {"id": 1, "label": "1", "x": 150.2, "y": 180.7},
           ...
       ],
       "links": [
           {"source": 0, "target": 1, "weight": 1.0},
           {"source": 0, "target": 2, "weight": 1.0},
           ...
       ],
       "metadata": {
           "num_nodes": 14,
           "num_links": 21
       }
   }

Node positions are computed using a spring layout algorithm (NetworkX) and cached.

----

codebase.py - Codebase Explorer
===============================

:Location: ``fusion/api/routes/codebase.py``
:Prefix: ``/api/codebase``
:Tag: ``codebase``

Provides codebase browsing and search functionality.

Endpoints
---------

**GET /api/codebase/tree**

Get the directory tree structure for the FUSION codebase.

Query parameters:

- ``path`` - Subdirectory to start from (default: project root)
- ``depth`` - Maximum recursion depth (default: 3)

.. code-block:: python

   # Response
   {
       "name": "fusion",
       "type": "directory",
       "children": [
           {
               "name": "core",
               "type": "directory",
               "children": [...]
           },
           {
               "name": "__init__.py",
               "type": "file",
               "size": 1234
           },
           ...
       ]
   }

**GET /api/codebase/file**

Get the content of a specific file.

Query parameters:

- ``path`` - File path relative to project root (required)

.. code-block:: python

   # Response
   {
       "path": "fusion/core/simulation.py",
       "content": "\"\"\"Simulation engine...",
       "language": "python",
       "size": 15000
   }

**GET /api/codebase/search**

Search for files by name pattern.

Query parameters:

- ``query`` - Search pattern (required)
- ``limit`` - Maximum results (default: 20)

.. code-block:: python

   # Response
   {
       "results": [
           {"path": "fusion/core/simulation.py", "type": "file"},
           {"path": "fusion/sim/simulation_engine.py", "type": "file"},
           ...
       ],
       "total": 5
   }

**GET /api/codebase/modules**

Get high-level module descriptions for the architecture view.

.. code-block:: python

   # Response
   {
       "modules": [
           {
               "name": "core",
               "description": "Simulation engine and orchestrator",
               "path": "fusion/core",
               "key_files": ["simulation.py", "orchestrator.py"]
           },
           ...
       ]
   }

----

artifacts.py - Run Artifacts
============================

:Location: ``fusion/api/routes/artifacts.py``
:Prefix: ``/api``
:Tag: ``artifacts``

Provides access to simulation output files.

Endpoints
---------

**GET /api/runs/{run_id}/artifacts**

List output files for a completed run.

.. code-block:: python

   # Response
   {
       "artifacts": [
           {
               "name": "erlang_results.json",
               "path": "data/output/NSFNet/my_run/erlang_results.json",
               "size": 4096,
               "modified": "2024-01-15T10:45:00Z"
           },
           ...
       ]
   }

**GET /api/runs/{run_id}/artifacts/{path:path}**

Download a specific artifact file. Returns the file with appropriate MIME type.

----

system.py - System Endpoints
============================

:Location: ``fusion/api/routes/system.py``
:Prefix: ``/api``
:Tag: ``system``

System-level endpoints for health checks and monitoring.

Endpoints
---------

**GET /api/health**

Health check endpoint for monitoring and load balancers.

.. code-block:: python

   # Response
   {
       "status": "healthy",
       "version": "1.0.0",
       "database": "connected"
   }
