.. _api-module:

==========
API Module
==========

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: REST API backend for the FUSION web GUI
   :Location: ``fusion/api/``
   :Key Files: ``main.py``, ``routes/*.py``, ``services/*.py``, ``db/*.py``
   :Framework: FastAPI with SQLAlchemy ORM
   :Depends On: ``fusion.cli``, ``fusion.configs``
   :Used By: Frontend React application

The API module provides the REST backend for the FUSION web-based GUI. It handles
simulation run management, configuration loading, log streaming, and serves the
built frontend assets.

Developers work here when adding new API endpoints, modifying the database schema,
or extending the GUI's backend capabilities.

Key Concepts
============

FastAPI Application
-------------------

The API is built with FastAPI, a modern async Python web framework. It provides:

- Automatic OpenAPI documentation at ``/docs``
- Request validation via Pydantic schemas
- Async support for streaming endpoints
- Dependency injection for database sessions

SQLite Database
---------------

Run metadata is stored in a local SQLite database (``fusion.db``). SQLAlchemy ORM
provides the data layer with:

- Automatic schema creation on startup
- Session management via FastAPI dependencies
- Type-safe queries with mapped columns

Server-Sent Events (SSE)
------------------------

Log and progress streaming use SSE for real-time updates:

- ``/api/runs/{id}/logs`` - Stream simulation log output
- ``/api/runs/{id}/progress`` - Stream progress updates

SSE provides efficient one-way server-to-client communication without WebSocket
complexity.

Static File Serving
-------------------

The API serves the pre-built React frontend from ``fusion/api/static/``:

- ``/assets/*`` - JavaScript, CSS, and other assets
- ``/*`` - SPA fallback to ``index.html`` for client-side routing

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/api/
   ├── __init__.py              # Module exports
   ├── main.py                  # FastAPI application entry point
   ├── config.py                # Configuration settings
   ├── dependencies.py          # Shared dependencies
   ├── db/                      # Database layer
   │   ├── database.py          # SQLAlchemy engine and session
   │   └── models.py            # ORM models (Run)
   ├── routes/                  # API route handlers
   │   ├── runs.py              # /api/runs/* endpoints
   │   ├── configs.py           # /api/configs/* endpoints
   │   ├── topology.py          # /api/topology/* endpoints
   │   ├── codebase.py          # /api/codebase/* endpoints
   │   ├── artifacts.py         # /api/runs/{id}/artifacts endpoints
   │   └── system.py            # /api/health endpoint
   ├── schemas/                 # Pydantic request/response schemas
   │   ├── run.py               # Run-related schemas
   │   └── ...
   ├── services/                # Business logic
   │   ├── run_manager.py       # Simulation process lifecycle
   │   ├── progress_watcher.py  # Progress file monitoring
   │   └── artifact_service.py  # Output file handling
   ├── static/                  # Built frontend assets
   │   ├── index.html           # SPA entry point
   │   └── assets/              # JS, CSS bundles
   └── tests/                   # Unit tests

Request Flow
------------

.. code-block:: text

   Browser Request
         │
         ▼
   ┌─────────────────┐
   │  FastAPI App    │  main.py - routing, middleware, exception handling
   │  (main.py)      │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Route Handler  │  routes/*.py - request validation, response formatting
   │  (routes/*.py)  │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Service Layer  │  services/*.py - business logic, process management
   │  (services/*.py)│
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Database       │  db/*.py - SQLAlchemy ORM, session management
   │  (db/*.py)      │
   └─────────────────┘

API Routes Summary
==================

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Endpoint
     - Method
     - Description
   * - ``/api/runs``
     - POST
     - Create and start a new simulation run
   * - ``/api/runs``
     - GET
     - List all runs with optional filtering
   * - ``/api/runs/{id}``
     - GET
     - Get details for a specific run
   * - ``/api/runs/{id}``
     - DELETE
     - Cancel a running job or delete a completed one
   * - ``/api/runs/{id}/logs``
     - GET
     - Stream logs via Server-Sent Events
   * - ``/api/runs/{id}/progress``
     - GET
     - Stream progress via Server-Sent Events
   * - ``/api/runs/{id}/artifacts``
     - GET
     - List output files for a run
   * - ``/api/runs/{id}/artifacts/{path}``
     - GET
     - Download a specific artifact file
   * - ``/api/configs/templates``
     - GET
     - List available configuration templates
   * - ``/api/configs/templates/{name}``
     - GET
     - Get content of a specific template
   * - ``/api/topology/{name}``
     - GET
     - Get topology data with node positions
   * - ``/api/codebase/tree``
     - GET
     - Get directory tree structure
   * - ``/api/codebase/file``
     - GET
     - Get file content
   * - ``/api/codebase/search``
     - GET
     - Search for files by name
   * - ``/api/health``
     - GET
     - Health check endpoint

Development Guide
=================

Running the API in Development
------------------------------

.. code-block:: bash

   # Start with auto-reload
   python -m fusion.cli.run_gui --reload --log-level debug

   # Access API documentation
   open http://127.0.0.1:8765/docs

Adding a New Endpoint
---------------------

**1. Create or update a route module** in ``routes/``:

.. code-block:: python

   # routes/my_feature.py
   from fastapi import APIRouter, Depends
   from sqlalchemy.orm import Session
   from ..db.database import get_db

   router = APIRouter()

   @router.get("/my-endpoint")
   def my_endpoint(db: Session = Depends(get_db)) -> dict:
       """Endpoint description."""
       return {"message": "Hello"}

**2. Register the router** in ``main.py``:

.. code-block:: python

   from .routes import my_feature

   app.include_router(
       my_feature.router,
       prefix="/api",
       tags=["my_feature"]
   )

**3. Add Pydantic schemas** for request/response validation in ``schemas/``.

**4. Add tests** in ``tests/``.

Testing
=======

:Test Location: ``fusion/api/tests/``
:Run Tests: ``pytest fusion/api/tests/ -v``

Tests use FastAPI's TestClient for endpoint testing:

.. code-block:: python

   from fastapi.testclient import TestClient
   from fusion.api.main import app

   client = TestClient(app)

   def test_health_check():
       response = client.get("/api/health")
       assert response.status_code == 200

----

Contents
========

.. toctree::
   :maxdepth: 1

   routes
   services
   database
