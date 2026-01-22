.. _gui-installation:

============
Installation
============

The GUI requires additional dependencies beyond the core FUSION installation.

Installing GUI Dependencies
===========================

Install FUSION with the ``gui`` extras:

.. code-block:: bash

   pip install fusion[gui]

This installs the following additional packages:

- **uvicorn** - ASGI server for running the FastAPI application
- **fastapi** - Web framework for the REST API
- **sqlalchemy** - Database ORM for run management
- **aiosqlite** - Async SQLite driver
- **sse-starlette** - Server-Sent Events for real-time log streaming

Verifying Installation
======================

Verify the GUI dependencies are installed:

.. code-block:: bash

   python -c "import uvicorn; import fastapi; print('GUI dependencies installed')"

If you see "GUI dependencies installed", you're ready to launch the GUI.

Pre-Built Frontend Assets
=========================

The frontend React application is pre-built and included in the ``fusion/api/static/``
directory. No additional build steps are required for normal usage.

.. note::

   If you need to modify the frontend, see the
   :doc:`frontend developer documentation </developer/fusion/frontend/index>`
   for build instructions.

Troubleshooting
===============

**ImportError: No module named 'uvicorn'**

The GUI dependencies are not installed. Run:

.. code-block:: bash

   pip install fusion[gui]

**Static files not found**

Ensure the ``fusion/api/static/`` directory exists and contains ``index.html``.
If developing from source, you may need to build the frontend first:

.. code-block:: bash

   cd frontend
   npm install
   npm run build
