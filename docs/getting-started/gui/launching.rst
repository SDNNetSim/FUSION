.. _gui-launching:

====================
Launching the GUI
====================

Basic Usage
===========

Start the GUI with the default settings:

.. code-block:: bash

   python -m fusion.cli.run_gui

The server starts at http://127.0.0.1:8765 by default.

CLI Options
===========

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--host``
     - 127.0.0.1
     - Host address to bind the server to
   * - ``--port``
     - 8765
     - Port number to bind the server to
   * - ``--reload``
     - False
     - Enable auto-reload for development (restarts on code changes)
   * - ``--log-level``
     - info
     - Logging level: debug, info, warning, error

Examples
========

**Run on a different port**

.. code-block:: bash

   python -m fusion.cli.run_gui --port 9000

**Allow external access**

.. code-block:: bash

   python -m fusion.cli.run_gui --host 0.0.0.0

.. warning::

   Binding to ``0.0.0.0`` makes the GUI accessible from other machines on your
   network. Only do this in trusted environments.

**Development mode with auto-reload**

.. code-block:: bash

   python -m fusion.cli.run_gui --reload --log-level debug

**Verbose logging**

.. code-block:: bash

   python -m fusion.cli.run_gui --log-level debug

Stopping the Server
===================

Press ``Ctrl+C`` in the terminal to stop the server gracefully.

What Happens at Startup
=======================

When the GUI starts, it:

1. Initializes the SQLite database for storing run metadata
2. Recovers any orphaned runs from previous sessions
3. Mounts the static frontend files
4. Starts the uvicorn ASGI server

You'll see log output similar to:

.. code-block:: text

   INFO:     Initializing FUSION GUI API...
   INFO:     FUSION GUI API ready
   INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
