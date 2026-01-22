.. _gui-overview:

==============
Web-Based GUI
==============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Web-based interface for managing simulations, visualizing networks, and exploring the codebase
   :Launch Command: ``python -m fusion.cli.run_gui``
   :URL: http://127.0.0.1:8765
   :Tech Stack: FastAPI (backend), React + TypeScript (frontend)

The FUSION GUI provides a modern web interface for interacting with the simulator.
Instead of editing INI files and running commands, you can manage simulations,
view results, and explore network topologies through your browser.

Quick Start
===========

**1. Install GUI dependencies**

.. code-block:: bash

   pip install fusion[gui]

**2. Launch the GUI**

.. code-block:: bash

   python -m fusion.cli.run_gui

**3. Open your browser**

Navigate to http://127.0.0.1:8765

Features at a Glance
====================

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - **Simulation Runs**
     - Create, monitor, and view simulation results with real-time log streaming
   * - **Network Topology**
     - Visualize network graphs with node selection and link utilization
   * - **Configuration Editor**
     - Edit INI configurations with syntax highlighting and validation
   * - **Codebase Explorer**
     - Browse the FUSION architecture with guided tours and code viewing
   * - **Settings**
     - Customize theme and display preferences

----

Contents
========

.. toctree::
   :maxdepth: 1

   installation
   launching
   features
