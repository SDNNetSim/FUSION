.. _data-pre-calc:

========
pre_calc
========

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Stores pre-calculated simulation data to avoid redundant computation
   :Location: ``data/pre_calc/``
   :File Format: NumPy ``.npy`` files

The ``pre_calc/`` directory stores pre-computed data that would otherwise be
calculated dynamically during each simulation run. By pre-calculating paths,
SNR values, and modulation selections, simulations can run significantly faster
since they simply look up values instead of computing them repeatedly.

When to Use Pre-calculated Data
-------------------------------

Pre-calculated data is useful when:

- You are running many simulations on the same network topology
- You want to eliminate computation overhead for path finding and SNR calculations
- Your routing configuration is fixed (e.g., always using the top 3 shortest paths)
- You need faster simulation execution for large-scale experiments

Directory Structure
===================

Pre-calculated data is organized by network name:

.. code-block:: text

   pre_calc/
   └── <network_name>/          # e.g., USbackbone60
       ├── paths/               # Pre-calculated k-shortest paths
       │   └── <network>-10SP.npy
       ├── snr/                 # Pre-calculated GSNR values
       │   └── GSNR-<network>-<config>.npy
       ├── modulations/         # Pre-calculated modulation selections
       │   └── MF-<network>-<config>.npy
       └── README.md

Data Types
==========

paths/
------

Contains pre-calculated k-shortest paths for all source-destination pairs in the
network.

For example, if you configure your simulation to always consider the top 3 paths
between any source and destination, you can pre-calculate all possible paths once
and store them here. When the simulation needs path 2 for a given source-destination
pair, it looks up the path directly from the ``.npy`` file instead of running a
shortest path algorithm.

**File naming**: ``<network_id>-<num_paths>SP.npy``

Example: ``USB6014-10SP.npy`` contains 10 shortest paths for the USB6014 network.

**File structure**:

The paths file is a NumPy array where each entry represents a source-destination
pair. Each entry is a nested structure containing:

.. code-block:: text

   loaded_data[i]              # Entry for connection i
       [3][0]                  # Path lengths array
       [5][0]                  # Path data
           [k][0]              # k-th path as list of node IDs

The code accesses paths like this:

.. code-block:: python

   # Get path length for k-th path
   path_length = precalc_matrix[3][0][k]

   # Get node sequence for k-th path
   path_nodes = precalc_matrix[5][0][k][0]  # e.g., [0, 3, 7, 12]

snr/
----

Contains pre-calculated Generalized SNR (GSNR) values for paths in the network.

SNR calculations can be computationally expensive, especially when considering
physical layer impairments. Pre-calculating these values eliminates the need to
run SNR computations during simulation.

**File naming**: ``GSNR-<network_id>-<fiber_config>.npy``

Example: ``GSNR-USB6014-MCF4-C2.npy`` contains GSNR values for the USB6014 network
with a 4-core multi-core fiber configuration.

**File structure**:

The GSNR file is a 3D NumPy array indexed by:

.. code-block:: text

   gsnr_data[connection_index][slot_index][path_index]

Where:

- ``connection_index``: Index of the source-destination pair
- ``slot_index``: Spectrum slot position
- ``path_index``: Which of the k-shortest paths (0, 1, 2, ...)

.. code-block:: python

   # Example: Get SNR value for connection 5, slot 10, path 0
   snr_value = gsnr_data[5][10][0]

modulations/
------------

Contains pre-calculated modulation format selections for each path.

Based on path length and SNR, the appropriate modulation format (QPSK, 16-QAM, etc.)
is pre-determined and stored. The simulation can then look up which modulation
format to use rather than calculating it.

**File naming**: ``MF-<network_id>-<fiber_config>.npy``

Example: ``MF-USB6014-MCF4-C2.npy`` contains modulation format assignments for the
USB6014 network.

**File structure**:

The modulation format file has the same 3D structure as the GSNR file:

.. code-block:: text

   mf_data[connection_index][slot_index][path_index]

Values are integers representing modulation formats (mapped via
``modulation_format_mapping_dict`` in the SNR properties).

.. code-block:: python

   # Example: Get modulation format for connection 5, slot 10, path 0
   mod_format = mf_data[5][10][0]  # Returns integer (e.g., 1, 2, 3...)

Configuration
=============

To enable pre-calculated modulation selection in your simulation, set the following
in your configuration file:

.. code-block:: ini

   [general_settings]
   pre_calc_mod_selection = True

When enabled, the simulation will load modulation and SNR data from the appropriate
pre_calc subdirectory based on your network configuration.

Connection to Simulation
========================

Pre-calculated data is loaded by:

- ``fusion/core/routing.py`` - Loads paths from the ``paths/`` subdirectory when
  ``route_method = external_ksp`` is set in your configuration. This triggers the
  ``_handle_external_ksp()`` method internally.
- ``fusion/modules/snr/utils.py`` - Loads modulation formats and GSNR values via
  the ``get_loaded_files()`` function, called from ``fusion/core/snr_measurements.py``

The code determines which files to load based on your network name and fiber
configuration settings.

Creating Pre-calculated Data
============================

To create your own pre-calculated data:

1. Create a subdirectory under ``data/pre_calc/`` matching your network name

2. Generate the ``.npy`` files for paths, SNR, and/or modulations using NumPy:

   .. code-block:: python

      import numpy as np

      # Example: Save pre-calculated paths
      paths_data = compute_all_ksp(topology, k=10)
      np.save('data/pre_calc/MyNetwork/paths/MyNetwork-10SP.npy', paths_data)

3. Update the file mapping in the relevant code if using a new network or
   configuration (see ``fusion/modules/snr/utils.py`` for the mapping structure)

4. Enable pre-calculated data in your configuration file

Supported Networks
==================

Currently, pre-calculated data is available for:

- **USbackbone60** (USB6014) - US backbone network with 60 nodes
- **Spainbackbone30** (SPNB3014) - Spain backbone network with 30 nodes

To add support for additional networks, you need to:

1. Generate the pre-calculated ``.npy`` files
2. Add the network to the file mapping dictionaries in the codebase
