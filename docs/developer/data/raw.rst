.. _data-raw:

===
raw
===

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Stores network topology definition files
   :Location: ``data/raw/``
   :Supported Format: ``.txt`` files only

The ``raw/`` directory contains network topology files that define the physical
structure of optical networks used in simulations. Each file describes the nodes
and links (with distances) that make up a network.

File Format
===========

Topology files use a simple tab-separated format:

.. code-block:: text

   <source_node>	<destination_node>	<length_km>

Where:

- **source_node**: The source node identifier (integer or string)
- **destination_node**: The destination node identifier
- **length_km**: The link length in kilometers

.. important::

   **Bi-directional connections**: If your network requires bi-directional links,
   you must explicitly add both directions. For example, a link between nodes 0
   and 1 requires two entries:

   .. code-block:: text

      0	1	1000
      1	0	1000

Example
-------

Here is an excerpt from ``nsf_network.txt`` (NSFNet topology):

.. code-block:: text

   0	1	1000
   0	2	1500
   0	7	2400
   1	0	1000
   1	3	700
   1	2	600
   2	0	1500
   2	1	600
   2	5	1800

This defines:

- A 1000 km link from node 0 to node 1 (and back)
- A 1500 km link from node 0 to node 2 (and back)
- And so on...

Directory Contents
==================

.. code-block:: text

   raw/
   ├── nsf_network.txt          # NSFNet (14 nodes)
   ├── us_network.txt           # US Network
   ├── europe_network.txt       # Pan-European Network
   ├── USB6014.txt              # US Backbone 60 nodes
   ├── SPNB3014.txt             # Spain Backbone 30 nodes
   ├── geant.txt                # GEANT Network
   ├── toy_network.txt          # Small test network
   ├── metro_net.txt            # Metro network
   ├── dt_network.txt           # Deutsche Telekom network
   ├── USB6014_core_nodes.txt   # Core nodes for USbackbone60
   ├── SPNB3014_core_nodes.txt  # Core nodes for Spainbackbone30
   └── (legacy files)           # .ini and .xlsx files (not used)

Core Node Files
---------------

Some networks have associated ``_core_nodes.txt`` files that identify which nodes
are core (backbone) nodes. These files contain one node identifier per line:

.. code-block:: text

   51
   46
   59
   55
   ...

Legacy Files
------------

The ``.ini`` and ``.xlsx`` files in this directory are from a legacy version of
FUSION and are not used by the current simulation. Only ``.txt`` files are
supported.

Configuration
=============

To select a network topology, set the ``network`` parameter in your configuration
file under ``[topology_settings]``:

.. code-block:: ini

   [topology_settings]
   network = NSFNet

Available Network Names
-----------------------

The following network names are supported and map to their respective files:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Config Name
     - File
     - Description
   * - ``NSFNet``
     - ``nsf_network.txt``
     - NSF Network (14 nodes, 21 links)
   * - ``USNet``
     - ``us_network.txt``
     - US Network
   * - ``Pan-European``
     - ``europe_network.txt``
     - Pan-European Network
   * - ``USbackbone60``
     - ``USB6014.txt``
     - US Backbone (60 nodes)
   * - ``Spainbackbone30``
     - ``SPNB3014.txt``
     - Spain Backbone (30 nodes)
   * - ``geant``
     - ``geant.txt``
     - GEANT Network
   * - ``toy_network``
     - ``toy_network.txt``
     - Small test network
   * - ``metro_net``
     - ``metro_net.txt``
     - Metro network
   * - ``dt_network``
     - ``dt_network.txt``
     - Deutsche Telekom network

Adding a Custom Network
=======================

To add your own network topology:

1. Create a ``.txt`` file in ``data/raw/`` with your topology:

   .. code-block:: text

      0	1	500
      1	0	500
      1	2	750
      2	1	750
      0	2	1000
      2	0	1000

2. Add a mapping in ``fusion/io/structure.py`` in the ``network_files`` dictionary:

   .. code-block:: python

      network_files = {
          # ... existing networks ...
          "MyNetwork": "my_network.txt",
      }

3. Use your network in the configuration:

   .. code-block:: ini

      [topology_settings]
      network = MyNetwork

4. (Optional) If your network has core nodes, create a corresponding
   ``my_network_core_nodes.txt`` file and add the loading logic in
   ``fusion/io/structure.py``.

Connection to Simulation
========================

Network topologies are loaded by ``fusion/io/structure.py`` via the
``create_network()`` function. This function:

1. Maps the config network name to the appropriate file
2. Reads the topology file and creates a network dictionary
3. Optionally loads core nodes if available
4. Returns the network structure for use in the simulation
