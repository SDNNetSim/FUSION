.. _data-directory:

==============
Data Directory
==============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Stores network topologies, pre-calculated data, configuration input, and result output
   :Location: ``data/``

The ``data/`` directory contains all static data files used by FUSION simulations,
including network topology definitions, pre-calculated routing tables,
configuration inputs, and simulation result outputs.

Architecture
============

.. code-block:: text

   data/
   ├── input/            # Configuration input files
   ├── json_input/       # JSON format input files
   ├── output/           # Simulation result output
   ├── pre_calc/         # Pre-calculated data (cached computations)
   └── raw/              # Network topology files

Components
==========

input/
------

Simulation configuration input files organized by network/date/time.
See :doc:`input_output` for detailed documentation.

json_input/
-----------

JSON format input files for simulations, primarily modulation format definitions.
See :doc:`json_input` for detailed documentation.

output/
-------

Simulation result output files organized by network/date/time/process.
See :doc:`input_output` for detailed documentation.

pre_calc/
---------

Stores pre-calculated paths, SNR values, and modulation selections to speed up
simulations. See :doc:`pre_calc` for detailed documentation.

raw/
----

Network topology files defining nodes and links with distances.
See :doc:`raw` for detailed documentation.

.. toctree::
   :maxdepth: 1
   :caption: Subdirectories

   input_output
   json_input
   pre_calc
   raw
