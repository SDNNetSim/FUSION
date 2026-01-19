.. _data-json-input:

==========
json_input
==========

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Stores JSON input files for FUSION simulations
   :Location: ``data/json_input/``
   :Primary Use: Modulation format definitions

The ``json_input/`` directory contains JSON-formatted input files used by FUSION
simulations. Currently, its primary purpose is storing modulation format
definitions that specify the relationship between bandwidth, modulation schemes,
transmission reach, and spectrum slot requirements.

Directory Structure
===================

.. code-block:: text

   json_input/
   └── example_mods/
       ├── default_mod_formats.json    # Default modulation assumptions
       └── example_mod_formats.json    # Example custom modulation formats

Modulation Formats
==================

Modulation format files define the physical layer assumptions for your simulation.
Each file can contain multiple named modulation format sets, allowing you to
store different assumptions (from different papers, vendors, or experiments) in
one place.

File Structure
--------------

A modulation format file follows this structure:

.. code-block:: text

   {
     "FORMAT_NAME": {
       "<bandwidth_gbps>": {
         "<modulation_scheme>": {
           "max_length": <reach_in_km>,
           "slots_needed": <slots_without_guardband>
         }
       }
     }
   }

Where:

- **FORMAT_NAME**: A unique identifier you choose for this set of assumptions. Name it whatever makes sense for your use case: your name (``JOHN_ASSUMPTIONS``), a paper reference (``ICC2020_ASSUMPTIONS``), a vendor (``VENDOR_X``), or a descriptive name (``HIGH_REACH_CONFIG``)
- **bandwidth_gbps**: The data rate in Gbps (e.g., ``"25"``, ``"100"``, ``"400"``)
- **modulation_scheme**: The modulation format (e.g., ``"QPSK"``, ``"16-QAM"``, ``"64-QAM"``)
- **max_length**: Maximum transmission distance in kilometers this configuration supports
- **slots_needed**: Number of spectrum slots required (without guardband)

Example
-------

Here is an example from ``default_mod_formats.json``:

.. code-block:: text

   {
     "DEFAULT": {                      // Format name (user-defined)
       "100": {                        // Bandwidth in Gbps
         "QPSK": {                     // Modulation scheme
           "max_length": 5540,         // Max reach: 5,540 km
           "slots_needed": 4           // Requires 4 slots (no guardband)
         },
         "16-QAM": {
           "max_length": 2375,
           "slots_needed": 2
         },
         "64-QAM": {
           "max_length": 916,
           "slots_needed": 2
         }
       },
       "400": {                        // Another bandwidth
         "QPSK": {
           "max_length": 1385,
           "slots_needed": 16
         },
         ...
       }
     }
   }

Multiple Format Sets
--------------------

A single file can contain multiple modulation format sets. This is useful for
comparing different assumptions or sourcing values from different references:

.. code-block:: text

   {
     "DEFAULT": {
       "100": { ... }
     },
     "SNR_ASSUMPTIONS": {
       "100": { ... }
     },
     "VENDOR_X_ASSUMPTIONS": {
       "100": { ... }
     }
   }

You can also store references alongside your format definitions:

.. code-block:: text

   {
     "my_format_reference": "10.1109/ICC40277.2020.9149215",
     "my_format": {
       "100": { ... }
     }
   }

Configuration
=============

To use modulation formats in your simulation, set these two parameters in your
configuration file:

.. code-block:: ini

   [general_settings]
   mod_assumption = DEFAULT
   mod_assumption_path = data/json_input/example_mods/default_mod_formats.json

- **mod_assumption**: The name of the format set to use (must match a key in the JSON file)
- **mod_assumption_path**: Path to the JSON file containing your modulation formats

Connection to Simulation
========================

The modulation format data flows into the simulation through ``fusion/io/generate.py``.
The ``create_bw_info()`` function loads the JSON file and extracts the specified
format set, which is then used throughout the simulation to determine:

- Which modulation format to use for a given path length
- How many spectrum slots to allocate for a request
- Whether a lightpath is feasible given distance constraints

Creating Custom Modulation Formats
==================================

To create your own modulation format file:

1. Create a new JSON file in ``data/json_input/`` (or a subdirectory)

2. Define your format set with proper JSON structure:

   .. code-block:: json

      {
        "MY_CUSTOM_FORMAT": {
          "100": {
            "QPSK": {"max_length": 6000, "slots_needed": 4},
            "16-QAM": {"max_length": 2500, "slots_needed": 2}
          },
          "200": {
            "QPSK": {"max_length": 3000, "slots_needed": 8},
            "16-QAM": {"max_length": 1250, "slots_needed": 4}
          }
        }
      }

3. Update your configuration file to point to the new file:

   .. code-block:: ini

      [general_settings]
      mod_assumption = MY_CUSTOM_FORMAT
      mod_assumption_path = data/json_input/my_custom_mods.json

4. Ensure your bandwidths match those generated by your traffic configuration
