.. _analysis-module:

===============
Analysis Module
===============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Analyze network topology, link utilization, and spectrum usage patterns
   :Location: ``fusion/analysis/``
   :Key Files: ``network_analysis.py``
   :Depends On: ``numpy``, ``fusion.utils.logging_config``
   :Used By: Simulation reporting, performance analysis scripts

The analysis module provides utilities for examining network state during and after
simulations. It answers questions like "How congested is the network?", "Which links
are bottlenecks?", and "What's the overall spectrum utilization?"

Developers work here when they need to add new analysis metrics, modify how
utilization is calculated, or integrate analysis into new reporting features.

Key Concepts
============

Link Utilization
   The ratio of occupied spectrum slots to total available slots on a link.
   A value of 1.0 means fully utilized, 0.0 means empty.

Bottleneck Link
   A link whose utilization exceeds a threshold (default 80%). These links
   limit network capacity and cause blocking.

Bidirectional Links
   Network links exist in both directions (A-B and B-A). Analysis methods
   handle this by either processing each direction separately or normalizing
   to count physical links once.

Cores Matrix
   The spectrum state for a link, stored as a 2D array with shape
   ``(num_cores, num_slots)``. Values indicate slot status:

   - ``0``: Free slot
   - ``positive int``: Occupied by lightpath with that ID
   - ``negative int``: Guard band for lightpath with ``abs(value)`` ID

.. note::

   This module currently uses the legacy dict format for ``network_spectrum``
   (v5.5.0 compatibility). It will be migrated to use ``NetworkState`` and
   ``LinkSpectrum`` objects directly in v6.1.0.

Architecture
============

.. code-block:: text

   fusion/analysis/
   ├── __init__.py              # Exports NetworkAnalyzer
   ├── network_analysis.py      # NetworkAnalyzer class
   ├── README.md                # Module documentation
   └── tests/
       ├── __init__.py
       ├── README.md
       └── test_network_analysis.py

Data Flow
---------

1. **Input**: ``network_spectrum`` dict from simulation engine (or ``NetworkState.network_spectrum_dict``)
2. **Processing**: Iterate over links, analyze cores matrices, aggregate statistics
3. **Output**: Dictionary of metrics (utilization stats, bottleneck lists, congestion data)

Components
==========

network_analysis.py
-------------------

:Purpose: Provides the ``NetworkAnalyzer`` class with static analysis methods
:Key Classes: ``NetworkAnalyzer``

The ``NetworkAnalyzer`` class contains four static methods for analyzing network state:

**get_link_usage_summary()**

Returns per-link usage statistics including usage count, throughput, and link number.
Processes each directional link separately.

.. code-block:: python

   from fusion.analysis import NetworkAnalyzer

   # Get usage summary for all links
   usage = NetworkAnalyzer.get_link_usage_summary(network_spectrum)
   # Returns: {
   #     "A-B": {
   #         "usage_count": 10,    # Number of allocations on this link
   #         "throughput": 100.5,  # Total bandwidth served (Gbps)
   #         "link_num": 1         # Link index in topology
   #     },
   #     ...
   # }

**analyze_network_congestion()**

Calculates network-wide congestion metrics including total occupied slots,
guard slots, and active request counts.

.. code-block:: python

   # Analyze full network
   congestion = NetworkAnalyzer.analyze_network_congestion(network_spectrum)

   # Analyze specific paths only
   congestion = NetworkAnalyzer.analyze_network_congestion(
       network_spectrum,
       specific_paths=[("A", "B"), ("C", "D")]
   )
   # Returns: {
   #     "total_occupied_slots": 1500,   # Slots with data or guard bands
   #     "total_guard_slots": 200,       # Slots used for guard bands only
   #     "active_requests": 45,          # Unique lightpath IDs found
   #     "links_analyzed": 10,           # Number of links processed
   #     "avg_occupied_per_link": 150.0, # Mean occupied slots per link
   #     "avg_guard_per_link": 20.0      # Mean guard slots per link
   # }

**get_network_utilization_stats()**

Calculates aggregate utilization statistics across the network.

.. code-block:: python

   stats = NetworkAnalyzer.get_network_utilization_stats(network_spectrum)
   # Returns: {
   #     "overall_utilization": 0.45,      # Ratio of occupied to total slots
   #     "average_link_utilization": 0.42, # Mean utilization across all links
   #     "max_link_utilization": 0.85,     # Highest link utilization
   #     "min_link_utilization": 0.10,     # Lowest link utilization
   #     "total_slots": 10240,             # Total spectrum slots in network
   #     "occupied_slots": 4608,           # Slots currently in use
   #     "links_processed": 32             # Number of physical links analyzed
   # }

**identify_bottleneck_links()**

Finds links exceeding a utilization threshold, sorted by utilization descending.

.. code-block:: python

   # Find links above 80% utilization (default)
   bottlenecks = NetworkAnalyzer.identify_bottleneck_links(network_spectrum)

   # Custom threshold
   bottlenecks = NetworkAnalyzer.identify_bottleneck_links(
       network_spectrum,
       threshold=0.9
   )
   # Returns: [
   #     {
   #         "link_key": "A-B",      # Normalized link identifier
   #         "utilization": 0.95,    # Max core utilization on this link
   #         "usage_count": 50,      # Number of allocations
   #         "throughput": 500.0     # Total bandwidth served (Gbps)
   #     },
   #     ...  # Sorted by utilization descending
   # ]

Dependencies
============

This Module Depends On
----------------------

- ``fusion.utils.logging_config`` - Logging utilities
- External: ``numpy`` - Array operations for spectrum matrix analysis

Modules That Depend On This
---------------------------

- ``fusion.reporting`` - Uses analysis for simulation reports
- ``fusion.core.metrics`` - May use for real-time monitoring

Development Guide
=================

Getting Started
---------------

1. Read the `Key Concepts`_ section above
2. Examine ``network_analysis.py`` to understand the ``NetworkAnalyzer`` class
3. Run the tests to see example inputs and expected outputs

Common Tasks
------------

**Adding a new analysis metric**

1. Add a new static method to ``NetworkAnalyzer`` in ``network_analysis.py``
2. Follow the existing pattern: accept ``network_spectrum`` dict, return a dict
3. Handle bidirectional links appropriately (count once or separately)
4. Add comprehensive tests in ``tests/test_network_analysis.py``
5. Update the module README

**Modifying utilization calculation**

1. Locate the relevant method (likely ``get_network_utilization_stats``)
2. Update the calculation logic
3. Update tests to reflect new expected values
4. Document any behavioral changes

Code Patterns
-------------

**Static Analysis Method Pattern**

All analysis methods follow this pattern:

.. code-block:: python

   @staticmethod
   def analyze_something(network_spectrum: dict) -> dict[str, Any]:
       """
       Analyze network for something.

       :param network_spectrum: Network spectrum database
       :return: Dictionary of analysis results
       """
       # Track processed links to avoid double-counting bidirectional
       processed_links = set()

       for (src, dst), link_data in network_spectrum.items():
           # Normalize key for bidirectional handling
           link_key = f"{min(src, dst)}-{max(src, dst)}"

           if link_key in processed_links:
               continue
           processed_links.add(link_key)

           # Process link_data["cores_matrix"]
           ...

       return {"metric": value, ...}

Testing
=======

:Test Location: ``fusion/analysis/tests/``
:Run Tests: ``pytest fusion/analysis/tests/ -v``

**Unit Tests**

All ``NetworkAnalyzer`` methods have comprehensive unit tests covering:

- Empty network edge cases
- Single link scenarios
- Multiple links with bidirectional handling
- Missing fields and graceful degradation
- Parametrized tests for link key formatting

**Adding New Tests**

Follow the AAA pattern with test classes organized by method:

.. code-block:: python

   class TestNewAnalysisMethod:
       """Tests for the new_analysis_method method."""

       def test_empty_network_returns_defaults(self) -> None:
           """Test that empty network returns sensible defaults."""
           # Arrange
           network_spectrum: dict = {}

           # Act
           result = NetworkAnalyzer.new_analysis_method(network_spectrum)

           # Assert
           assert result["metric"] == 0
