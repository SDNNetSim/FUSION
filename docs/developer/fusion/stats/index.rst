.. _stats-module:

============
Stats Module
============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Modern, domain-driven statistics collection for simulations
   :Location: ``fusion/stats/``
   :Key Files: ``collector.py``
   :Depends On: ``fusion.domain`` (Request, AllocationResult, SimulationConfig)
   :Used By: New simulation engine (v6.x+)

The ``stats`` module provides ``StatsCollector``, a modern statistics aggregation
class designed to work with domain objects (Request, AllocationResult) and produce
metrics compatible with analysis tools.

.. important::

   This module is part of the **v6.x refactoring effort** to consolidate
   statistics collection into a clean, domain-driven design. It will eventually
   replace the legacy statistics scattered across the codebase.

Understanding the Statistics Landscape
======================================

.. warning::

   **This is confusing - and we know it!**

   FUSION has **four different statistics-related components** that evolved over
   time. This section explains what each does and the migration plan.

The Statistics Confusion Matrix
-------------------------------

.. list-table:: Statistics Components in FUSION
   :header-rows: 1
   :widths: 18 15 25 22 20

   * - Component
     - Location
     - Purpose
     - Status
     - When to Use
   * - ``StatsCollector``
     - ``fusion/stats/``
     - Modern domain-driven collection
     - **New (v6.x)**
     - New code, orchestrator path
   * - ``SimStats``
     - ``fusion/core/metrics.py``
     - Legacy comprehensive statistics
     - **Legacy (v5.x)**
     - Legacy simulation engine
   * - ``StatsProps``
     - ``fusion/core/properties.py``
     - Legacy statistics container
     - **Legacy (v5.x)**
     - Legacy data structures
   * - ``GroomingStatistics``
     - ``fusion/reporting/statistics.py``
     - Grooming-specific metrics
     - **Active**
     - Traffic grooming features

Visual Comparison
-----------------

.. code-block:: text

   LEGACY ARCHITECTURE (v5.x)
   ==========================

   SimulationEngine
         |
         +---> SimStats (fusion/core/metrics.py)
         |     - 2000+ lines, monolithic
         |     - Tightly coupled to engine
         |     - Computes during simulation
         |
         +---> StatsProps (fusion/core/properties.py)
               - Data container class
               - Stores computed statistics
               - Passed between components


   NEW ARCHITECTURE (v6.x+)
   ========================

   SimulationEngine + SDNOrchestrator
         |
         +---> StatsCollector (fusion/stats/collector.py)
               - ~800 lines, focused
               - Consumes domain objects
               - Clean separation of concerns
               - Exports to multiple formats

Why So Many Statistics Classes?
-------------------------------

**Historical Evolution:**

1. **v4.x**: ``StatsProps`` was a simple data container in ``properties.py``

2. **v5.x**: ``SimStats`` in ``metrics.py`` grew to handle all computation,
   becoming a 2000+ line monolith tightly coupled to the simulation engine

3. **v6.0** (current): ``StatsCollector`` introduced as part of domain-driven
   refactoring, designed to consume domain objects (Request, AllocationResult)
   cleanly. ``GroomingStatistics`` added to ``reporting/`` for specialized
   traffic grooming metrics.

**The Plan:**

.. note::

   **Migration Roadmap (Future Releases)**

   1. ``StatsCollector`` becomes the primary statistics interface
   2. ``SimStats`` deprecated, functionality migrated to StatsCollector
   3. ``StatsProps`` deprecated, replaced by StatsCollector export methods
   4. ``GroomingStatistics`` remains for grooming-specific features

Detailed Comparison
-------------------

.. list-table:: Feature Comparison
   :header-rows: 1
   :widths: 30 20 20 30

   * - Feature
     - StatsCollector (new)
     - SimStats (legacy)
     - Notes
   * - **Lines of code**
     - ~800
     - ~2000+
     - New design is more focused
   * - **Input**
     - Domain objects (Request, Result)
     - Raw dicts, engine internals
     - Domain objects are cleaner
   * - **Output**
     - Multiple export formats
     - Single dict format
     - New supports comparison format
   * - **Coupling**
     - Loosely coupled
     - Tightly coupled to engine
     - New can be used standalone
   * - **Grooming stats**
     - Basic tracking
     - Embedded in class
     - Separate GroomingStatistics class
   * - **SNR tracking**
     - List of values + aggregates
     - Complex per-link tracking
     - Simplified in new design
   * - **Protection stats**
     - Switchover tracking
     - Limited
     - New has better protection support

Which One Should I Use?
-----------------------

**Use StatsCollector when:**

- Writing new code using SDNOrchestrator
- Working with domain objects (Request, AllocationResult)
- Need clean export to comparison format
- Building new analysis pipelines

**Use SimStats when:**

- Maintaining legacy simulation engine code
- Working with code that expects legacy format
- Need detailed per-link SNR tracking (not yet in StatsCollector)

**Use GroomingStatistics when:**

- Working specifically with traffic grooming features
- Need grooming rate, lightpath utilization, bandwidth savings

Module Architecture
===================

StatsCollector Design
---------------------

.. code-block:: text

   +-------------------+
   | SimulationEngine  |
   +-------------------+
            |
            | record_arrival(request, result)
            | record_release(request)
            v
   +-------------------+
   |  StatsCollector   |
   +-------------------+
   | - total_requests  |
   | - blocked_requests|
   | - snr_values[]    |
   | - hop_counts[]    |
   | - modulations{}   |
   +-------------------+
            |
            +---> to_comparison_format()  --> Analysis scripts
            +---> to_legacy_stats_dict()  --> Legacy compatibility
            +---> blocking_probability    --> Real-time access

Data Flow
---------

.. code-block:: text

   Request Processing:
   -------------------

   1. Request arrives
            |
            v
   2. SDNOrchestrator.handle_arrival()
            |
            v
   3. AllocationResult returned (success/blocked)
            |
            v
   4. StatsCollector.record_arrival(request, result)
            |
            +---> Updates counters (total, success, blocked)
            +---> Tracks bandwidth (requested, allocated)
            +---> Records modulation used
            +---> Records SNR values
            +---> Records path metrics (hops, length)
            +---> Updates block reasons (if blocked)

   5. Request departs
            |
            v
   6. StatsCollector.record_release(request)
            |
            +---> (Future: utilization tracking)

Key Data Structures
-------------------

**Request Counters:**

.. code-block:: python

   total_requests: int = 0
   successful_requests: int = 0
   blocked_requests: int = 0

**Tracking Dictionaries:**

.. code-block:: python

   block_reasons: dict[str, int]      # "NO_PATH" -> 5, "CONGESTION" -> 12
   modulations_used: dict[str, int]   # "QPSK" -> 100, "16-QAM" -> 50
   core_usage: dict[int, int]         # 0 -> 45, 1 -> 38, 2 -> 42
   band_usage: dict[str, int]         # "c" -> 80, "l" -> 20

**Measurement Lists:**

.. code-block:: python

   snr_values: list[float]       # [18.5, 19.2, 17.8, ...]
   hop_counts: list[int]         # [3, 4, 2, 5, ...]
   path_lengths_km: list[float]  # [450.0, 320.5, 680.2, ...]
   xt_values: list[float]        # [-30.5, -28.0, ...]

Components
==========

collector.py
------------

:Purpose: Centralized statistics collection for simulation runs
:Key Class: ``StatsCollector``

**Initialization:**

.. code-block:: python

   from fusion.stats import StatsCollector
   from fusion.domain.config import SimulationConfig

   config = SimulationConfig(network_name="NSFNet", erlang=300.0, ...)
   collector = StatsCollector(config)

**Recording Events:**

.. code-block:: python

   # After each request is processed
   collector.record_arrival(request, allocation_result)

   # After request departs
   collector.record_release(request)

   # Manual SNR/XT recording (if needed)
   collector.record_snr(18.5)
   collector.record_xt(-30.0)

**Accessing Metrics:**

.. code-block:: python

   # Computed properties (real-time)
   bp = collector.blocking_probability      # 0.0 to 1.0
   sr = collector.success_rate              # 1.0 - bp
   avg_snr = collector.average_snr          # Mean SNR in dB
   avg_hops = collector.average_hop_count   # Mean path hops

   # Feature ratios
   grooming_ratio = collector.grooming_ratio
   slicing_ratio = collector.slicing_ratio
   protection_ratio = collector.protection_ratio

**Exporting Results:**

.. code-block:: python

   # For analysis scripts (run_comparison.py compatible)
   results = collector.to_comparison_format()

   # For legacy code compatibility
   legacy_dict = collector.to_legacy_stats_dict()

**Merging (for parallel runs):**

.. code-block:: python

   # Combine results from parallel simulation runs
   main_collector.merge(worker_collector_1)
   main_collector.merge(worker_collector_2)

**Reset (between iterations):**

.. code-block:: python

   # Clear all statistics for new iteration
   collector.reset()

Computed Properties
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Property
     - Description
     - Type
   * - ``blocking_probability``
     - blocked / total requests
     - float
   * - ``success_rate``
     - 1.0 - blocking_probability
     - float
   * - ``average_snr``
     - Mean of snr_values
     - float
   * - ``min_snr``
     - Minimum SNR recorded
     - float
   * - ``max_snr``
     - Maximum SNR recorded
     - float
   * - ``grooming_ratio``
     - groomed / successful requests
     - float
   * - ``slicing_ratio``
     - sliced / successful requests
     - float
   * - ``protection_ratio``
     - protected / successful requests
     - float
   * - ``bandwidth_utilization``
     - allocated / requested bandwidth
     - float
   * - ``average_hop_count``
     - Mean of hop_counts
     - float
   * - ``average_path_length_km``
     - Mean of path_lengths_km
     - float

Integration with Other Modules
==============================

Relationship to fusion/core/metrics.py
--------------------------------------

``SimStats`` in ``fusion/core/metrics.py`` is the **legacy** statistics class:

.. code-block:: text

   fusion/core/metrics.py (LEGACY)
   ===============================
   - SimStats class (~2000 lines)
   - Tightly coupled to simulation engine
   - Computes statistics during simulation
   - Uses raw dicts and engine internals

   fusion/stats/collector.py (NEW)
   ===============================
   - StatsCollector class (~800 lines)
   - Loosely coupled, domain-driven
   - Records from domain objects
   - Clean export to multiple formats

**Key Difference:** SimStats *computes* during simulation. StatsCollector *records*
from domain objects after the fact.

Relationship to fusion/reporting/statistics.py
----------------------------------------------

``GroomingStatistics`` in ``fusion/reporting/statistics.py`` handles
**grooming-specific** metrics:

.. code-block:: text

   StatsCollector (fusion/stats/)
   ==============================
   - General simulation statistics
   - Request counts, SNR, paths
   - Used by simulation engine

   GroomingStatistics (fusion/reporting/)
   ======================================
   - Grooming-specific only
   - Lightpath utilization, bandwidth savings
   - Used when grooming is enabled

**Why separate?** Grooming statistics are specialized and only relevant when
``is_grooming_enabled=True``. Keeping them separate avoids bloating StatsCollector.

Relationship to fusion/core/properties.py
-----------------------------------------

``StatsProps`` in ``fusion/core/properties.py`` is a **legacy data container**:

.. code-block:: text

   StatsProps (LEGACY)
   ===================
   - Data container class
   - Stores computed statistics
   - Passed between components
   - No computation logic

   StatsCollector (NEW)
   ====================
   - Active recording and computation
   - to_legacy_stats_dict() for compatibility
   - Will replace StatsProps over time

Development Guide
=================

Adding New Metrics
------------------

1. Add the field to ``StatsCollector`` dataclass:

   .. code-block:: python

      # In collector.py
      new_metric_values: list[float] = field(default_factory=list)

2. Update recording method (``_record_success`` or ``_record_block``):

   .. code-block:: python

      # In _record_success or record_arrival
      if result.new_metric is not None:
          self.new_metric_values.append(result.new_metric)

3. Add computed property if needed:

   .. code-block:: python

      @property
      def average_new_metric(self) -> float:
          """Calculate average of new metric."""
          if not self.new_metric_values:
              return 0.0
          return sum(self.new_metric_values) / len(self.new_metric_values)

4. Add to export methods (``to_comparison_format``, ``to_legacy_stats_dict``)

5. Update ``reset()`` and ``merge()`` methods

6. Add tests in ``fusion/tests/stats/test_collector.py``

Best Practices
--------------

1. **Use domain objects:** Record from Request/AllocationResult, not raw dicts
2. **Keep it focused:** General stats here, specialized stats elsewhere
3. **Immutable inputs:** Don't modify request or result in recording methods
4. **Thread safety:** StatsCollector is NOT thread-safe; use one per process
5. **Reset between iterations:** Call ``reset()`` before each new iteration

Testing
=======

:Test Location: ``fusion/tests/stats/``
:Run Tests: ``pytest fusion/tests/stats/ -v``

.. code-block:: bash

   # Run stats module tests
   pytest fusion/tests/stats/ -v

   # Run with coverage
   pytest --cov=fusion.stats fusion/tests/stats/

Migration Guide
===============

Migrating from SimStats to StatsCollector
-----------------------------------------

**Before (Legacy):**

.. code-block:: python

   from fusion.core.metrics import SimStats

   stats = SimStats(engine_props, sim_info, stats_props)
   # ... simulation runs ...
   stats.iter_update(request_data, sdn_data, network_spectrum)
   blocking = stats.get_blocking_probability()

**After (New):**

.. code-block:: python

   from fusion.stats import StatsCollector
   from fusion.domain.config import SimulationConfig

   config = SimulationConfig.from_engine_props(engine_props)
   collector = StatsCollector(config)

   # After each request
   collector.record_arrival(request, result)

   # Access metrics
   blocking = collector.blocking_probability

   # Export for analysis
   results = collector.to_comparison_format()

Related Documentation
=====================

- :ref:`core-metrics` - Legacy SimStats class
- :ref:`reporting-module` - GroomingStatistics and result formatting
- :ref:`core-module` - Simulation engine integration
- ``fusion/domain/`` - Domain objects (Request, AllocationResult)
