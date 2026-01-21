.. _spectrum-module:

===============
Spectrum Module
===============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Spectrum slot assignment algorithms for optical network resource allocation
   :Location: ``fusion/modules/spectrum/``
   :Key Files: ``registry.py``, ``first_fit.py``, ``best_fit.py``, ``last_fit.py``
   :Depends On: ``fusion.interfaces.spectrum``, ``fusion.core.properties``
   :Used By: ``fusion.core`` (SDNController), ``fusion.core.adapters`` (SpectrumAdapter), ``fusion.pipelines``

The spectrum module provides **algorithm implementations** for assigning spectrum slots
to lightpaths in elastic optical networks. These algorithms are the second half of the
RSA (Routing and Spectrum Assignment) problem.

**What this module does:**

- Assigns contiguous spectrum slots to lightpath requests
- Implements multiple assignment strategies (First-Fit, Best-Fit, Last-Fit)
- Supports multi-band operation (C-band, L-band, etc.)
- Supports multi-core fiber (Space Division Multiplexing)
- Manages spectrum fragmentation
- Provides light path slicing for large bandwidth requests

**When you would work here:**

- Adding a new spectrum assignment algorithm
- Modifying how spectrum slots are selected
- Implementing custom fragmentation metrics
- Optimizing spectrum utilization

Understanding Legacy vs. Orchestrator
=====================================

.. important::

   FUSION supports **two simulation architectures** that coexist. Understanding
   which path your code uses is critical for making modifications.

The spectrum module algorithms are used by **BOTH** architecture paths:

.. code-block:: text

   +===========================================================================+
   |                     SPECTRUM MODULE USAGE                                  |
   +===========================================================================+
   |                                                                            |
   |   use_orchestrator = False              use_orchestrator = True            |
   |   (Legacy Path)                         (Orchestrator Path)                |
   |                                                                            |
   |   +------------------+                  +------------------+               |
   |   | SDNController    |                  | SDNOrchestrator  |               |
   |   | (fusion/core/)   |                  | (fusion/core/)   |               |
   |   +--------+---------+                  +--------+---------+               |
   |            |                                     |                         |
   |            | Direct instantiation                | Via SpectrumAdapter     |
   |            v                                     v                         |
   |   +------------------+                  +------------------+               |
   |   | fusion/core/     |                  | fusion/core/     |               |
   |   | spectrum.py      |                  | adapters/        |               |
   |   | (Spectrum class) |                  | spectrum_adapter |               |
   |   +--------+---------+                  +--------+---------+               |
   |            |                                     |                         |
   |            | Uses                                | Wraps                   |
   |            v                                     v                         |
   |   +-------------------------------------------------------+                |
   |   |              fusion/modules/spectrum/                  |               |
   |   |                                                        |               |
   |   |   FirstFitSpectrum, BestFitSpectrum,                  |                |
   |   |   LastFitSpectrum, LightPathSlicingManager            |                |
   |   +-------------------------------------------------------+                |
   |                                                                            |
   +===========================================================================+

**Key insight:** The algorithms in ``fusion/modules/spectrum/`` are the **same code**
used by both paths. The difference is only in how they are invoked.

How Adapters Work with Spectrum
-------------------------------

.. list-table:: Integration Points
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - Location
     - Role
   * - **SpectrumAdapter**
     - ``fusion/core/adapters/spectrum_adapter.py``
     - Wraps legacy spectrum for orchestrator. Converts ``NetworkState`` to legacy
       ``SpectrumProps``, calls legacy assignment, converts results to ``SpectrumResult``.
   * - **SpectrumPipeline**
     - ``fusion/pipelines/`` (if exists)
     - Fresh orchestrator implementation (if applicable).

**When to modify which:**

- **Adding a new algorithm** -> ``fusion/modules/spectrum/`` + register
- **Changing adapter behavior** -> ``fusion/core/adapters/spectrum_adapter.py``

Key Concepts
============

Spectrum Slots
   The optical spectrum is divided into discrete frequency slots. Each slot has a
   fixed bandwidth (typically 12.5 GHz in flexible grid systems). A lightpath requires
   a contiguous block of slots.

Contiguity Constraint
   All slots assigned to a single lightpath must be adjacent (contiguous) in the
   frequency domain. This is a fundamental constraint in elastic optical networks.

Continuity Constraint
   The same slots must be available on ALL links along the lightpath. A slot that's
   free on one link but occupied on another cannot be used.

Guard Bands
   Empty slots between adjacent lightpaths to prevent inter-channel interference.
   Typically 1-2 slots depending on modulation format.

Fragmentation
   As lightpaths are allocated and released, the spectrum becomes fragmented with
   small unusable gaps. High fragmentation increases blocking probability.

Multi-Band Operation
   Modern systems support multiple bands (C-band ~1530-1565nm, L-band ~1565-1625nm).
   Each band has independent spectrum that can be allocated.

Multi-Core Fiber (MCF)
   Space Division Multiplexing uses fibers with multiple cores. Each core has its
   own spectrum, but cross-talk between cores must be managed.

.. tip::

   The key insight is that spectrum assignment is a **bin packing problem** with
   additional physical constraints. The algorithms differ in their packing strategy
   and how they balance utilization vs. fragmentation.

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/modules/spectrum/
   |-- __init__.py              # Public API exports
   |-- README.md                # Module overview
   |-- registry.py              # Algorithm discovery and creation
   |-- first_fit.py             # First-Fit algorithm (fastest)
   |-- best_fit.py              # Best-Fit algorithm (minimizes fragmentation)
   |-- last_fit.py              # Last-Fit algorithm (distributes allocation)
   |-- light_path_slicing.py    # Manager for segmented allocation
   |-- utils.py                 # Helper functions (SpectrumHelpers)
   |
   |-- visualization/           # Visualization plugin (BETA)
   |   |-- __init__.py
   |   `-- spectrum_plugin.py   # Plugin for spectrum plots
   |
   `-- tests/                   # Unit tests (in fusion/modules/tests/spectrum/)

Data Flow
---------

.. code-block:: text

   1. ROUTING COMPLETES (path selected)
          |
          v
   2. ALGORITHM SELECTION (via registry)
          |
          | create_spectrum_algorithm("first_fit", engine_props, sdn_props, route_props)
          v
   3. SPECTRUM ASSIGNMENT
          |
          | algorithm.assign(path=[node_list], request=request_obj)
          v
   4. RESULTS STORED IN spectrum_props
          |
          | - start_slot: First assigned slot index
          | - end_slot: Last assigned slot index
          | - core_number: Assigned core (for MCF)
          | - current_band: Assigned frequency band
          | - is_free: Whether assignment succeeded
          v
   5. CONSUMER READS spectrum_props
          |
          | (SDNController or SpectrumAdapter)
          v
   6. LIGHTPATH PROVISIONED

Components
==========

registry.py
-----------

:Purpose: Centralized registry for spectrum algorithm discovery and instantiation
:Key Classes: ``SpectrumRegistry``, ``SPECTRUM_ALGORITHMS``
:Key Functions: ``create_spectrum_algorithm()``, ``list_spectrum_algorithms()``

The registry pattern enables dynamic algorithm selection based on configuration:

.. code-block:: python

   from fusion.modules.spectrum import (
       SpectrumRegistry,
       create_spectrum_algorithm,
       list_spectrum_algorithms,
   )

   # List available algorithms
   algorithms = list_spectrum_algorithms()
   # ['first_fit', 'best_fit', 'last_fit']

   # Create an algorithm instance
   assigner = create_spectrum_algorithm(
       "first_fit", engine_props, sdn_props, route_props
   )

   # Assign spectrum to a path
   result = assigner.assign(path=[0, 1, 3, 5], request=request_obj)

   # Check results
   if result and result.get("is_free"):
       print(f"Assigned slots {result['start_slot']}-{result['end_slot']}")

**Registered Algorithms:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Description
   * - ``first_fit``
     - Assigns the first available contiguous block. Fast but may fragment.
   * - ``best_fit``
     - Finds the smallest sufficient block. Minimizes fragmentation.
   * - ``last_fit``
     - Assigns from the end of spectrum. Distributes allocation.

first_fit.py
------------

:Purpose: First-Fit spectrum assignment - fastest algorithm
:Key Class: ``FirstFitSpectrum``

First-Fit scans the spectrum from the lowest index and assigns the **first**
contiguous block that satisfies the request. This is the simplest and fastest
algorithm but can lead to spectrum fragmentation.

.. code-block:: python

   from fusion.modules.spectrum import FirstFitSpectrum

   assigner = FirstFitSpectrum(engine_props, sdn_props, route_props)
   result = assigner.assign(path=[0, 1, 2, 3], request=request_obj)

   # Result structure
   # {
   #     "start_slot": 5,
   #     "end_slot": 8,
   #     "core_num": 0,
   #     "current_band": "c",
   #     "is_free": True
   # }

**Characteristics:**

- **Time Complexity:** O(n) where n = number of slots
- **Fragmentation:** High - tends to create gaps in lower spectrum
- **Use Case:** High-throughput scenarios where speed matters more than efficiency

best_fit.py
-----------

:Purpose: Best-Fit spectrum assignment - minimizes fragmentation
:Key Class: ``BestFitSpectrum``

Best-Fit finds the **smallest** contiguous block that can satisfy the request.
This minimizes wasted spectrum and reduces fragmentation, at the cost of
additional computation.

.. code-block:: python

   from fusion.modules.spectrum import BestFitSpectrum

   assigner = BestFitSpectrum(engine_props, sdn_props, route_props)
   result = assigner.assign(path=[0, 1, 2, 3], request=request_obj)

**Characteristics:**

- **Time Complexity:** O(n) with extra bookkeeping
- **Fragmentation:** Low - fills gaps efficiently
- **Use Case:** When spectrum efficiency is critical

last_fit.py
-----------

:Purpose: Last-Fit spectrum assignment - distributes allocation
:Key Class: ``LastFitSpectrum``

Last-Fit scans from the highest index and assigns the **last** available block.
This distributes allocations across the spectrum and can reduce contention
in certain traffic patterns.

.. code-block:: python

   from fusion.modules.spectrum import LastFitSpectrum

   assigner = LastFitSpectrum(engine_props, sdn_props, route_props)
   result = assigner.assign(path=[0, 1, 2, 3], request=request_obj)

**Characteristics:**

- **Time Complexity:** O(n)
- **Fragmentation:** Medium - spreads allocation
- **Use Case:** Load balancing across spectrum

light_path_slicing.py
---------------------

:Purpose: Manages segmented allocation for large bandwidth requests
:Key Class: ``LightPathSlicingManager``

When a single contiguous block cannot satisfy a large request, light path slicing
splits it into multiple smaller allocations across different spectrum segments
or even different paths.

.. code-block:: python

   from fusion.modules.spectrum import LightPathSlicingManager

   manager = LightPathSlicingManager(engine_props, sdn_props)

   # Try sliced allocation
   result = manager.allocate_slicing(
       path=[0, 1, 2, 3],
       slots_needed=50,  # Large request
       max_segments=5
   )

**Slicing Modes:**

- **Static Slicing:** Pre-defined segment sizes
- **Dynamic Slicing:** Adaptive segmentation based on available spectrum

utils.py
--------

:Purpose: Helper utilities for spectrum operations
:Key Class: ``SpectrumHelpers``

Provides common operations used across spectrum algorithms:

.. code-block:: python

   from fusion.modules.spectrum.utils import SpectrumHelpers

   helpers = SpectrumHelpers(engine_props, sdn_props)

   # Check if slots are available on other links
   available = helpers.check_other_links(path, start_slot, end_slot, core, band)

   # Find the best core for multi-core fiber
   best_core = helpers.find_best_core(path, slots_needed, band)

   # Get link intersections for cross-talk analysis
   intersections = helpers.find_link_inters(path)

visualization/ (BETA)
---------------------

:Purpose: Visualization plugin for spectrum analysis
:Status: **BETA** - API may change in future releases

The visualization submodule provides a plugin that extends FUSION's core
visualization system with spectrum-specific plots:

- **spectrum_heatmap**: Utilization heatmap across links and slots
- **fragmentation_plot**: Fragmentation analysis vs traffic load

See :ref:`spectrum-visualization` for details.

.. warning::

   The visualization plugin is in BETA. It requires the core visualization
   system at ``fusion/visualization/`` to be properly configured.

Development Guide
=================

Getting Started
---------------

1. Read the ``AbstractSpectrumAssigner`` interface in ``fusion/interfaces/spectrum.py``
2. Examine ``first_fit.py`` as the reference implementation
3. Understand how results are stored in ``spectrum_props``
4. Look at existing algorithms for patterns

Adding a New Spectrum Algorithm
-------------------------------

**Step 1: Create the algorithm file**

.. code-block:: python

   # fusion/modules/spectrum/my_spectrum.py
   """My custom spectrum assignment algorithm."""

   from typing import Any
   from fusion.interfaces.spectrum import AbstractSpectrumAssigner


   class MySpectrum(AbstractSpectrumAssigner):
       """
       My custom spectrum assignment algorithm.

       Implements [describe what makes it special].
       """

       def __init__(
           self,
           engine_props: dict[str, Any],
           sdn_props: Any,
           route_props: Any,
       ) -> None:
           super().__init__(engine_props, sdn_props, route_props)
           # Initialize algorithm-specific state

       @property
       def algorithm_name(self) -> str:
           return "my_spectrum"

       @property
       def supports_multiband(self) -> bool:
           return True

       def assign(self, path: list, request: Any) -> dict | None:
           # Find available slots
           slots_needed = self._calculate_slots_needed(request)

           # YOUR ASSIGNMENT LOGIC HERE
           start_slot, end_slot, core, band = self._find_assignment(
               path, slots_needed
           )

           if start_slot is None:
               return None

           # Return assignment result
           return {
               "start_slot": start_slot,
               "end_slot": end_slot,
               "core_num": core,
               "current_band": band,
               "is_free": True,
           }

       def check_spectrum_availability(
           self, path, start_slot, end_slot, core_num, band
       ) -> bool:
           # Check if slots are free on all links
           pass

       def allocate_spectrum(
           self, path, start_slot, end_slot, core_num, band, request_id
       ) -> bool:
           # Mark slots as occupied
           pass

       def deallocate_spectrum(
           self, path, start_slot, end_slot, core_num, band
       ) -> bool:
           # Mark slots as free
           pass

       def get_fragmentation_metric(self, path) -> float:
           # Calculate fragmentation (0.0-1.0)
           pass

       def get_metrics(self) -> dict[str, Any]:
           return {"algorithm": self.algorithm_name}

**Step 2: Register in registry.py**

Add to the ``_register_default_algorithms`` method in ``SpectrumRegistry``:

.. code-block:: python

   from .my_spectrum import MySpectrum

   algorithm_classes = [
       # ... existing algorithms
       MySpectrum,
   ]

   algorithm_name_mapping = {
       # ... existing mappings
       MySpectrum: "my_spectrum",
   }

**Step 3: Export in __init__.py**

.. code-block:: python

   from .my_spectrum import MySpectrum

   __all__ = [
       # ... existing exports
       "MySpectrum",
   ]

**Step 4: Add tests**

Create ``tests/test_my_spectrum.py`` following the AAA pattern.

Configuration
=============

The spectrum algorithm is selected via configuration:

.. list-table:: Configuration Options
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``spectrum_assignment``
     - ``first_fit``
     - Algorithm name from registry
   * - ``guard_slots``
     - ``1``
     - Guard band slots between allocations
   * - ``cores_per_link``
     - ``1``
     - Number of cores (for MCF)
   * - ``band_list``
     - ``["c"]``
     - Supported frequency bands

**INI Configuration Example:**

.. code-block:: ini

   [simulation_settings]
   spectrum_assignment = best_fit
   guard_slots = 1
   cores_per_link = 7
   band_list = c,l

Testing
=======

:Test Location: ``fusion/modules/tests/spectrum/``
:Run Tests: ``pytest fusion/modules/tests/spectrum/ -v``

**Existing Tests:**

- ``test_first_fit.py``: Tests First-Fit assignment
- ``test_best_fit.py``: Tests Best-Fit assignment
- ``test_last_fit.py``: Tests Last-Fit assignment
- ``test_light_path_slicing.py``: Tests slicing manager
- ``test_registry.py``: Tests algorithm registry
- ``test_utils.py``: Tests helper utilities

**Adding New Tests:**

.. code-block:: python

   # tests/test_my_spectrum.py
   import pytest
   import numpy as np
   from fusion.modules.spectrum.my_spectrum import MySpectrum


   @pytest.fixture
   def spectrum_state():
       """Create test spectrum state."""
       # 10 links, 100 slots each
       return np.zeros((10, 100), dtype=int)


   def test_my_spectrum_assigns_slots(spectrum_state):
       """Test that algorithm assigns valid slots."""
       engine_props = {"guard_slots": 1, "cores_per_link": 1}
       sdn_props = type("SDNProps", (), {"spectrum": spectrum_state})()
       route_props = type("RouteProps", (), {"slots_needed": 5})()

       assigner = MySpectrum(engine_props, sdn_props, route_props)
       result = assigner.assign(path=[0, 1, 2], request=None)

       assert result is not None
       assert result["is_free"] is True
       assert result["end_slot"] - result["start_slot"] + 1 >= 5

Troubleshooting
===============

**Issue: No spectrum available**

:Symptom: ``assign()`` returns ``None`` or ``is_free: False``
:Cause: No contiguous block of required size exists
:Solution: Check fragmentation level; consider light path slicing

**Issue: High blocking probability**

:Symptom: Many requests blocked despite available total spectrum
:Cause: Spectrum fragmentation preventing contiguous allocation
:Solution: Use Best-Fit algorithm or implement defragmentation

**Issue: Multi-core assignment not working**

:Symptom: All assignments on core 0
:Cause: ``cores_per_link`` not configured or core selection disabled
:Solution: Verify ``engine_props["cores_per_link"] > 1``

Related Documentation
=====================

- :ref:`modules-directory` - Overview of all FUSION modules
- :ref:`routing-module` - Routing algorithms (first half of RSA)
- :ref:`core-module` - How core uses spectrum via adapters
- :ref:`interfaces-module` - ``AbstractSpectrumAssigner`` interface

.. seealso::

   - `Elastic Optical Networks Overview <https://en.wikipedia.org/wiki/Elastic_optical_network>`_
   - `Spectrum Fragmentation in EON <https://ieeexplore.ieee.org/document/7105631>`_

.. _spectrum-visualization:

Visualization Submodule (BETA)
==============================

.. note::

   **Status: BETA**

   The visualization submodule is in BETA and actively being developed.
   The API may evolve in future releases.

The visualization submodule provides a plugin that extends FUSION's core
visualization system (``fusion/visualization/``) with spectrum-specific
plot types and metrics.

**What It Provides:**

- ``spectrum_heatmap``: Utilization heatmap showing slot usage across links
- ``fragmentation_plot``: Fragmentation analysis with traffic correlation

**Registered Metrics:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Description
   * - ``spectrum_utilization``
     - Slot utilization percentage across links
   * - ``fragmentation_index``
     - Fragmentation measure (0=none, 1=max)
   * - ``avg_fragment_size``
     - Average size of contiguous free blocks
   * - ``largest_fragment``
     - Size of largest available block
   * - ``spectrum_efficiency``
     - Spectral efficiency (bps/Hz)

**Usage:**

.. code-block:: python

   from fusion.visualization.plugins import get_global_registry

   # Load the plugin
   registry = get_global_registry()
   registry.discover_plugins()
   registry.load_plugin("spectrum")

   # Generate plots via standard API
   from fusion.visualization.application.use_cases.generate_plot import generate_plot

   result = generate_plot(
       config_path="my_experiment.yml",
       plot_type="spectrum_heatmap",
       output_path="plots/spectrum.png",
   )

For full details, see the docstrings in ``spectrum_plugin.py``.
