.. _utils-module:

============
Utils Module
============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Common utility functions used throughout FUSION
   :Location: ``fusion/utils/``
   :Key Files: ``config.py``, ``logging_config.py``, ``spectrum.py``, ``network.py``, ``random.py``, ``os.py``, ``data.py``
   :Depends On: ``fusion.configs.constants``, ``fusion.configs.errors``, ``numpy``
   :Used By: Nearly all FUSION modules

The ``utils`` module provides shared utility functions organized by domain:
configuration handling, logging, spectrum management, network analysis,
random number generation, file operations, and data manipulation.

.. important::

   To avoid circular dependencies, import directly from specific modules
   rather than from the package:

   .. code-block:: python

      # Preferred
      from fusion.utils.logging_config import get_logger
      from fusion.utils.spectrum import find_free_slots

      # Avoid (limited exports)
      from fusion.utils import get_logger

Module Summary
==============

.. list-table:: Utils Module Files
   :header-rows: 1
   :widths: 20 45 15 20

   * - File
     - Purpose
     - Functions
     - Category
   * - ``config.py``
     - Configuration type conversion and CLI override handling
     - 5
     - Configuration
   * - ``logging_config.py``
     - Centralized logging setup with rotation and formatting
     - 8 + 1 class
     - Logging
   * - ``os.py``
     - Directory creation and project root discovery
     - 2
     - File I/O
   * - ``random.py``
     - Reproducible random number generation for simulations
     - 3
     - RNG
   * - ``spectrum.py``
     - Spectrum slot/channel finding and overlap detection
     - 8
     - Spectrum
   * - ``network.py``
     - Path analysis, congestion, fragmentation, modulation
     - 6
     - Network
   * - ``data.py``
     - Dictionary sorting and manipulation
     - 2
     - Data

Helper Function Reference
=========================

This section provides a quick reference for all helper functions in each file.
Use this to find the right function for your task.

config.py - Configuration Management
------------------------------------

:Purpose: Type conversion, dictionary parsing, CLI argument handling
:Lines: ~126

.. list-table:: config.py Functions
   :header-rows: 1
   :widths: 28 35 37

   * - Function
     - Signature
     - Description
   * - ``str_to_bool``
     - ``(string: str) -> bool``
     - Convert "true", "yes", "1" to boolean
   * - ``convert_string_to_dict``
     - ``(value: str) -> str | dict``
     - Parse string dict representation using ``ast.literal_eval``
   * - ``apply_cli_override``
     - ``(config_val, cli_val, converter) -> Any``
     - Apply CLI argument override with type conversion
   * - ``safe_type_convert``
     - ``(value: str, converter, option: str) -> Any``
     - Convert with error context; raises ``ConfigTypeConversionError``
   * - ``convert_dict_params_if_needed``
     - ``(value: Any, option: str) -> Any``
     - Auto-convert dict params based on option name

**Usage Example:**

.. code-block:: python

   from fusion.utils.config import str_to_bool, safe_type_convert

   # Boolean conversion
   enabled = str_to_bool("true")   # True
   enabled = str_to_bool("yes")    # True
   enabled = str_to_bool("false")  # False

   # Safe type conversion with error context
   value = safe_type_convert("100", int, "num_requests")

logging_config.py - Centralized Logging
---------------------------------------

:Purpose: Standardized logging setup with file rotation and formatting
:Lines: ~330

.. list-table:: logging_config.py Functions
   :header-rows: 1
   :widths: 32 35 33

   * - Function
     - Signature
     - Description
   * - ``setup_logger``
     - ``(name, level, log_file, ...) -> Logger``
     - Configure logger with console/file handlers and rotation
   * - ``get_logger``
     - ``(name: str, level: str) -> Logger``
     - Get existing or create basic logger (convenience)
   * - ``configure_simulation_logging``
     - ``(sim_name, erlang, thread_num, ...) -> Logger``
     - Configure simulation-specific logging with naming
   * - ``log_function_call``
     - ``(logger: Logger) -> Callable``
     - Decorator to log function calls with args/return
   * - ``set_global_log_level``
     - ``(level: str) -> None``
     - Set log level for all FUSION loggers
   * - ``log_message``
     - ``(message: str, log_queue) -> None``
     - Log to queue (multi-process) or standard logger

.. list-table:: logging_config.py Classes
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Purpose
     - Key Methods
   * - ``LoggerAdapter``
     - Add contextual info to log messages
     - ``__init__(logger, extra)``, ``process(msg, kwargs)``

**Constants:**

- ``DEFAULT_FORMAT``: Standard log format with timestamp, name, level, message
- ``DETAILED_FORMAT``: Includes filename and line number for debugging
- ``LOG_LEVELS``: Dict mapping level names to logging constants

**Usage Example:**

.. code-block:: python

   from fusion.utils.logging_config import get_logger, setup_logger

   # Simple usage
   logger = get_logger(__name__)
   logger.info("Starting simulation")

   # Advanced with file rotation
   logger = setup_logger(
       name="simulation",
       level="DEBUG",
       log_file="sim.log",
       max_bytes=10_485_760,  # 10MB
       backup_count=5
   )

   # Simulation-specific logging
   from fusion.utils.logging_config import configure_simulation_logging
   logger = configure_simulation_logging("NSFNet", erlang=300.0, thread_num=1)

os.py - File and Path Operations
--------------------------------

:Purpose: Directory creation and project root discovery
:Lines: ~63

.. list-table:: os.py Functions
   :header-rows: 1
   :widths: 25 35 40

   * - Function
     - Signature
     - Description
   * - ``create_directory``
     - ``(directory_path: str) -> None``
     - Create directory with parent dirs; raises ``ValueError`` if None
   * - ``find_project_root``
     - ``(start_path: str | None) -> str``
     - Find root by searching for ``.git`` or ``run_sim.py``

**Usage Example:**

.. code-block:: python

   from fusion.utils.os import create_directory, find_project_root

   # Create nested directories
   create_directory("results/experiment_1/data")

   # Find project root from anywhere
   root = find_project_root()
   config_path = os.path.join(root, "configs", "default.ini")

random.py - Random Number Generation
------------------------------------

:Purpose: Reproducible RNG for simulations using numpy
:Lines: ~79

.. list-table:: random.py Functions
   :header-rows: 1
   :widths: 35 30 35

   * - Function
     - Signature
     - Description
   * - ``set_random_seed``
     - ``(seed_value: int) -> None``
     - Set numpy seed for reproducibility
   * - ``generate_uniform_random_variable``
     - ``(scale: float | None) -> float | int``
     - Uniform [0,1] or scaled integer [0, scale)
   * - ``generate_exponential_random_variable``
     - ``(scale: float) -> float``
     - Exponential distribution via inverse transform

**Usage Example:**

.. code-block:: python

   from fusion.utils.random import (
       set_random_seed,
       generate_uniform_random_variable,
       generate_exponential_random_variable
   )

   # Reproducible simulation
   set_random_seed(42)

   # Uniform random for path selection
   rand = generate_uniform_random_variable()  # [0, 1)

   # Scaled integer for node selection
   node = generate_uniform_random_variable(14)  # [0, 14) integer

   # Inter-arrival time (Poisson process)
   inter_arrival = generate_exponential_random_variable(scale=10.0)

spectrum.py - Spectrum Allocation Utilities
-------------------------------------------

:Purpose: Slot/channel finding, overlap detection, multi-core fiber support
:Lines: ~471

.. list-table:: spectrum.py Functions
   :header-rows: 1
   :widths: 32 33 35

   * - Function
     - Signature
     - Description
   * - ``find_free_slots``
     - ``(network_spectrum, link_tuple, ...) -> dict``
     - Find unallocated spectral slots per link/core
   * - ``find_free_channels``
     - ``(network_spectrum, slots_needed, ...) -> dict``
     - Find contiguous free super-channels
   * - ``find_taken_channels``
     - ``(network_spectrum, link_tuple, ...) -> dict``
     - Find occupied super-channels on link
   * - ``get_channel_overlaps``
     - ``(free_channels, free_slots) -> dict``
     - Find overlapping channels between adjacent cores
   * - ``find_common_channels_on_paths``
     - ``(spectrum, paths, slots, band, core) -> list``
     - Find slots available on ALL paths (for 1+1 protection)
   * - ``adjacent_core_indices``
     - ``(core_id: int, cores_per_link: int) -> list``
     - Get adjacent cores for 7/4/13/19-core layouts
   * - ``edge_set``
     - ``(path: list, bidirectional: bool) -> set``
     - Normalize path to set of link tuples
   * - ``get_overlapping_lightpaths``
     - ``(new_lp, lp_list, ...) -> list``
     - Find lightpaths that overlap with new lightpath

**Core Adjacency Diagram:**

.. code-block:: text

   7-Core Fiber Layout:
         1
       /   \
      6  0  2      Core 0 (center) -> adjacent to 1,2,3,4,5,6
       \   /       Core 1 -> adjacent to 0,2,6
      5  4  3      Core 2 -> adjacent to 0,1,3
                   ...etc (hexagonal pattern)

**Usage Example:**

.. code-block:: python

   from fusion.utils.spectrum import (
       find_free_slots,
       find_free_channels,
       adjacent_core_indices
   )

   # Find free slots on a link
   free = find_free_slots(
       network_spectrum=network_state.spectrum,
       link_tuple=(0, 1)
   )
   # Returns: {"c": {0: [0,1,2,5,6], 1: [0,1,2,3,4], ...}}

   # Find contiguous channels for allocation
   channels = find_free_channels(
       network_spectrum=network_state.spectrum,
       slots_needed=4,
       link_tuple=(0, 1)
   )

   # Get adjacent cores for crosstalk analysis
   adjacent = adjacent_core_indices(core_id=0, cores_per_link=7)
   # Returns: [1, 2, 3, 4, 5, 6]

network.py - Network Path and Congestion Utilities
--------------------------------------------------

:Purpose: Path analysis, modulation selection, congestion/fragmentation metrics
:Lines: ~269

.. list-table:: network.py Functions
   :header-rows: 1
   :widths: 28 35 37

   * - Function
     - Signature
     - Description
   * - ``find_path_length``
     - ``(path: list, topology: nx.Graph) -> float``
     - Sum edge weights along path (km)
   * - ``find_core_congestion``
     - ``(core: int, spectrum, path) -> float``
     - Average congestion % on core along path
   * - ``get_path_modulation``
     - ``(formats: dict, length: float) -> str | bool``
     - Choose modulation by max reach; False if too long
   * - ``find_path_congestion``
     - ``(path, spectrum, band) -> tuple[float, float]``
     - Return (avg_congestion, scaled_capacity)
   * - ``find_path_fragmentation``
     - ``(path, spectrum, band) -> float``
     - Compute fragmentation ratio [0,1] along path
   * - ``average_bandwidth_usage``
     - ``(bw_dict, departure_time) -> float``
     - Time-weighted average bandwidth utilization

**Usage Example:**

.. code-block:: python

   from fusion.utils.network import (
       find_path_length,
       get_path_modulation,
       find_path_congestion
   )

   # Calculate path length
   path = [0, 3, 5, 8]
   length = find_path_length(path, topology)  # e.g., 850.5 km

   # Select appropriate modulation
   mod = get_path_modulation(
       modulation_formats=engine_props["mod_per_bw"]["50GHz"],
       path_length=length
   )  # Returns "QPSK", "16-QAM", "64-QAM", or False

   # Check path congestion
   congestion, capacity = find_path_congestion(
       path_list=path,
       network_spectrum=spectrum,
       band="c"
   )

data.py - Data Structure Manipulation
-------------------------------------

:Purpose: Dictionary sorting utilities
:Lines: ~41

.. list-table:: data.py Functions
   :header-rows: 1
   :widths: 30 35 35

   * - Function
     - Signature
     - Description
   * - ``sort_dict_keys``
     - ``(dictionary: dict) -> dict``
     - Sort keys descending (numeric order)
   * - ``sort_nested_dict_values``
     - ``(dict, nested_key: str) -> dict``
     - Sort by nested key value ascending

**Usage Example:**

.. code-block:: python

   from fusion.utils.data import sort_dict_keys, sort_nested_dict_values

   # Sort modulation formats by bandwidth (descending)
   bw_dict = {"100": {...}, "50": {...}, "200": {...}}
   sorted_bw = sort_dict_keys(bw_dict)
   # Returns: {"200": {...}, "100": {...}, "50": {...}}

   # Sort paths by weight
   paths = {
       "path1": {"hops": 3, "weight": 0.8},
       "path2": {"hops": 2, "weight": 0.3},
   }
   sorted_paths = sort_nested_dict_values(paths, "weight")
   # Returns paths sorted by weight ascending

Architecture
============

Module Organization
-------------------

.. code-block:: text

   fusion/utils/
   +-- __init__.py          # Limited exports (get_logger, setup_logger)
   +-- config.py            # Configuration helpers
   +-- logging_config.py    # Logging infrastructure
   +-- os.py                # File/path operations
   +-- random.py            # RNG for simulations
   +-- spectrum.py          # Spectrum allocation
   +-- network.py           # Network analysis
   +-- data.py              # Data manipulation
   +-- tests/               # Unit tests

Dependency Flow
---------------

.. code-block:: text

   External Dependencies:
   +---------------------+
   | numpy               | <-- random.py, network.py
   | networkx            | <-- network.py
   | logging (stdlib)    | <-- logging_config.py
   | pathlib (stdlib)    | <-- os.py
   | ast (stdlib)        | <-- config.py
   +---------------------+

   Internal Dependencies:
   +---------------------+
   | fusion.configs      |
   |   .constants        | <-- os.py (PROJECT_ROOT)
   |   .errors           | <-- config.py (ConfigTypeConversionError)
   +---------------------+

Why Utils Exists Here
---------------------

The ``fusion/utils/`` module exists to:

1. **Avoid circular dependencies**: Common functions that multiple modules need
   (logging, spectrum, network) are centralized here to prevent import cycles
   between ``fusion.core``, ``fusion.sim``, and ``fusion.pipelines``.

2. **Provide shared infrastructure**: Logging configuration, RNG seeding, and
   path utilities are used everywhere and need a single source of truth.

3. **Keep domain modules focused**: By extracting utility functions, domain
   modules like ``fusion.core`` can focus on simulation logic.

.. note::

   There is also a ``fusion/sim/utils/`` module with simulation-specific
   utilities. The distinction:

   - ``fusion/utils/``: General utilities used across the entire codebase
   - ``fusion/sim/utils/``: Simulation orchestration utilities

Common Patterns
===============

Backward Compatibility
----------------------

Many functions support legacy parameter names:

.. code-block:: python

   # Both work - new name preferred
   find_free_slots(network_spectrum=state)       # New
   find_free_slots(network_spectrum_dict=state)  # Legacy

   # Both work - new name preferred
   get_path_modulation(modulation_formats=mods)  # New
   get_path_modulation(mods_dict=mods)           # Legacy

Error Handling
--------------

Functions raise specific exceptions with context:

.. list-table:: Error Handling by Module
   :header-rows: 1
   :widths: 25 35 40

   * - Module
     - Exception
     - Condition
   * - ``config.py``
     - ``ConfigTypeConversionError``
     - Type conversion fails
   * - ``os.py``
     - ``ValueError``
     - Directory path is None
   * - ``os.py``
     - ``RuntimeError``
     - Project root not found
   * - ``random.py``
     - ``ValueError``
     - Negative seed or non-positive scale
   * - ``spectrum.py``
     - ``ValueError``
     - Missing required parameters
   * - ``network.py``
     - ``ValueError``
     - Invalid time progression

Logging Best Practices
----------------------

.. code-block:: python

   # DO: Use get_logger with module name
   logger = get_logger(__name__)

   # DO: Use appropriate log levels
   logger.debug("Detailed debug info")
   logger.info("Normal operation info")
   logger.warning("Something unexpected")
   logger.error("Something failed")

   # DON'T: Use print statements
   print("Debug info")  # Bad!

   # DON'T: Create loggers manually
   logger = logging.getLogger(__name__)  # Use get_logger instead

Related Documentation
=====================

- :ref:`configs-module` - Configuration system using ``config.py``
- :ref:`core-module` - Simulation engine using spectrum/network utilities
- :ref:`sim-module` - Orchestration layer using logging utilities
- :ref:`pipelines-module` - Spectrum assignment using spectrum utilities
