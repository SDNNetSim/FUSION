.. _configs-module:

==============
Configs Module
==============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Unified configuration management system for simulation parameters
   :Location: ``fusion/configs/``
   :Key Files: ``config.py``, ``validate.py``, ``registry.py``, ``cli_to_config.py``, ``schema.py``
   :Depends On: ``configparser``, ``json``, ``dataclasses``
   :Used By: :ref:`cli-module`, ``fusion.core.simulation``, ``fusion.core.orchestrator``

The ``configs`` module is the central configuration hub for FUSION. It handles loading,
validating, and distributing configuration parameters throughout the simulation pipeline.
Whether you're running a quick test or a complex RL experiment, all parameters flow
through this module.

.. rubric:: In This Section

* :doc:`tutorials` - Step-by-step guides for common tasks
* :doc:`examples` - Complete configuration examples for each use case

**When you work here:**

- Adding new configuration parameters
- Creating configuration templates for new experiment types
- Modifying validation rules for configuration values
- Integrating configuration with new simulation features

Key Concepts
============

Configuration File Formats
--------------------------

INI Format (Primary)
   The default and most commonly used format. Supports sections, comments, and
   type inference. Process-specific overrides use ``[s1]``, ``[s2]`` sections.

JSON Format
   Full hierarchical structure. Useful for programmatic configuration generation.
   Validated against JSON schemas in ``schemas/`` directory.

YAML Format
   Human-readable alternative to JSON. Supports anchors and references for
   reducing duplication.

Configuration Sections
----------------------

The configuration is organized into logical sections:

**Core Sections** (Used by all simulation modes)

general_settings
   Traffic load parameters, simulation control, iteration counts

topology_settings
   Network topology, fiber cores, bandwidth configuration

spectrum_settings
   Optical spectrum band configuration (C-band slots)

snr_settings
   Signal-to-noise ratio and modulation parameters

file_settings
   Output file format and paths

**Legacy Sections** (Pre-v6.0, being phased out)

rl_settings
   Reinforcement learning parameters (37 options)

ml_settings
   Machine learning model configuration (6 options)

**Orchestrator Sections** (v6.0+, recommended for new work)

policy_settings
   Routing policy configuration

heuristic_settings
   Heuristic algorithm parameters

protection_settings
   1+1 protection configuration

failure_settings
   Failure injection setup

routing_settings
   Routing algorithm parameters

reporting_settings
   Results export configuration

Configuration Profiles
----------------------

Predefined configurations optimized for specific use cases:

quick_test
   Fast development iteration (max_iters=1, num_requests=50)

development
   Debug mode with verbose output (print_step=5, save_snapshots=true)

production
   Optimized for full experiments (max_iters=10, thread_erlangs=true)

rl_experiment
   RL training setup (n_trials=50, optimize_hyperparameters=true)

benchmark
   Performance testing (max_iters=20, num_requests=2000)

.. tip::

   Use profiles via ``ConfigRegistry.create_profile_config('quick_test')`` during
   development to avoid writing configuration files.

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/configs/
   ├── Core Components
   │   ├── __init__.py           # Public API exports
   │   ├── config.py             # ConfigManager - central config management
   │   ├── validate.py           # SchemaValidator - JSON schema validation
   │   ├── registry.py           # ConfigRegistry - templates and profiles
   │   ├── cli_to_config.py      # CLIToConfigMapper - CLI argument mapping
   │   ├── schema.py             # Type definitions and converters
   │   ├── errors.py             # Custom exception classes
   │   └── constants.py          # Module constants and paths
   ├── schemas/
   │   ├── main.json             # Core configuration schema
   │   └── survivability.json    # Survivability experiment schema
   ├── templates/
   │   ├── minimal.ini           # Fast testing template
   │   ├── default.ini           # Production baseline
   │   ├── cross_platform.ini    # CI/CD portable config
   │   ├── runtime_config.ini    # Multi-process configuration
   │   └── xtar_example_config.ini  # Cross-talk aware routing
   └── tests/
       └── test_*.py             # Unit tests

Visual: How Configs Works Internally
------------------------------------

The following diagram shows how configuration flows within the configs module:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                        CONFIGURATION SOURCES                            │
   └─────────────────────────────────────────────────────────────────────────┘
                │                    │                    │
                ▼                    ▼                    ▼
         ┌──────────┐         ┌──────────┐         ┌──────────┐
         │ INI File │         │ Template │         │ Profile  │
         │ (user)   │         │ (preset) │         │ (preset) │
         └────┬─────┘         └────┬─────┘         └────┬─────┘
              │                    │                    │
              │    ┌───────────────┴────────────────────┘
              │    │
              ▼    ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                         ConfigRegistry                                  │
   │  ┌───────────────────────┐  ┌────────────────────────┐                  │
   │  │ list_templates()      │  │ get_config_profiles    │                  │
   │  │ load_template()       │  │ create_profile_config()│                  │
   │  │ create_custom_config()│  │                        │                  │
   │  │                       │  │                        │                  │
   │  └───────────────────────┘  └────────────────────────┘                  │
   └─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                         ConfigManager                                   │
   │  ┌─────────────────────────────────────────────────────────────────┐    │
   │  │ 1. load_config()  - Auto-detect format (.ini/.json/.yaml)       │    │
   │  │ 2. _parse_value() - Type inference for INI values               │    │
   │  │ 3. merge_cli_args() - Apply CLI overrides                       │    │
   │  │ 4. get_config() - Return SimulationConfig dataclass             │    │
   │  └─────────────────────────────────────────────────────────────────┘    │
   └─────────────────────────────────────────────────────────────────────────┘
              │
              ├──────────────────────────┐
              ▼                          ▼
   ┌─────────────────────┐    ┌─────────────────────┐
   │  SchemaValidator    │    │  CLIToConfigMapper  │
   │  ┌───────────────┐  │    │  ┌───────────────┐  │
   │  │ validate()    │  │    │  │ map_args_to_  │  │
   │  │ against JSON  │  │    │  │ config()      │  │
   │  │ schemas       │  │    │  │ 164 mappings  │  │
   │  └───────────────┘  │    │  └───────────────┘  │
   └─────────────────────┘    └─────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                      SimulationConfig (Output)                         │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
   │  │ general:    │  │ topology:   │  │ spectrum:   │  │ snr:        │    │
   │  │ dict        │  │ dict        │  │ dict        │  │ dict        │    │
   │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
   │  │ rl: dict    │  │ ml: dict    │  │ file: dict  │                     │
   │  └─────────────┘  └─────────────┘  └─────────────┘                     │
   └────────────────────────────────────────────────────────────────────────┘

Type Inference Pipeline
-----------------------

When loading INI files, values go through a type inference pipeline:

.. code-block:: text

   INI String Value
         │
         ▼
   ┌─────────────────┐
   │ Try JSON parse  │ ─── Success ──→ dict, list, or JSON primitive
   │ (handles arrays │
   │  and objects)   │
   └────────┬────────┘
            │ Fail
            ▼
   ┌─────────────────┐
   │ Try int()       │ ─── Success ──→ integer
   └────────┬────────┘
            │ Fail
            ▼
   ┌─────────────────┐
   │ Try float()     │ ─── Success ──→ float
   └────────┬────────┘
            │ Fail
            ▼
   ┌─────────────────┐
   │ Boolean check   │ ─── "true"/"false" ──→ bool
   │ (case-insens.)  │
   └────────┬────────┘
            │ Fail
            ▼
   ┌─────────────────┐
   │ Keep as string  │ ──→ str
   └─────────────────┘

**Example type inference:**

.. code-block:: ini

   [general_settings]
   erlang_start = 300                    ; → int(300)
   holding_time = 3600.5                 ; → float(3600.5)
   is_training = true                    ; → bool(True)
   network = NSFNet                      ; → str("NSFNet")
   request_distribution = {"100": 1.0}   ; → dict({"100": 1.0})

Visual: Config Integration with CLI
-----------------------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                            USER INVOCATION                              │
   │                                                                         │
   │  python -m fusion.cli.run_sim run_sim                                   │
   │      --config_path config.ini                                           │
   │      --erlang_start 500                                                 │
   │      --k_paths 5                                                        │
   │      --run_id my_experiment                                             │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      fusion/cli/main_parser.py                          │
   │                                                                         │
   │  ArgumentParser built using ArgumentRegistry                            │
   │  Parameters defined in fusion/cli/parameters/*.py                       │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      fusion/cli/config_setup.py                         │
   │                                                                         │
   │  1. Load INI file: config.read(args.config_path)                        │
   │  2. Process required options (SIM_REQUIRED_OPTIONS_DICT)                │
   │  3. Process optional options (OPTIONAL_OPTIONS_DICT)                    │
   │  4. Apply CLI overrides to ALL processes                                │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ INI Base Config │  │ CLI Overrides   │  │ Result          │
   │                 │  │                 │  │                 │
   │ erlang_start:   │  │ erlang_start:   │  │ erlang_start:   │
   │ 300             │  │ 500             │  │ 500  (CLI wins) │
   │                 │  │                 │  │                 │
   │ k_paths: 3      │  │ k_paths: 5      │  │ k_paths: 5      │
   └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    CLIToConfigMapper.map_args_to_config()               │
   │                                                                         │
   │  Converts flat CLI args to hierarchical config structure:               │
   │                                                                         │
   │  CLI: --erlang_start 500  →  general_settings.erlang_start = 500        │
   │  CLI: --k_paths 5         →  general_settings.k_paths = 5               │
   │  CLI: --policy-type ksp_ff →  policy_settings.policy_type = ksp_ff      │
   │                                                                         │
   │  (164 argument mappings total)                                          │
   └─────────────────────────────────────────────────────────────────────────┘

.. note::

   CLI arguments override values for **ALL processes** (s1, s2, etc.).
   See :ref:`cli-module` for the complete CLI argument reference.

Visual: Config Integration with Simulation
------------------------------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                fusion/configs SimulationConfig                          │
   │                   (output from ConfigManager)                           │
   │            Contains: general, topology, spectrum, snr dicts             │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    fusion/cli/config_setup.py                           │
   │                                                                         │
   │  Flattens and merges all settings → engine_props dict                   │
   │                                                                         │
   │  engine_props = {                                                       │
   │      "erlang_start": 500,                                               │
   │      "network": "NSFNet",                                               │
   │      "k_paths": 5,                                                      │
   │      "route_method": "k_shortest_path",                                 │
   │      ...  (all merged settings)                                         │
   │  }                                                                      │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
   ┌───────────────────────────┐     ┌───────────────────────────┐
   │    Legacy Path            │     │    Orchestrator Path      │
   │    (use_orchestrator=     │     │    (use_orchestrator=     │
   │     False)                │     │     True)                 │
   └─────────────┬─────────────┘     └─────────────┬─────────────┘
                 │                                 │
                 │                                 ▼
                 │                   ┌───────────────────────────┐
                 │                   │ SimulationConfig.from_    │
                 │                   │ engine_props(engine_props)│
                 │                   │                           │
                 │                   │ Converts dict to frozen   │
                 │                   │ dataclass with validation │
                 │                   │ and computed properties   │
                 │                   │ (fusion/domain/config.py) │
                 │                   └─────────────┬─────────────┘
                 │                                 │
                 ▼                                 ▼
   ┌───────────────────────────┐     ┌───────────────────────────┐
   │     SDNController         │     │     SDNOrchestrator       │
   │                           │     │                           │
   │  Uses engine_props dict:  │     │  Uses SimulationConfig:   │
   │  - engine_props["k_paths"]│     │  - config.k_paths         │
   │  - engine_props["network"]│     │  - config.network_name    │
   │  - Mutable, unvalidated   │     │  - Immutable, typed       │
   │                           │     │  - config.total_slots     │
   │                           │     │  - config.is_multicore    │
   └───────────────────────────┘     └───────────────────────────┘
                 │                                 │
                 │                                 ▼
                 │                   ┌───────────────────────────┐
                 │                   │     PolicyFactory         │
                 │                   │                           │
                 │                   │  Uses PolicyConfig for    │
                 │                   │  policy instantiation:    │
                 │                   │  - policy_type            │
                 │                   │  - policy_name            │
                 │                   │  - model_path             │
                 │                   └───────────────────────────┘

Data Structures
===============

This section documents the data structures at each stage of configuration processing.

Input: INI File Structure
-------------------------

.. code-block:: ini

   ; Required section - must be present
   [general_settings]
   erlang_start = 300
   erlang_stop = 1200
   erlang_step = 100
   max_iters = 4
   num_requests = 1000
   holding_time = 3600
   network = NSFNet
   k_paths = 3

   ; Optional sections
   [topology_settings]
   cores_per_link = 3
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [snr_settings]
   snr_type = None
   xt_type = None

   [file_settings]
   file_type = json

   ; Process-specific overrides
   [s1]
   k_paths = 5

   [s2]
   k_paths = 7

Intermediate: Parsed Config Dict
--------------------------------

After ConfigManager parses the INI file:

.. code-block:: python

   {
       "general_settings": {
           "erlang_start": 300,          # int (inferred)
           "erlang_stop": 1200,          # int
           "erlang_step": 100,           # int
           "max_iters": 4,               # int
           "num_requests": 1000,         # int
           "holding_time": 3600,         # int
           "network": "NSFNet",          # str
           "k_paths": 3,                 # int
       },
       "topology_settings": {
           "cores_per_link": 3,          # int
           "bw_per_slot": 12.5,          # float
       },
       "spectrum_settings": {
           "c_band": 320,                # int
       },
       "snr_settings": {
           "snr_type": None,             # NoneType (parsed from "None")
           "xt_type": None,              # NoneType
       },
       "file_settings": {
           "file_type": "json",          # str
       },
       "s1": {
           "k_paths": 5,                 # Process-specific override
       },
       "s2": {
           "k_paths": 7,                 # Process-specific override
       },
   }

Output: SimulationConfig Dataclass
----------------------------------

The final output is a ``SimulationConfig`` dataclass:

.. code-block:: python

   @dataclass
   class SimulationConfig:
       """Structured configuration for simulation."""
       general: dict[str, Any]      # general_settings
       topology: dict[str, Any]     # topology_settings
       spectrum: dict[str, Any]     # spectrum_settings
       snr: dict[str, Any]          # snr_settings
       rl: dict[str, Any]           # rl_settings (optional)
       ml: dict[str, Any]           # ml_settings (optional)
       file: dict[str, Any]         # file_settings

   # Example access:
   config = ConfigManager('config.ini').get_config()
   config.general['erlang_start']  # 300
   config.topology['network']      # "NSFNet"
   config.spectrum['c_band']       # 320

Final: Simulation Engine Configuration
--------------------------------------

The configuration takes different forms depending on which simulation path is used:

**Legacy Path** (``use_orchestrator=False``): engine_props Dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The legacy simulation engine receives a flattened ``engine_props`` dict:

.. code-block:: python

   engine_props = {
       # From general_settings
       "erlang_start": 300,
       "erlang_stop": 1200,
       "erlang_step": 100,
       "max_iters": 4,
       "num_requests": 1000,
       "holding_time": 3600,
       "network": "NSFNet",
       "k_paths": 3,

       # From topology_settings
       "cores_per_link": 3,
       "bw_per_slot": 12.5,

       # From spectrum_settings
       "c_band": 320,

       # From snr_settings
       "snr_type": None,
       "xt_type": None,

       # From file_settings
       "file_type": "json",

       # Derived/computed values
       "output_dir": Path("/path/to/output"),
       "run_id": "my_experiment",

       # Process-specific (for this process)
       "process_name": "s1",
   }

**Orchestrator Path** (``use_orchestrator=True``): SimulationConfig Dataclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator (v6.0+) uses an immutable ``SimulationConfig`` frozen dataclass
defined in ``fusion/domain/config.py``:

.. code-block:: python

   from fusion.domain.config import SimulationConfig

   # Create from engine_props dict
   config = SimulationConfig.from_engine_props(engine_props)

   # Access typed, validated attributes
   config.network_name           # "NSFNet"
   config.k_paths                # 3
   config.cores_per_link         # 3
   config.band_slots             # {"c": 320}
   config.grooming_enabled       # False
   config.protection_enabled     # False (computed property)

   # Computed properties
   config.total_slots            # 320 (sum of all bands)
   config.arrival_rate           # computed from erlang/holding_time
   config.is_multicore           # True if cores_per_link > 1
   config.is_multiband           # True if len(band_list) > 1

   # Convert back to legacy dict if needed
   legacy_dict = config.to_engine_props()

**Key differences:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Legacy (engine_props)
     - Orchestrator (SimulationConfig)
   * - Type
     - ``dict[str, Any]``
     - Frozen ``@dataclass``
   * - Mutability
     - Mutable
     - Immutable (frozen=True)
   * - Type Safety
     - No type checking
     - Typed attributes with validation
   * - Validation
     - Manual
     - Automatic in ``__post_init__``
   * - Computed Properties
     - Must compute manually
     - Built-in (``total_slots``, ``arrival_rate``, etc.)
   * - Location
     - Created in ``config_setup.py``
     - ``fusion/domain/config.py``

**PolicyConfig for Orchestrator Policies:**

The orchestrator also uses ``PolicyConfig`` (from ``fusion/policies/policy_factory.py``)
for policy-specific configuration:

.. code-block:: python

   from fusion.policies.policy_factory import PolicyConfig, PolicyFactory

   # Create policy configuration
   policy_config = PolicyConfig(
       policy_type="heuristic",       # "heuristic", "ml", or "rl"
       policy_name="shortest",        # Policy variant name
       model_path=None,               # Path to model (for ml/rl)
       fallback_policy="first_feasible",
       k_paths=3,
       seed=42,
   )

   # Create policy instance
   policy = PolicyFactory.create(policy_config)

Components
==========

config.py
---------

:Purpose: Central configuration management class
:Key Classes: ``ConfigManager``, ``SimulationConfig``

The ``ConfigManager`` is the primary interface for loading and accessing configuration:

.. code-block:: python

   from fusion.configs import ConfigManager

   # Load from file
   config_manager = ConfigManager('path/to/config.ini')

   # Get structured config
   config = config_manager.get_config()
   print(config.general['erlang_start'])

   # Merge CLI arguments
   config_manager.merge_cli_args({'max_iters': 10})

   # Save to different format
   config_manager.save_config('output.json', format_type='json')

   # Get module-specific config
   routing_config = config_manager.get_module_config('routing')

validate.py
-----------

:Purpose: JSON schema-based validation
:Key Classes: ``SchemaValidator``, ``ValidationError``

Validates configuration against JSON schemas:

.. code-block:: python

   from fusion.configs import SchemaValidator, ValidationError

   validator = SchemaValidator(schema_dir='fusion/configs/schemas')

   # Validate against main schema
   try:
       validator.validate(config_dict, 'main')
   except ValidationError as e:
       print(f"Validation failed: {e}")

   # Generate default config from schema
   defaults = validator.get_default_config('main')

   # Validate survivability config
   validator.validate_survivability_config(config_dict)

registry.py
-----------

:Purpose: Template and profile management
:Key Classes: ``ConfigRegistry``

Manages configuration templates and predefined profiles:

.. code-block:: python

   from fusion.configs import ConfigRegistry

   registry = ConfigRegistry()

   # List available templates
   templates = registry.list_templates()
   # ['minimal', 'default', 'cross_platform', 'runtime_config', 'xtar_example_config']

   # Load a template
   config_manager = registry.load_template('minimal')

   # Get available profiles
   profiles = registry.get_config_profiles()
   # {'quick_test': {...}, 'development': {...}, 'production': {...}, ...}

   # Create config from profile
   config_manager = registry.create_profile_config('quick_test')

   # Create custom config with overrides
   config_manager = registry.create_custom_config(
       'default',
       overrides={'general_settings.max_iters': 10}
   )

   # Export current config as new template
   registry.export_config_template(config_manager, 'my_custom_template')

**Profile Details:**

.. list-table:: Configuration Profiles
   :header-rows: 1
   :widths: 15 35 50

   * - Profile
     - Description
     - Key Settings
   * - ``quick_test``
     - Fast development testing
     - max_iters=1, num_requests=50
   * - ``development``
     - Debug mode with verbose output
     - print_step=5, save_snapshots=true
   * - ``production``
     - Optimized for real experiments
     - max_iters=10, thread_erlangs=true
   * - ``rl_experiment``
     - RL training configuration
     - n_trials=50, optimize_hyperparameters=true
   * - ``benchmark``
     - Performance and stress testing
     - max_iters=20, num_requests=2000

cli_to_config.py
----------------

:Purpose: Maps CLI arguments to configuration structure
:Key Classes: ``CLIToConfigMapper``

Bridges CLI arguments and configuration sections:

.. code-block:: python

   from fusion.configs import CLIToConfigMapper

   mapper = CLIToConfigMapper()

   # Convert CLI args dict to hierarchical config
   cli_args = {
       'erlang_start': 500,
       'k_paths': 5,
       'policy_type': 'ksp_ff',
   }
   config_dict = mapper.map_args_to_config(cli_args)
   # {
   #     'general_settings': {'erlang_start': 500, 'k_paths': 5},
   #     'policy_settings': {'policy_type': 'ksp_ff'},
   # }

   # Get reverse mapping (config path -> CLI arg name)
   reverse = mapper.get_reverse_mapping()
   # {'general_settings.erlang_start': 'erlang_start', ...}

schema.py
---------

:Purpose: Configuration parameter definitions and type converters
:Key Data: ``SIM_REQUIRED_OPTIONS_DICT``, ``OPTIONAL_OPTIONS_DICT``

Defines all configuration parameters with their types:

.. code-block:: python

   # Required parameters (must be in INI file)
   SIM_REQUIRED_OPTIONS_DICT = {
       "general_settings": {
           "erlang_start": float,
           "erlang_stop": float,
           "erlang_step": float,
           "max_iters": int,
           "num_requests": int,
           # ... (36 total)
       },
   }

   # Optional parameters (use defaults if not specified)
   OPTIONAL_OPTIONS_DICT = {
       "general_settings": {
           "k_paths": (int, 3),           # (type, default)
           "print_step": (int, 0),
           # ...
       },
       "policy_settings": {
           "policy_type": (str, "ksp_ff"),
           "policy_name": (str, None),
           # ...
       },
       # ... (200+ options across 9+ sections)
   }

schemas/ Directory
------------------

:Purpose: JSON Schema definitions for validation
:Contents: ``main.json``, ``survivability.json``

JSON schemas define the valid structure and constraints:

**main.json** - Core configuration schema:

.. code-block:: json

   {
     "$schema": "https://json-schema.org/draft/2020-12/schema",
     "type": "object",
     "required": ["general_settings", "topology_settings", "spectrum_settings"],
     "properties": {
       "general_settings": {
         "type": "object",
         "properties": {
           "erlang_start": {"type": "number", "minimum": 0},
           "erlang_stop": {"type": "number", "minimum": 0},
           "max_iters": {"type": "integer", "minimum": 1},
           "num_requests": {"type": "integer", "minimum": 1}
         }
       },
       "topology_settings": {
         "type": "object",
         "properties": {
           "network": {
             "type": "string",
             "enum": ["NSFNet", "Pan-European", "USbackbone60"]
           },
           "cores_per_link": {"type": "integer", "minimum": 1}
         }
       }
     }
   }

**survivability.json** - Failure and protection experiments:

.. code-block:: json

   {
     "failure_settings": {
       "type": "object",
       "properties": {
         "failure_type": {
           "enum": ["none", "link", "node", "srlg", "geo"]
         },
         "t_fail_arrival_index": {"type": "integer"},
         "t_repair_after_arrivals": {"type": "integer"}
       }
     },
     "protection_settings": {
       "type": "object",
       "properties": {
         "protection_switchover_ms": {"type": "number"},
         "restoration_latency_ms": {"type": "number"},
         "revert_to_primary": {"type": "boolean"}
       }
     }
   }

templates/ Directory
--------------------

:Purpose: Pre-built configuration templates
:Contents: ``minimal.ini``, ``default.ini``, ``cross_platform.ini``, ``runtime_config.ini``, ``xtar_example_config.ini``

See :ref:`configuration-tutorials` for detailed examples of each template.

errors.py
---------

:Purpose: Custom exception hierarchy for configuration errors
:Key Classes: ``ConfigError``, ``ConfigFileNotFoundError``, ``ConfigParseError``, ``MissingRequiredOptionError``, ``ConfigTypeConversionError``

Exception hierarchy:

.. code-block:: text

   ConfigError (base)
   ├── ConfigFileNotFoundError     # Config file does not exist
   ├── ConfigParseError            # Invalid INI/JSON/YAML syntax
   ├── MissingRequiredOptionError  # Required parameter missing
   └── ConfigTypeConversionError   # Cannot convert value to expected type

Usage:

.. code-block:: python

   from fusion.configs import (
       ConfigManager,
       ConfigFileNotFoundError,
       MissingRequiredOptionError,
   )

   try:
       config = ConfigManager('missing.ini')
   except ConfigFileNotFoundError as e:
       print(f"File not found: {e}")

   try:
       config = ConfigManager('incomplete.ini')
   except MissingRequiredOptionError as e:
       print(f"Missing required: {e.option_name} in {e.section}")

constants.py
------------

:Purpose: Module constants and paths
:Key Values: ``PROJECT_ROOT``, ``DEFAULT_CONFIG_PATH``, ``CONFIG_DIR_PATH``

.. code-block:: python

   from fusion.configs.constants import (
       PROJECT_ROOT,          # Root project path
       DEFAULT_CONFIG_PATH,   # ini/run_ini/config.ini
       CONFIG_DIR_PATH,       # fusion/configs/
       REQUIRED_SECTION,      # "general_settings"
       DICT_PARAM_OPTIONS,    # Parameters stored as JSON dicts
   )

Dependencies
============

This Module Depends On
----------------------

- ``configparser`` - Standard library INI file parsing
- ``json`` - JSON format handling and type inference
- ``yaml`` (optional) - YAML format support
- ``dataclasses`` - SimulationConfig structure
- ``pathlib`` - Cross-platform path handling

Modules That Depend On This
---------------------------

- :ref:`cli-module` - Uses configs for argument mapping and loading
- ``fusion.core.simulation`` - Receives engine_props from processed config
- ``fusion.core.orchestrator`` - Uses policy and protection settings
- ``fusion.modules.failures`` - Uses failure_settings configuration
- ``fusion.policies`` - Uses policy_settings and offline_rl_settings

Related Documentation
=====================

- :ref:`cli-module` - Command-line interface and argument reference
- ``fusion/core/simulation.py`` - Simulation engine that consumes configuration
- ``CODING_STANDARDS.md`` - Project coding conventions

.. seealso::

   - `Python configparser <https://docs.python.org/3/library/configparser.html>`_ - INI file parsing
   - `JSON Schema <https://json-schema.org/>`_ - Validation schema specification

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tutorials
   examples
