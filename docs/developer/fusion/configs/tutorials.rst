.. _configuration-tutorials:

========================
Configuration Tutorials
========================

This section provides step-by-step tutorials for working with FUSION's configuration
system. Start with the basics and progress to advanced topics.

.. contents:: On This Page
   :local:
   :depth: 2

Getting Started
===============

Your First Configuration File
-----------------------------

This tutorial walks you through creating a minimal configuration file from scratch.

**Step 1: Create the file structure**

Create a new file called ``my_config.ini``:

.. code-block:: ini

   ; My first FUSION configuration
   ; Lines starting with ; are comments

   [general_settings]
   ; These are the REQUIRED parameters - simulation will fail without them

   ; Traffic parameters
   erlang_start = 300
   erlang_stop = 600
   erlang_step = 100

   ; Simulation control
   max_iters = 2
   num_requests = 100
   holding_time = 3600

   ; Network selection
   network = NSFNet

**Step 2: Add topology settings**

.. code-block:: ini

   [topology_settings]
   ; How many cores per fiber
   cores_per_link = 3

   ; Bandwidth per frequency slot (GHz)
   bw_per_slot = 12.5

**Step 3: Add spectrum settings**

.. code-block:: ini

   [spectrum_settings]
   ; Number of frequency slots in C-band
   c_band = 320

**Step 4: Run your simulation**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim --config_path my_config.ini --run_id first_test

**Complete minimal configuration:**

.. code-block:: ini

   ; my_config.ini - Minimal FUSION configuration
   [general_settings]
   erlang_start = 300
   erlang_stop = 600
   erlang_step = 100
   max_iters = 2
   num_requests = 100
   holding_time = 3600
   network = NSFNet

   [topology_settings]
   cores_per_link = 3
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

Using Templates
---------------

Templates provide pre-configured starting points. This tutorial shows how to use them.

**Step 1: List available templates**

.. code-block:: bash

   ls fusion/configs/templates/
   # Output: default.ini  minimal.ini  cross_platform.ini  runtime_config.ini  xtar_example_config.ini

**Step 2: Run with a template directly**

.. code-block:: bash

   # Quick test with minimal template
   python -m fusion.cli.run_sim run_sim \
       --config_path fusion/configs/templates/minimal.ini \
       --run_id template_test

**Step 3: Copy and customize a template**

.. code-block:: bash

   # Copy template to your working directory
   cp fusion/configs/templates/default.ini my_experiment.ini

   # Edit with your preferred editor
   vim my_experiment.ini

**Step 4: Use templates programmatically**

.. code-block:: python

   from fusion.configs import ConfigRegistry

   registry = ConfigRegistry()

   # Load a template
   config_manager = registry.load_template('minimal')

   # Modify values
   config_manager.update_config('general_settings', 'max_iters', 5)

   # Save customized version
   config_manager.save_config('my_custom.ini', format_type='ini')

Using Profiles
--------------

Profiles are pre-defined configurations optimized for specific use cases.

**Step 1: View available profiles**

.. code-block:: python

   from fusion.configs import ConfigRegistry

   registry = ConfigRegistry()
   profiles = registry.get_config_profiles()

   for name, settings in profiles.items():
       print(f"{name}: {settings}")

**Output:**

.. code-block:: text

   quick_test: {'max_iters': 1, 'num_requests': 50}
   development: {'print_step': 5, 'save_snapshots': True}
   production: {'max_iters': 10, 'thread_erlangs': True}
   rl_experiment: {'n_trials': 50, 'optimize_hyperparameters': True}
   benchmark: {'max_iters': 20, 'num_requests': 2000}

**Step 2: Create config from profile**

.. code-block:: python

   # Get config manager with profile settings applied
   config_manager = registry.create_profile_config('quick_test')

   # Use the config
   config = config_manager.get_config()
   print(f"max_iters: {config.general['max_iters']}")  # 1
   print(f"num_requests: {config.general['num_requests']}")  # 50

**Step 3: Combine template with profile overrides**

.. code-block:: python

   # Start from a template, apply profile-like overrides
   config_manager = registry.create_custom_config(
       'default',
       overrides={
           'general_settings.max_iters': 1,
           'general_settings.num_requests': 50,
       }
   )

Multi-Process Configuration
===========================

Running Multiple Configurations in Parallel
-------------------------------------------

FUSION can run multiple simulation configurations simultaneously using process sections.

**Step 1: Understand process sections**

Process sections (``[s1]``, ``[s2]``, etc.) let you override base settings for each
parallel process:

.. code-block:: ini

   [general_settings]
   ; Base configuration - all processes start with these values
   network = NSFNet
   num_requests = 1000
   k_paths = 3
   max_iters = 4

   [s1]
   ; Process 1: Test with k_paths=3 (inherits from base, no override needed)

   [s2]
   ; Process 2: Test with k_paths=5
   k_paths = 5

   [s3]
   ; Process 3: Test with k_paths=7
   k_paths = 7

**Step 2: How inheritance works**

.. code-block:: text

   Base config (general_settings)
         │
         ├──→ s1: k_paths = 3 (inherited)
         │
         ├──→ s2: k_paths = 5 (overridden)
         │
         └──→ s3: k_paths = 7 (overridden)

   All processes share: network, num_requests, max_iters

**Step 3: Running multi-process simulations**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path multi_process.ini \
       --run_id compare_k_paths

The simulation engine will spawn separate processes for s1, s2, and s3.

**Step 4: Comparing different algorithms**

.. code-block:: ini

   [general_settings]
   network = Pan-European
   num_requests = 2000
   max_iters = 5

   [s1]
   ; First-fit allocation
   route_method = k_shortest_path
   allocation_method = first_fit

   [s2]
   ; Best-fit allocation
   route_method = k_shortest_path
   allocation_method = best_fit

   [s3]
   ; Priority-first allocation
   route_method = k_shortest_path
   allocation_method = priority_first

.. warning::

   CLI arguments override values for **ALL processes**. Running with ``--k_paths=10``
   will set k_paths=10 for s1, s2, AND s3, ignoring their individual overrides.

Adding New Configuration Parameters
===================================

When You Need a New Parameter
-----------------------------

Follow this tutorial when you need to add a new configuration option to FUSION.

**Step 1: Decide which section**

Choose the appropriate section based on what the parameter controls:

.. list-table:: Section Selection Guide
   :header-rows: 1
   :widths: 25 75

   * - Section
     - Use For
   * - general_settings
     - Traffic load, iteration control, routing basics
   * - topology_settings
     - Network topology, fiber properties
   * - spectrum_settings
     - Optical spectrum configuration
   * - snr_settings
     - Signal quality, modulation, crosstalk
   * - policy_settings
     - Routing policy selection (v6.0+)
   * - protection_settings
     - 1+1 protection, recovery (v6.0+)
   * - failure_settings
     - Failure injection (v6.0+)

**Step 2: Add to schema.py**

Edit ``fusion/configs/schema.py``:

.. code-block:: python

   # For a required parameter:
   SIM_REQUIRED_OPTIONS_DICT = {
       "general_settings": {
           # ... existing options ...
           "my_new_param": int,  # Add type converter
       },
   }

   # For an optional parameter with default:
   OPTIONAL_OPTIONS_DICT = {
       "general_settings": {
           # ... existing options ...
           "my_new_param": (int, 10),  # (type, default_value)
       },
   }

**Step 3: Add CLI argument (if needed)**

Edit the appropriate file in ``fusion/cli/parameters/``:

.. code-block:: python

   # In fusion/cli/parameters/simulation.py (or appropriate file)
   def add_my_args(parser):
       group = parser.add_argument_group("My Feature")
       group.add_argument(
           "--my_new_param",
           type=int,
           default=10,
           help="Description of what this parameter does",
       )

**Step 4: Add CLI-to-config mapping**

Edit ``fusion/configs/cli_to_config.py``:

.. code-block:: python

   # In the argument mapping dictionary
   _ARGUMENT_MAPPING = {
       # ... existing mappings ...
       "my_new_param": "general_settings.my_new_param",
   }

**Step 5: Update JSON schema (if using validation)**

Edit ``fusion/configs/schemas/main.json``:

.. code-block:: json

   {
     "properties": {
       "general_settings": {
         "properties": {
           "my_new_param": {
             "type": "integer",
             "minimum": 1,
             "maximum": 100,
             "description": "Description of the parameter"
           }
         }
       }
     }
   }

**Step 6: Update template files**

Add the parameter to relevant templates with documentation:

.. code-block:: ini

   ; In fusion/configs/templates/default.ini
   [general_settings]
   ; ... existing options ...

   ; My new parameter - controls XYZ behavior
   ; Valid range: 1-100, default: 10
   my_new_param = 10

**Step 7: Add tests**

Create or update tests in ``fusion/configs/tests/``:

.. code-block:: python

   def test_my_new_param_loaded_correctly():
       config = ConfigManager('test_config.ini')
       assert config.get_config().general['my_new_param'] == 10

   def test_my_new_param_validation():
       # Test that invalid values are rejected
       validator = SchemaValidator()
       invalid_config = {'general_settings': {'my_new_param': -1}}
       with pytest.raises(ValidationError):
           validator.validate(invalid_config, 'main')

Modifying Existing Templates
============================

Customizing Templates for Your Experiments
------------------------------------------

**Step 1: Understand template structure**

Templates have three types of content:

1. **Header comments** - Purpose and authorship
2. **Section comments** - What each section controls
3. **Parameter comments** - What each value means

.. code-block:: ini

   ; =============================================================
   ; FUSION Configuration Template: Default
   ; Purpose: Production baseline with balanced settings
   ; =============================================================

   [general_settings]
   ; --- Traffic Parameters ---
   ; Starting Erlang load for the simulation
   erlang_start = 300

**Step 2: Copy, don't modify originals**

.. code-block:: bash

   # Good: Copy to your project
   cp fusion/configs/templates/default.ini experiments/my_experiment.ini

   # Bad: Don't modify the template directly
   # vim fusion/configs/templates/default.ini  # DON'T DO THIS

**Step 3: Document your changes**

.. code-block:: ini

   ; =============================================================
   ; Custom Configuration: High Load Experiment
   ; Based on: default.ini
   ; Author: Your Name
   ; Date: 2024-01-15
   ; Purpose: Testing system under high traffic load
   ; =============================================================

   [general_settings]
   ; MODIFIED: Increased load range for stress testing
   erlang_start = 1000    ; Was: 300
   erlang_stop = 3000     ; Was: 1200
   erlang_step = 500      ; Was: 100

   ; MODIFIED: More iterations for statistical significance
   max_iters = 10         ; Was: 4

Creating New Templates
----------------------

**Step 1: Start from minimal or default**

.. code-block:: bash

   cp fusion/configs/templates/minimal.ini fusion/configs/templates/my_new_template.ini

**Step 2: Add comprehensive documentation**

.. code-block:: ini

   ; =============================================================
   ; FUSION Configuration Template: My New Template
   ; =============================================================
   ;
   ; PURPOSE:
   ;   [Describe when to use this template]
   ;
   ; USE CASE:
   ;   [Describe the experimental scenario]
   ;
   ; KEY FEATURES:
   ;   - [Feature 1]
   ;   - [Feature 2]
   ;
   ; PREREQUISITES:
   ;   [Any required setup]
   ;
   ; =============================================================

**Step 3: Register in documentation**

Update the templates list in ``fusion/configs/README.md`` and this documentation.

Validation and Error Handling
=============================

Validating Configuration Files
------------------------------

**Step 1: Basic validation**

.. code-block:: python

   from fusion.configs import ConfigManager, ConfigRegistry

   # ConfigManager validates on load
   try:
       config = ConfigManager('my_config.ini')
   except Exception as e:
       print(f"Configuration error: {e}")

**Step 2: Explicit schema validation**

.. code-block:: python

   from fusion.configs import SchemaValidator, ValidationError

   validator = SchemaValidator(schema_dir='fusion/configs/schemas')

   config_dict = {
       'general_settings': {
           'erlang_start': -100,  # Invalid: negative
       }
   }

   try:
       validator.validate(config_dict, 'main')
   except ValidationError as e:
       print(f"Validation error: {e}")
       # "general_settings.erlang_start: -100 is less than minimum 0"

**Step 3: Using ConfigRegistry for validation**

.. code-block:: python

   registry = ConfigRegistry()

   # Validate a file before loading
   is_valid, errors = registry.validate_config('my_config.ini')
   if not is_valid:
       for error in errors:
           print(f"Error: {error}")

Handling Configuration Errors
-----------------------------

**Common errors and solutions:**

.. code-block:: python

   from fusion.configs import (
       ConfigManager,
       ConfigFileNotFoundError,
       ConfigParseError,
       MissingRequiredOptionError,
       ConfigTypeConversionError,
   )

   try:
       config = ConfigManager('config.ini')
   except ConfigFileNotFoundError as e:
       # File doesn't exist
       print(f"Create the file: {e.file_path}")

   except ConfigParseError as e:
       # Invalid INI syntax
       print(f"Fix syntax at line {e.line_number}: {e.message}")

   except MissingRequiredOptionError as e:
       # Required parameter missing
       print(f"Add '{e.option_name}' to [{e.section}] section")

   except ConfigTypeConversionError as e:
       # Value can't be converted to expected type
       print(f"'{e.value}' should be {e.expected_type}, not {type(e.value)}")

Troubleshooting
===============

Common Issues
-------------

**Issue: "Missing required option: erlang_start"**

:Symptom: Simulation fails to start with missing option error
:Cause: Required parameter not in ``[general_settings]`` section
:Solution: Add the parameter to your INI file:

.. code-block:: ini

   [general_settings]
   erlang_start = 300  ; Add this line

**Issue: Type conversion warnings**

:Symptom: Warning about value type conversion
:Cause: Value in INI doesn't match expected type
:Solution: Check the value format:

.. code-block:: ini

   ; Wrong: string instead of number
   erlang_start = three hundred

   ; Correct: numeric value
   erlang_start = 300

**Issue: JSON in INI file not parsing**

:Symptom: Dictionary values appear as strings
:Cause: Invalid JSON syntax
:Solution: Ensure valid JSON:

.. code-block:: ini

   ; Wrong: using = instead of :
   request_distribution = {"100" = 1.0}

   ; Correct: proper JSON syntax
   request_distribution = {"100": 1.0}

**Issue: Process sections not applying**

:Symptom: s1, s2 sections have no effect
:Cause: Section names don't match pattern
:Solution: Use ``s`` followed by a digit:

.. code-block:: ini

   ; Wrong: invalid section names
   [process1]
   [thread_1]

   ; Correct: s + digit
   [s1]
   [s2]

**Issue: CLI args not overriding config**

:Symptom: CLI argument value not used
:Cause: Argument name mismatch
:Solution: Check exact argument name in CLI docs:

.. code-block:: bash

   # Wrong: underscores vs hyphens
   --erlang-start 500

   # Correct: check actual argument name
   --erlang_start 500

Next Steps
==========

- See :ref:`configuration-examples` for complete configuration examples for each use case
- See :ref:`cli-module` for complete CLI argument documentation
- Check ``fusion/configs/templates/`` for more template examples
