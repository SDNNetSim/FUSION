.. _api-reference:

=============
API Reference
=============

This section provides auto-generated API documentation from source code docstrings.
Use this reference for looking up specific classes, functions, and their parameters.

For conceptual documentation, tutorials, and examples, see the :ref:`developer-docs`.

.. tip::

   Use your browser's search (Ctrl+F / Cmd+F) or the documentation search box
   to quickly find specific classes or functions.

.. contents:: Module Index
   :local:
   :depth: 1

----

Configuration System
====================

fusion.configs
--------------

Configuration loading, validation, and management.

.. automodule:: fusion.configs.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.configs.validate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.configs.registry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.configs.cli_to_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.configs.schema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.configs.errors
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.configs.constants
   :members:
   :undoc-members:
   :show-inheritance:

----

Command Line Interface
======================

fusion.cli
----------

CLI entry points and argument parsing.

.. automodule:: fusion.cli.run_sim
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.cli.run_train
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.cli.config_setup
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.cli.main_parser
   :members:
   :undoc-members:
   :show-inheritance:

----

Core Simulation
===============

fusion.core
-----------

Core simulation engine and orchestration.

.. automodule:: fusion.core.simulation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.core.orchestrator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: fusion.core.sdn_controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.core.pipeline_factory
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: fusion.core.request
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.core.metrics
   :members:
   :undoc-members:
   :show-inheritance:

----

Domain Models
=============

fusion.domain
-------------

Domain objects and data structures.

.. automodule:: fusion.domain.config
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: fusion.domain.request
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: fusion.domain.lightpath
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: fusion.domain.network_state
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: fusion.domain.results
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

Interfaces
==========

fusion.interfaces
-----------------

Abstract base classes and protocols.

.. automodule:: fusion.interfaces
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

Policies
========

fusion.policies
---------------

Routing and allocation policies.

.. automodule:: fusion.policies.policy_factory
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: fusion.policies.heuristic_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.policies.rl_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fusion.policies.ml_policy
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

Pipelines
=========

fusion.pipelines
----------------

Processing pipelines for routing, spectrum, and SNR.

.. automodule:: fusion.pipelines
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

Modules
=======

fusion.modules
--------------

Algorithm implementations for routing, spectrum assignment, and SNR.

fusion.modules.routing
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fusion.modules.routing
   :members:
   :undoc-members:
   :show-inheritance:

fusion.modules.spectrum
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fusion.modules.spectrum
   :members:
   :undoc-members:
   :show-inheritance:

fusion.modules.snr
~~~~~~~~~~~~~~~~~~

.. automodule:: fusion.modules.snr
   :members:
   :undoc-members:
   :show-inheritance:

fusion.modules.failures
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fusion.modules.failures
   :members:
   :undoc-members:
   :show-inheritance:

fusion.modules.ml
~~~~~~~~~~~~~~~~~

.. automodule:: fusion.modules.ml
   :members:
   :undoc-members:
   :show-inheritance:

----

Analysis
========

fusion.analysis
---------------

Statistics and results analysis.

.. automodule:: fusion.analysis
   :members:
   :undoc-members:
   :show-inheritance:

----

Reporting
=========

fusion.reporting
----------------

Results export and dataset logging.

.. automodule:: fusion.reporting
   :members:
   :undoc-members:
   :show-inheritance:

----

I/O Operations
==============

fusion.io
---------

File input/output and topology loading.

.. automodule:: fusion.io
   :members:
   :undoc-members:
   :show-inheritance:

----

Utilities
=========

fusion.utils
------------

Helper functions and utilities.

.. automodule:: fusion.utils
   :members:
   :undoc-members:
   :show-inheritance:
