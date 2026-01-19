============
Architecture
============

FUSION system architecture and design.

.. contents:: Table of Contents
   :local:
   :depth: 2

System Overview
===============

FUSION uses a modular, plugin-based architecture:

* **Core**: Simulation engine, SDN controller
* **Modules**: Pluggable algorithms (routing, spectrum, ML, RL)
* **Interfaces**: Abstract base classes for plugins
* **Configs**: Configuration management
* **CLI**: Command-line interface
* **I/O**: Data generation and export
* **Visualization**: Plotting and analysis

.. image:: /_static/architecture_diagram.svg
   :alt: FUSION system architecture diagram showing five layers: CLI and Configuration at top, Simulation Orchestration layer, Core Engine with SDN Controller and Network Graph, Abstract Interfaces layer, and Modules layer with pluggable routing, spectrum, SNR, ML, and RL algorithms at bottom
   :align: center
   :width: 90%

The architecture follows a clean separation of concerns with well-defined interfaces between layers.

Plugin Architecture
===================

.. image:: /_static/plugin_system.svg
   :alt: Diagram of FUSION's registry-based plugin system showing abstract RouterInterface at top, registry system in middle using @register_router decorator, and four plugin implementations (K-Shortest Paths, Dijkstra, Adaptive, and Custom Algorithm) at bottom, illustrating how new algorithms can be added without modifying core code
   :align: center
   :width: 90%

FUSION uses a registry-based plugin system for all algorithms. This allows you to easily extend
functionality without modifying core code.

**Defining a Plugin:**

.. code-block:: python

   from fusion.modules.routing.registry import register_router

   @register_router("my_algorithm")
   class MyRouter(RouterInterface):
       def calculate_paths(self, graph, src, dst):
           # Implementation
           pass

**Using Plugins:**

.. code-block:: ini

   [general_settings]
   route_method = my_algorithm

The registry system automatically discovers and registers all decorated classes at import time.

Key Components
==============

**SDN Controller**
   Central network management and decision-making

**Simulation Engine**
   Event-driven simulation of connection requests

**Registry System**
   Dynamic algorithm discovery and loading

**Configuration System**
   INI-based configuration with validation

See Also
========

* :doc:`extending` - Add custom algorithms
* :doc:`../api/index` - API reference
