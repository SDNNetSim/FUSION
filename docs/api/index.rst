=============
API Reference
=============

Complete API documentation for all FUSION modules.

.. contents:: Table of Contents
   :local:
   :depth: 2

Package Overview
================

FUSION is organized into the following main packages:

Core Packages
-------------

:doc:`core`
   Core simulation engine, SDN controller, routing, spectrum assignment, and SNR measurements

:doc:`modules`
   Pluggable algorithm implementations:

   * **Routing**: Path computation algorithms
   * **Spectrum**: Spectrum assignment strategies
   * **SNR**: Signal-to-noise ratio calculation
   * **ML**: Machine learning algorithms and utilities
   * **RL**: Reinforcement learning agents, algorithms, and environments

:doc:`sim`
   High-level simulation pipelines and workflow orchestration

Configuration & CLI
-------------------

:doc:`configs`
   Configuration management, validation, and schema definition

:doc:`cli`
   Command-line interface for running simulations and training

Data & I/O
----------

:doc:`io`
   Data generation, export, and structured file handling

:doc:`utils`
   General-purpose utilities (logging, network, random, spectrum operations)

Interfaces
----------

:doc:`interfaces`
   Abstract base classes for plugin architecture

Analysis & Reporting
--------------------

:doc:`analysis`
   Network analysis and metrics computation

:doc:`reporting`
   Simulation reporting and results generation

:doc:`visualization`
   Plotting and visualization tools

Integration
-----------

:doc:`unity`
   Unity ML-Agents integration for distributed RL training

API Documentation
=================

.. toctree::
   :maxdepth: 2

   core
   modules
   sim
   configs
   cli
   io
   utils
   interfaces
   analysis
   reporting
   visualization
   unity
