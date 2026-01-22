.. _index:

======
FUSION
======

**Flexible, Unified Simulator for Intelligent Optical Networking**

FUSION is an open-source discrete-event simulation framework for Software-Defined
Elastic Optical Networks (SD-EONs). It provides researchers and engineers with tools
to model, analyze, and optimize optical network behavior under realistic conditions,
with built-in support for reinforcement learning integration and network survivability
experiments.

.. tip::

   **Version 6.0** introduces the Orchestrator architecture with pipeline-based
   request processing. Legacy simulations continue to work unchanged.

----

Why Use FUSION?
===============

FUSION is designed to solve real research problems in optical networking:

**Routing and Spectrum Assignment (RSA)**
   Evaluate algorithms that jointly solve the path selection and spectrum
   allocation problem in elastic optical networks.
   :doc:`Learn more <developer/fusion/modules/routing/index>`

**Network Survivability**
   Test protection and restoration mechanisms against link failures, node
   failures, SRLG events, and geographic disasters.
   :doc:`Learn more <developer/fusion/modules/failures/index>`

**AI/ML Integration**
   Train reinforcement learning agents to make intelligent routing decisions,
   with built-in Stable-Baselines3 integration and action masking for safe exploration.
   :doc:`Learn more <developer/fusion/modules/rl/index>`

**Algorithm Benchmarking**
   Compare routing algorithms (K-shortest-path, congestion-aware, crosstalk-aware)
   and spectrum assignment strategies (First-Fit, Best-Fit, Last-Fit) under
   identical conditions.

**Physical Layer Modeling**
   Account for signal quality constraints including SNR, crosstalk in multi-core
   fiber, and modulation format selection based on path reach.
   :doc:`Learn more <developer/fusion/modules/snr/index>`

----

How FUSION Simulates Networks
=============================

FUSION uses **discrete-event simulation** to model optical network behavior:

.. code-block:: text

   Time    Event
   -----   -------------------------
   0.00    Request 1 arrives (A->D, 100Gbps)
   0.05    Request 2 arrives (B->E, 200Gbps)
   0.12    Request 1 departs
   0.18    Link failure on link (0,1)
   ...

**What Gets Simulated**

- **Traffic arrivals**: Poisson process with configurable Erlang load
- **Request handling**: Routing + spectrum assignment + optional SNR validation
- **Resource allocation**: Spectrum slots on each link, core, and band
- **Network events**: Failures, repairs, protection switchovers
- **Metrics collection**: Blocking probability, fragmentation, utilization

**What You Can Configure**

- Network topology (NSFNet, COST239, USNet, custom)
- Traffic parameters (Erlang load, bandwidth distribution, holding time)
- Algorithm selection (routing, spectrum, SNR measurement)
- Physical layer (cores, bands, guard slots, modulation formats)
- Experiment control (iterations, confidence intervals, parallel threads)

See :doc:`developer/fusion/configs/tutorials` for configuration examples.

----

Core Capabilities
=================

**Simulation Engine**

- Discrete-event simulation with configurable traffic models
- Support for multiple Erlang loads with automatic sweeping
- Confidence interval-based stopping for statistical significance
- Multi-process parallel execution for parameter studies

:doc:`Learn more <developer/fusion/core/index>`

**RSA Algorithms**

- Routing: K-shortest-path, congestion-aware, fragmentation-aware, XT-aware
- Spectrum: First-Fit, Best-Fit, Last-Fit with multi-band and multi-core support
- Pluggable registry pattern for adding custom algorithms

:doc:`Routing <developer/fusion/modules/routing/index>` | :doc:`Spectrum <developer/fusion/modules/spectrum/index>`

**Reinforcement Learning**

- Stable-Baselines3 integration (PPO, A2C, DQN, QR-DQN)
- In-house algorithms (Q-learning, bandits)
- Action masking for safe exploration
- GNN-based feature extractors (GAT, SAGE, GraphConv)
- Optuna hyperparameter optimization

:doc:`Learn more <developer/fusion/modules/rl/index>`

**Network Survivability**

- Failure types: Link (F1), Node (F2), SRLG (F3), Geographic (F4)
- 1+1 disjoint path protection
- Failure injection and recovery metrics

:doc:`Learn more <developer/fusion/modules/failures/index>`

**Physical Layer Modeling**

- SNR calculation with crosstalk for multi-core fiber
- Modulation format selection based on path reach
- Guard band management

:doc:`Learn more <developer/fusion/modules/snr/index>`

----

Choose Your Path
================

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Researcher**
     - Running baseline simulations, comparing algorithms, publishing results.

       Start: :doc:`getting-started/installation` then :doc:`developer/fusion/core/tutorial`

   * - **RL Researcher**
     - Training agents to make routing decisions in optical networks.

       Start: :doc:`getting-started/installation` then :doc:`developer/fusion/modules/rl/index`

   * - **Student**
     - Learning about elastic optical networks and simulation concepts.

       Start: :doc:`getting-started/installation` then :doc:`developer/fusion/core/architecture`

   * - **Contributor**
     - Adding features, fixing bugs, improving documentation.

       Start: :doc:`getting-started/installation`, then read the
       `Development Quickstart <https://github.com/SDNNetSim/FUSION/blob/main/DEVELOPMENT_QUICKSTART.md>`_

   * - **HPC User**
     - Running large-scale experiments on clusters.

       Start: :doc:`getting-started/installation`, then see :doc:`developer/fusion/unity/index`

   * - **GUI User**
     - Prefer a visual interface over the command line.

       Start: :doc:`getting-started/installation`, then :doc:`getting-started/gui/index`

----

Quickstart: Your First Simulation
=================================

Run your first simulation in under 10 minutes:

.. code-block:: bash

   # 1. Clone and install
   git clone git@github.com:SDNNetSim/FUSION.git
   cd FUSION
   python3.11 -m venv venv && source venv/bin/activate
   ./install.sh

   # 2. Run a minimal simulation
   python -m fusion.cli.run_sim --config_path fusion/configs/templates/minimal.ini

   # 3. View results
   ls data/output/NSFNet/
   cat data/output/NSFNet/*/*/s1/*_erlang.json

**What just happened?**

1. You ran a simulation on the NSFNet topology (14 nodes, 21 links)
2. Traffic loads of 300, 400, 500 Erlang were simulated
3. Each load ran for 100 requests over 2 iterations
4. Results include blocking probability, hop counts, and resource utilization

**Next steps**

- :doc:`Customize your configuration <developer/fusion/configs/tutorials>`
- :doc:`Follow the 5-part core tutorial <developer/fusion/core/tutorial>`
- :doc:`Try reinforcement learning <developer/fusion/modules/rl/index>`

----

Web-Based GUI
=============

.. tip::

   **New in v6.1**: FUSION includes a web-based interface for managing
   simulations, visualizing topologies, and exploring the codebase.

Launch the GUI with:

.. code-block:: bash

   python -m fusion.cli.run_gui

Open http://127.0.0.1:8765 in your browser.

**Features:**

- Real-time log streaming for running simulations
- Interactive network topology visualization
- Configuration editor with syntax highlighting
- Codebase explorer with guided architecture tour

:doc:`Learn more <getting-started/gui/index>`

----

Common Tasks
============

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Task
     - Documentation
   * - Launch the GUI
     - :doc:`GUI Getting Started <getting-started/gui/index>`
   * - Run a basic simulation
     - :doc:`Core Tutorial Part 1 <developer/fusion/core/tutorial>`
   * - Create a custom configuration
     - :doc:`Configuration Tutorials <developer/fusion/configs/tutorials>`
   * - Add a new routing algorithm
     - :doc:`Routing Module <developer/fusion/modules/routing/index>`
   * - Train an RL agent
     - :doc:`RL Module <developer/fusion/modules/rl/index>`
   * - Run survivability experiments
     - :doc:`Failures Module <developer/fusion/modules/failures/index>`
   * - Understand blocking metrics
     - :doc:`Core Metrics <developer/fusion/core/metrics>`
   * - Use the orchestrator (v6.0+)
     - :doc:`Core Architecture <developer/fusion/core/architecture>`
   * - Contribute to FUSION
     - `Contributing Guide <https://github.com/SDNNetSim/FUSION/blob/main/CONTRIBUTING.md>`_

----

Architecture at a Glance
========================

.. code-block:: text

   +---------------------------------------------------------------+
   |                      CLI Entry Points                          |
   |    fusion-sim (run_sim)         fusion-train (run_train)       |
   +---------------------------------------------------------------+
                                 |
                                 v
   +---------------------------------------------------------------+
   |                   Configuration System                         |
   |         INI files -> ConfigManager -> SimulationConfig         |
   +---------------------------------------------------------------+
                                 |
             +-------------------+-------------------+
             |                                       |
             v                                       v
   +---------------------+               +---------------------+
   |    LEGACY PATH      |               |   ORCHESTRATOR      |
   | (use_orchestrator   |               | (use_orchestrator   |
   |  = False)           |               |  = True)            |
   +---------------------+               +---------------------+
   |                     |               |                     |
   | SimulationEngine    |               | SimulationEngine    |
   |       |             |               |       |             |
   |       v             |               |       v             |
   | SDNController       |               | SDNOrchestrator     |
   |   - routing         |               |   - pipelines       |
   |   - spectrum        |               |   - policies        |
   |   - snr             |               |   - adapters        |
   +---------------------+               +---------------------+
             |                                       |
             +-------------------+-------------------+
                                 |
                                 v
   +---------------------------------------------------------------+
   |                     Algorithm Modules                          |
   |  routing/    spectrum/    snr/    rl/    failures/             |
   +---------------------------------------------------------------+
                                 |
                                 v
   +---------------------------------------------------------------+
   |                    Output & Analysis                           |
   |     data/output/    logs/    SimStats -> JSON/CSV              |
   +---------------------------------------------------------------+

See :doc:`developer/fusion/core/architecture` for detailed documentation.

----

Which Architecture Should I Use?
================================

FUSION v6.0 supports two simulation architectures:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * -
     - Legacy (SDNController)
     - Orchestrator (SDNOrchestrator)
   * - **Enable with**
     - ``use_orchestrator = False`` (default)
     - ``use_orchestrator = True``
   * - **Best for**
     - Reproducing existing experiments, simple simulations
     - New development, RL integration, protection features
   * - **Maturity**
     - Stable, well-tested
     - New in v6.0, under active development
   * - **Extensibility**
     - Modify source directly
     - Add pipeline stages via protocols
   * - **RL Support**
     - Limited
     - Full ControlPolicy interface

**Recommendation**

- Use **Legacy** for: Baseline experiments, algorithm comparison studies, reproducing
  published results
- Use **Orchestrator** for: RL experiments, survivability testing with protection,
  new feature development

See :doc:`developer/fusion/core/architecture` for a detailed comparison.

----

Project Status
==============

**Current Version:** 6.0.0

**Actively Developed**

- Web-based GUI (v6.1)
- Sphinx documentation (this site)
- Orchestrator pipeline architecture
- RL module (UnifiedSimEnv environment)
- Survivability features (1+1 protection)

**Beta Status**

- Failures module (partial orchestrator integration)
- ML utilities module
- Visualization plugins

**Planned**

- Replace adapters with native orchestrator implementations
- Multi-band RL support (L-band)
- Multi-process RL training

See the `GitHub Issues <https://github.com/SDNNetSim/FUSION/issues>`_ for the roadmap.

----

A Note from the Developer
=========================

FUSION has been in development for over three years. What started as an internal
research tool has evolved into an open-source project built for the broader
optical networking community.

We believe the field needs a trusted baseline simulator—one that is maintainable,
well-documented, and designed to outlast any single paper or student. Version 6
marks our commitment to making FUSION a project meant for other people, not just
for us.

:doc:`Read the full manifesto <manifesto>`

----

The Team
========

FUSION is developed by the **Advanced Communications and Networking Laboratory**
under the guidance of **Dr. Vinod Vokkarane** (Principal Investigator).

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Ryan McCann**
     - *Technical & Full-Stack Architect | AI Lead | Product Manager*

       Responsible for overall simulator architecture, internal system design, and
       integration of artificial intelligence components. Oversees HPC compatibility,
       experiment automation (e.g., SLURM), configuration management, and reproducibility.
       Leads development of intelligent control frameworks including reinforcement learning.
       Maintains modularity, extensibility, and testing standards to support sustainable
       growth of the codebase and its scientific integrity.

   * - **Arash Rezaee**
     - *Networking Systems Lead | Research & Development Lead*

       Leads implementation of core optical networking features, including traffic modeling,
       request generation, physical layer constraints, and resource allocation. Develops and
       maintains modules related to routing, spectrum allocation, SNR measurements, fixed and
       flex grid handling, grooming support, multi-band and multi-core support, and key
       performance metrics. Ensures the simulator reflects realistic network assumptions and
       constraints and guides ongoing networking model extensions.

**Contact**

- **GitHub Issues**: `Report bugs or request features <https://github.com/SDNNetSim/FUSION/issues>`_
- **GitHub Discussions**: `Ask questions or discuss ideas <https://github.com/SDNNetSim/FUSION/discussions>`_
- **Email**:

  - Ryan McCann: ryan_mccann@student.uml.edu
  - Arash Rezaee: arash_rezaee@student.uml.edu
  - Dr. Vinod Vokkarane: vinod_vokkarane@uml.edu

----

Getting Help
============

**Documentation**

- :doc:`Developer Guide <developer/index>` - Module documentation and architecture
- :doc:`API Reference <api/index>` - Auto-generated API documentation
- :doc:`Configuration Tutorials <developer/fusion/configs/tutorials>` - INI file examples

**Community**

- `GitHub Issues <https://github.com/SDNNetSim/FUSION/issues>`_ - Bug reports and feature requests
- `GitHub Discussions <https://github.com/SDNNetSim/FUSION/discussions>`_ - Questions and general discussion

**Contributing**

We welcome contributions of all kinds:

- `Contribution Guidelines <https://github.com/SDNNetSim/FUSION/blob/main/CONTRIBUTING.md>`_
- `Coding Standards <https://github.com/SDNNetSim/FUSION/blob/main/CODING_STANDARDS.md>`_
- `Development Quickstart <https://github.com/SDNNetSim/FUSION/blob/main/DEVELOPMENT_QUICKSTART.md>`_

**Citation**

If you use FUSION in your research, please cite:

.. code-block:: bibtex

   @INPROCEEDINGS{10898199,
     author={McCann, Ryan and Rezaee, Arash and Vokkarane, Vinod M.},
     booktitle={2024 IEEE International Conference on Advanced Networks and
                Telecommunications Systems (ANTS)},
     title={FUSION: A Flexible Unified Simulator for Intelligent Optical Networking},
     year={2024},
     pages={1-6},
     doi={10.1109/ANTS63515.2024.10898199}
   }

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/gui/index
   getting-started/git-github/index
   getting-started/claude-code/index
   manifesto

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
