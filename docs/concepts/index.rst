==================
Concepts & Theory
==================

This section provides comprehensive educational content about optical networking,
elastic optical networks, software-defined networking, and related concepts.

.. contents:: In This Section
   :local:
   :depth: 2

Overview
========

Understanding the theoretical foundations of optical networking is essential for
effectively using FUSION and interpreting simulation results. This section covers
everything from basic fiber-optic communication to advanced AI-driven network optimization.

Who This Section Is For
=======================

**Students**
   Build a strong foundation in optical networking and learn how modern networks operate

**Researchers**
   Understand the context and challenges your algorithms aim to address

**Professionals**
   Gain insights into next-generation optical network technologies

**Developers**
   Learn the domain knowledge needed to contribute effectively to FUSION

Learning Path
=============

We recommend following this sequence:

**Foundation** (Start Here)
   1. :doc:`optical_networking_basics` - How fiber-optic communication works
   2. :doc:`wdm_vs_eon` - Evolution from WDM to Elastic Optical Networks
   3. :doc:`modulation_formats` - Signal encoding techniques

**Network Architecture**
   4. :doc:`flex_grid_networks` - Flexible spectrum allocation
   5. :doc:`sdn_overview` - Software-Defined Networking principles
   6. :doc:`network_topologies` - Common network structures

**Optimization**
   7. :doc:`resource_allocation` - Routing and Spectrum Assignment
   8. :doc:`machine_learning_optical` - AI in optical networking

Topics Covered
==============

Optical Networking Basics
--------------------------

Learn the fundamentals:

* How light carries data through fiber
* Basic components of optical networks
* Wavelength division multiplexing (WDM)
* Signal quality and impairments
* Why optical networks matter

:doc:`optical_networking_basics` →

WDM vs. Elastic Optical Networks
---------------------------------

Understand the evolution:

* Fixed-grid WDM limitations
* Elastic optical network advantages
* Spectrum efficiency improvements
* Real-world comparisons
* When to use each approach

:doc:`wdm_vs_eon` →

Flex-Grid Optical Networks
---------------------------

Deep dive into flexible spectrum:

* Flex-grid architecture
* Spectrum contiguity and continuity
* Super-channels and multi-flow
* Fragmentation challenges
* Standards (ITU-T G.694.1)

:doc:`flex_grid_networks` →

Software-Defined Networking (SDN)
----------------------------------

Modern network control:

* SDN architecture and principles
* Separation of control and data planes
* OpenFlow protocol
* SDN controllers
* Benefits for optical networks

:doc:`sdn_overview` →

Resource Allocation Algorithms
-------------------------------

The optimization challenge:

* Routing and Wavelength Assignment (RWA)
* Routing, Modulation, Core, and Spectrum Assignment (RMCSA)
* Algorithm families (greedy, metaheuristic, ML-based)
* Performance metrics
* Trade-offs and complexity

:doc:`resource_allocation` →

Machine Learning in Optical Networks
-------------------------------------

AI-driven optimization:

* Why ML for optical networks?
* Supervised learning for RSA
* Reinforcement learning for dynamic networks
* Deep learning architectures (DNNs, CNNs, GNNs)
* Real-world applications and challenges

:doc:`machine_learning_optical` →

Network Topologies
------------------

Common network structures:

* Real-world topologies (NSFNet, COST239, etc.)
* Topology characteristics (diameter, connectivity)
* Impact on algorithm performance
* Creating custom topologies

:doc:`network_topologies` →

Modulation Formats
------------------

Signal encoding techniques:

* Modulation basics (BPSK, QPSK, QAM)
* Reach vs. data rate trade-offs
* Distance-adaptive modulation
* SNR requirements
* Choosing the right modulation

:doc:`modulation_formats` →

Key Concepts Quick Reference
=============================

Essential Terms
---------------

**Elastic Optical Network (EON)**
   Network with flexible, fine-granularity spectrum allocation

**Flex-Grid**
   Flexible frequency grid allowing variable-sized channels

**Software-Defined Networking (SDN)**
   Architecture separating network control from forwarding

**Routing and Spectrum Assignment (RSA)**
   Problem of finding paths and allocating spectrum for requests

**Super-Channel**
   Multiple adjacent subcarriers treated as single high-capacity channel

**Spectrum Fragmentation**
   Scattered free spectrum slots that cannot be used efficiently

**Modulation Format**
   Method of encoding data onto optical carrier (e.g., QPSK, 16QAM)

**Blocking Probability**
   Fraction of requests that cannot be accommodated due to resource constraints

**Guard Band**
   Empty spectrum between channels to prevent interference

**Distance-Adaptive Modulation**
   Selecting modulation format based on path length and signal quality

Important Constraints
---------------------

**Spectrum Continuity**
   Same spectrum slots must be available on all links of a path

**Spectrum Contiguity**
   Allocated spectrum slots must be adjacent (no gaps)

**Modulation Reach**
   Each modulation format has maximum transmission distance

**Non-Overlapping Spectrum**
   Different connections cannot use same spectrum on same fiber

**Physical Layer**
   Signal quality must meet SNR thresholds for chosen modulation

Performance Metrics
-------------------

**Blocking Probability (BP)**
   Percentage of requests blocked due to insufficient resources

**Bandwidth Blocking Ratio (BBR)**
   Percentage of requested bandwidth that is blocked

**Spectrum Utilization**
   Average percentage of spectrum in use

**Spectrum Efficiency**
   Bits/s/Hz achieved by the network

**Average Hops**
   Mean number of links traversed by connections

**Fragmentation Ratio**
   Measure of spectrum fragmentation

Fundamental Challenges
======================

The Optimization Problem
------------------------

Optical network optimization is inherently complex:

* **NP-Hard**: Finding optimal solutions is computationally intractable
* **Multi-Objective**: Must balance blocking, utilization, fairness, etc.
* **Dynamic**: Networks change over time as requests arrive/depart
* **Constrained**: Physical and spectrum constraints limit options
* **Large-Scale**: Real networks have hundreds of nodes and links

Why Simulation Matters
----------------------

Simulations like FUSION are essential because:

* **Testing**: Evaluate algorithms before deploying on real networks
* **Comparison**: Fairly compare different approaches
* **What-If**: Explore scenarios too expensive or risky for real networks
* **Research**: Develop and validate new algorithms
* **Education**: Learn how networks behave under different conditions

Real-World Context
==================

Industry Trends
---------------

* **Traffic Growth**: 30-40% annual increase in data traffic
* **5G and Beyond**: Fronthaul/backhaul requirements
* **Cloud Services**: East-west data center traffic
* **Video Streaming**: Majority of internet traffic
* **IoT**: Billions of connected devices

Technology Evolution
--------------------

* **400G/800G**: Higher per-channel data rates
* **Space Division Multiplexing (SDM)**: Multi-core and multi-mode fibers
* **Coherent Detection**: Advanced modulation formats
* **Photonic Integration**: Smaller, cheaper optical components
* **AI/ML**: Intelligent network operation

Why This Matters
-----------------

Understanding these concepts helps you:

* **Design Better Algorithms**: Know the constraints and objectives
* **Interpret Results**: Understand what metrics mean and why they matter
* **Make Trade-offs**: Balance conflicting objectives intelligently
* **Innovate**: Identify opportunities for improvement
* **Communicate**: Discuss your work with domain experts

Further Reading
===============

**Academic Papers**

* See :doc:`../reference/bibliography` for key papers in optical networking

**Standards**

* ITU-T G.694.1: Spectral grids for WDM applications
* ITU-T G.709: Optical Transport Network (OTN)
* IEEE 802.3: Ethernet standards

**Books**

* "Elastic Optical Networks" - Georgios Zervas
* "Optical Networks: A Practical Perspective" - Ramaswami et al.
* "Optical Fiber Communications" - Gerd Keiser

**Online Resources**

* :doc:`../reference/helpful_links` - Curated list of resources

Start Learning
==============

Ready to dive in? Begin with :doc:`optical_networking_basics` to build your
foundation in optical networking!

.. toctree::
   :maxdepth: 2
   :hidden:

   optical_networking_basics
   wdm_vs_eon
   flex_grid_networks
   sdn_overview
   resource_allocation
   machine_learning_optical
   network_topologies
   modulation_formats
