========================
Optical Networking Basics
========================

Introduction
============

Optical networking represents a fundamental shift in how we transmit data across distances. Instead of using electrical signals over copper wires, optical networks use light pulses traveling through fiber-optic cables. This technology forms the backbone of modern internet infrastructure, enabling the high-speed, high-capacity communication that powers everything from streaming video to cloud computing.

This guide provides a comprehensive introduction to optical networking concepts, helping you understand the foundations upon which FUSION is built.

How Fiber-Optic Communication Works
====================================

Light as a Data Carrier
------------------------

At its core, fiber-optic communication uses light to encode and transmit data. The basic principle is remarkably elegant:

1. **Data Encoding**: Digital data (0s and 1s) is converted into light pulses
2. **Transmission**: These light pulses travel through thin glass fibers
3. **Reception**: At the destination, the light is converted back into electrical signals

The key advantage of using light is its extremely high frequency (hundreds of terahertz), which allows for enormous bandwidth capacity. A single fiber can carry multiple wavelengths of light simultaneously, each carrying independent data streams—a technique called wavelength division multiplexing (WDM).

Total Internal Reflection
-------------------------

The "magic" that keeps light confined within a fiber relies on a physical phenomenon called **total internal reflection**. A fiber-optic cable consists of:

- **Core**: The central glass strand where light travels
- **Cladding**: A surrounding layer with a lower refractive index
- **Coating**: Protective outer layer

When light traveling through the core hits the boundary with the cladding at a shallow angle, it reflects back into the core rather than escaping. This allows light to propagate through kilometers of fiber with minimal loss.

::

    ┌─────────────────────────────────────────┐
    │         Protective Coating              │
    │  ┌───────────────────────────────────┐  │
    │  │      Cladding (low n)             │  │
    │  │  ┌─────────────────────────────┐  │  │
    │  │  │   Core (high n)  ~~~>       │  │  │  Light propagates
    │  │  └─────────────────────────────┘  │  │  via total internal
    │  │                                    │  │  reflection
    │  └───────────────────────────────────┘  │
    └─────────────────────────────────────────┘

Basic Components of Optical Networks
=====================================

Optical Fibers
--------------

**Single-Mode Fiber (SMF)**
    - Core diameter: ~9 micrometers
    - Allows only one mode (path) of light propagation
    - Used for long-distance, high-bandwidth transmission
    - Lower attenuation and dispersion
    - Most common in backbone networks

**Multi-Mode Fiber (MMF)**
    - Core diameter: 50-62.5 micrometers
    - Allows multiple modes of light propagation
    - Used for shorter distances (< 2 km)
    - Higher attenuation and modal dispersion
    - Common in data centers and LANs

Transmitters
------------

Transmitters convert electrical signals into optical signals. The key component is a **light source**:

**Laser Diodes (LD)**
    - Coherent, monochromatic light
    - High power and narrow spectral width
    - Used for long-haul transmission
    - Temperature-sensitive and more expensive

**Light-Emitting Diodes (LED)**
    - Incoherent, broader spectrum
    - Lower cost and power consumption
    - Used for short-distance applications
    - More robust to temperature variations

Modern transmitters also include **modulators** that encode data onto the light wave using various modulation formats (see :doc:`modulation_formats`).

Receivers
---------

Receivers convert incoming optical signals back into electrical signals. The main component is a **photodetector**:

**PIN Photodiodes**
    - Simple, low-cost design
    - Good for moderate data rates
    - Moderate sensitivity

**Avalanche Photodiodes (APD)**
    - Internal gain mechanism
    - Higher sensitivity
    - More expensive and requires higher bias voltage

Receivers also include amplification and signal processing circuits to recover the original data.

Optical Amplifiers
------------------

As light travels through fiber, it experiences **attenuation** (signal loss). For long-distance transmission, amplifiers boost the signal without converting it to electrical form.

**Erbium-Doped Fiber Amplifiers (EDFA)**
    - Most common type
    - Amplifies signals in the C-band (1530-1565 nm)
    - Can amplify multiple wavelengths simultaneously
    - Introduces amplified spontaneous emission (ASE) noise

**Raman Amplifiers**
    - Uses stimulated Raman scattering
    - Can amplify across a wide range of wavelengths
    - Lower noise figure than EDFA
    - Requires high pump power

Optical Switches and Routers
-----------------------------

These devices direct light signals along different paths in the network:

**Optical Cross-Connects (OXC)**
    - Switch entire wavelength channels
    - Can be wavelength-selective or waveband-selective
    - Used in backbone networks for routing

**Reconfigurable Optical Add-Drop Multiplexers (ROADM)**
    - Add, drop, or pass-through wavelength channels at nodes
    - Enable dynamic network reconfiguration
    - Support flexible wavelength routing
    - Critical for modern flexible-grid networks

Multiplexers and Demultiplexers
--------------------------------

**Multiplexer (MUX)**
    - Combines multiple wavelengths onto a single fiber
    - Enables WDM transmission

**Demultiplexer (DEMUX)**
    - Separates combined wavelengths at the receiver
    - Each wavelength directed to its specific receiver

Technologies include arrayed waveguide gratings (AWG), thin-film filters, and fiber Bragg gratings (FBG).

Why Optical Networks Matter
============================

Bandwidth Capacity
------------------

Modern communication demands are growing exponentially. Video streaming, cloud computing, IoT devices, and 5G networks all require massive bandwidth. Optical networks provide:

- **Terabits per second** capacity on a single fiber
- Ability to upgrade capacity by adding wavelengths (WDM)
- Future-proof infrastructure

Distance and Geography
----------------------

Optical fibers enable:

- **Transoceanic cables** connecting continents
- **Metropolitan area networks** covering cities
- **Long-haul backbone networks** spanning countries
- Minimal signal degradation over hundreds of kilometers

Energy Efficiency
-----------------

Compared to copper-based transmission:

- Lower power consumption per bit transmitted
- Less heat generation
- Reduced cooling requirements in data centers
- Important for sustainable infrastructure

Reliability and Security
------------------------

Optical networks offer:

- **Electromagnetic immunity**: No interference from EMI/RFI
- **Difficult to tap**: Physical security advantage
- **Low error rates**: Fewer bit errors than copper
- **Longevity**: Fiber infrastructure lasts decades

Key Concepts in Optical Networking
===================================

Wavelength and Frequency
------------------------

Light is an electromagnetic wave characterized by:

**Wavelength (λ)**
    Distance between successive peaks of the wave, measured in nanometers (nm)

**Frequency (f)**
    Number of wave cycles per second, measured in Hertz (Hz)

**Relationship**: c = λ × f (where c is the speed of light)

Optical networks typically operate in specific wavelength bands:

- **O-band**: 1260-1360 nm (original band)
- **E-band**: 1360-1460 nm (extended band)
- **S-band**: 1460-1530 nm (short wavelength band)
- **C-band**: 1530-1565 nm (conventional band) - **Most common**
- **L-band**: 1565-1625 nm (long wavelength band)
- **U-band**: 1625-1675 nm (ultra-long wavelength band)

The C-band is most widely used because it aligns with EDFA amplification windows and experiences minimal fiber attenuation.

Bandwidth
---------

Bandwidth has multiple meanings in optical networks:

**Spectral Bandwidth**
    The range of frequencies or wavelengths used by a signal
    Measured in Hz (frequency) or nm (wavelength)

**Data Rate Bandwidth**
    The amount of data transmitted per unit time
    Measured in bits per second (bps)

**Relationship**: Higher spectral bandwidth generally enables higher data rates, though the exact relationship depends on the modulation format.

For a traffic demand requiring 100 Gbps, different modulation formats might need:

- BPSK: 50 GHz spectral bandwidth
- QPSK: 25 GHz spectral bandwidth
- 16-QAM: 12.5 GHz spectral bandwidth

(See :doc:`modulation_formats` for details)

Attenuation
-----------

**Attenuation** is the loss of signal power as light travels through fiber. It's measured in decibels per kilometer (dB/km).

**Sources of Attenuation**:

- **Absorption**: Light energy converted to heat in the glass
- **Scattering**: Light scattered by microscopic variations in fiber
- **Bending losses**: Light escapes when fiber is bent

Modern single-mode fiber has attenuation around **0.2 dB/km** in the C-band.

**Practical Impact**:
- After 100 km: Signal reduced by 20 dB (99% power loss)
- Requires amplification for long-distance transmission
- Limits the reach of optical signals

Dispersion
----------

**Dispersion** causes different parts of the signal to travel at different speeds, leading to pulse spreading and inter-symbol interference.

**Types of Dispersion**:

**Chromatic Dispersion (CD)**
    Different wavelengths travel at different speeds in fiber
    Measured in ps/(nm·km)
    Can be compensated using dispersion-compensating fiber or digital signal processing

**Polarization Mode Dispersion (PMD)**
    Different polarization states travel at different speeds
    Caused by fiber asymmetries
    Random and more difficult to compensate

**Modal Dispersion**
    Different modes travel at different speeds (multi-mode fiber only)
    Limits bandwidth-distance product

**Impact on Network Design**:
- Limits transmission distance at high data rates
- Requires dispersion compensation techniques
- Influences modulation format selection

Signal-to-Noise Ratio (SNR)
----------------------------

The **signal-to-noise ratio** measures signal quality—the ratio of signal power to noise power.

**Noise Sources**:

- **ASE noise**: From optical amplifiers
- **Crosstalk**: Interference from adjacent channels
- **Nonlinear effects**: Signal distortion at high powers
- **Receiver thermal noise**: Electronics in the receiver

**Importance**:
- Higher SNR enables higher data rates and longer distances
- Determines bit error rate (BER)
- Key constraint in network design

For a target BER of 10^-9, different modulation formats require different SNR levels:

- BPSK: ~12 dB
- QPSK: ~15 dB
- 16-QAM: ~21 dB

Quality of Transmission (QoT)
------------------------------

**Quality of Transmission** is a holistic measure of signal quality at the receiver, considering:

- Optical Signal-to-Noise Ratio (OSNR)
- Chromatic dispersion
- Polarization mode dispersion
- Nonlinear effects
- Filter penalties

FUSION uses QoT-aware routing algorithms to ensure that established lightpaths meet minimum quality thresholds (see :doc:`../guides/qot_estimation`).

Network Layers and Architecture
================================

The OSI Model and Optical Networks
-----------------------------------

While optical networks primarily operate at the physical layer (Layer 1), understanding the layered architecture helps contextualize their role:

::

    ┌──────────────────────────────────────┐
    │  Application Layer (Layer 7)         │  HTTP, FTP, etc.
    ├──────────────────────────────────────┤
    │  Presentation Layer (Layer 6)        │  Data formatting
    ├──────────────────────────────────────┤
    │  Session Layer (Layer 5)             │  Session management
    ├──────────────────────────────────────┤
    │  Transport Layer (Layer 4)           │  TCP, UDP
    ├──────────────────────────────────────┤
    │  Network Layer (Layer 3)             │  IP routing
    ├──────────────────────────────────────┤
    │  Data Link Layer (Layer 2)           │  Ethernet, MPLS
    ├──────────────────────────────────────┤
    │  Physical Layer (Layer 1)            │  Optical transmission
    │  ┌────────────────────────────────┐  │
    │  │ Optical Layer (lightpaths)     │  │  ← FUSION operates here
    │  └────────────────────────────────┘  │
    └──────────────────────────────────────┘

Optical Network Hierarchy
--------------------------

**Access Networks**
    - Connect end users to service providers
    - Fiber to the Home (FTTH), Passive Optical Networks (PON)
    - Distances: < 20 km

**Metropolitan Area Networks (MAN)**
    - Cover cities and urban regions
    - Connect access networks to core networks
    - Distances: 20-200 km

**Long-Haul Core Networks**
    - Backbone networks connecting cities and regions
    - High capacity, multiple wavelengths
    - Distances: 200-2000 km
    - **Primary focus of FUSION**

**Ultra-Long-Haul Networks**
    - Transoceanic and transcontinental links
    - Require advanced amplification and regeneration
    - Distances: > 2000 km

Circuit vs. Packet Switching
-----------------------------

**Circuit Switching (Optical Networks)**
    - Dedicated end-to-end path (lightpath)
    - Resources reserved for entire connection duration
    - Low latency, predictable performance
    - Efficient for high-volume, continuous traffic
    - **How traditional optical networks operate**

**Packet Switching (IP Networks)**
    - Data divided into packets
    - Packets routed independently
    - Resources shared statistically
    - Flexible, handles bursty traffic well
    - Operates at higher layers (Layer 2/3)

Modern networks use **multi-layer architectures** where packet-switched IP traffic is carried over circuit-switched optical lightpaths.

Network Topologies
------------------

Physical topology—how nodes are connected—impacts network design:

**Mesh Topology**
    - Nodes connected with multiple alternative paths
    - High reliability and fault tolerance
    - More expensive (more fiber links)
    - Common in backbone networks

**Ring Topology**
    - Nodes connected in a closed loop
    - Bidirectional for protection
    - Simpler than mesh
    - Used in metropolitan networks

**Star Topology**
    - Central hub connected to multiple nodes
    - Simple but single point of failure
    - Common in access networks

**Hybrid Topologies**
    - Combination of above patterns
    - Most real-world networks
    - Balance cost, reliability, and performance

The Wavelength Division Multiplexing (WDM) Paradigm
====================================================

Fixed-Grid WDM
--------------

Traditional WDM systems use a **fixed grid** defined by the ITU-T G.694.1 standard:

- Channels spaced at fixed intervals (e.g., 50 GHz or 100 GHz)
- Each channel carries a single wavelength
- All channels have the same bandwidth
- Simple but inflexible

Example fixed grid (50 GHz spacing):

::

    Frequency →
    ├────┼────┼────┼────┼────┼────┼────┤
    │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │  Channels
    └────┴────┴────┴────┴────┴────┴────┘
      50   50   50   50   50   50   50    GHz spacing

**Limitation**: A 10 Gbps demand and a 100 Gbps demand both use the same 50 GHz channel, wasting spectrum.

Flex-Grid (Elastic Optical Networks)
-------------------------------------

**Flexible-grid** or **elastic optical networks** address the inefficiency of fixed grids:

- Spectrum divided into narrow frequency slots (e.g., 12.5 GHz)
- Channels can occupy multiple contiguous slots
- Bandwidth allocated matches demand
- Spectrum efficiency improved significantly

Example flex-grid (12.5 GHz slots):

::

    Frequency →
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    │ 1 │ 2 │   3   │     4     │ 5 │  Channels (variable width)
    └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
      12.5 GHz slots

Channel 1: 2 slots (25 GHz) - for 10 Gbps demand
Channel 4: 4 slots (50 GHz) - for 100 Gbps demand

**Advantages**:
- Better spectrum utilization
- Supports diverse data rates
- Enables super-channels (see :doc:`flex_grid_networks`)

**FUSION models flex-grid networks**, providing sophisticated algorithms for spectrum assignment, routing, and optimization.

Network Control and Management
===============================

Control Plane Architectures
----------------------------

**Distributed Control (GMPLS)**
    - Each node runs control protocols
    - Nodes exchange signaling messages
    - Distributed decision making
    - Robust to controller failures
    - Complex to configure and update

**Centralized Control (SDN)**
    - Central controller with global network view
    - Controller computes paths and provisions resources
    - Simplified network management
    - Enables network-wide optimization
    - **FUSION models this architecture** (see :doc:`sdn_overview`)

Management Operations
---------------------

**Network Planning**
    - Topology design
    - Capacity planning
    - Equipment placement

**Provisioning**
    - Establishing lightpaths
    - Configuring transponders and switches
    - Resource allocation

**Monitoring**
    - Performance metrics (BER, OSNR, etc.)
    - Fault detection
    - Traffic statistics

**Optimization**
    - Defragmentation
    - Re-routing for better efficiency
    - Load balancing

**Restoration and Protection**
    - Detecting failures
    - Switching to backup paths
    - Minimizing service disruption

Practical Considerations
========================

Cost Factors
------------

- **Fiber installation**: Civil works, trenching, conduits
- **Equipment**: Transponders, switches, amplifiers
- **Operations**: Power, maintenance, monitoring
- **Spectrum**: Limited resource, especially in C-band

Performance Trade-offs
----------------------

**Reach vs. Capacity**
    - Higher data rates require better signal quality
    - Better signal quality means shorter reach
    - Trade-off between distance and capacity

**Cost vs. Flexibility**
    - More flexible networks require more sophisticated equipment
    - Advanced modulation formats need expensive transceivers
    - Balance capability with budget

**Efficiency vs. Simplicity**
    - Optimal algorithms can be computationally complex
    - Heuristics sacrifice optimality for speed
    - Choose based on network scale and requirements

Emerging Trends
---------------

**Space-Division Multiplexing (SDM)**
    - Multiple cores or modes in a single fiber
    - Multiply capacity per fiber
    - Multi-core fiber, few-mode fiber

**Coherent Detection**
    - Phase and polarization information recovered
    - Enables advanced modulation formats
    - Digital signal processing in receivers

**Elastic Optical Networks**
    - Dynamic bandwidth allocation
    - Spectrum and energy efficiency
    - **Core focus of FUSION research**

**Machine Learning Integration**
    - Traffic prediction and forecasting
    - Anomaly detection
    - QoT estimation and optimization

Connection to FUSION
=====================

FUSION is designed to model, simulate, and optimize **elastic optical networks** with **SDN control**. Understanding these optical networking basics is essential to using FUSION effectively:

- **Network topology**: FUSION represents fibers, nodes, and connections
- **Physical layer**: FUSION models signal propagation, QoT, and impairments
- **Spectrum allocation**: FUSION implements flex-grid spectrum assignment
- **Routing**: FUSION computes paths considering physical constraints
- **Control**: FUSION simulates centralized SDN control

To dive deeper into specific topics:

- Learn about flex-grid architecture: :doc:`flex_grid_networks`
- Understand SDN principles: :doc:`sdn_overview`
- Explore modulation formats: :doc:`modulation_formats`
- See FUSION's QoT model: :doc:`../guides/qot_estimation`

Further Reading
===============

Standards
---------

- ITU-T G.694.1: Spectral grids for WDM applications
- ITU-T G.652: Characteristics of single-mode optical fiber
- ITU-T G.709: Interfaces for optical transport networks
- IEEE 802.3: Ethernet standards (physical layer)

Books
-----

- Ramaswami, A., Sivarajan, K., & Sasaki, G. (2010). *Optical Networks: A Practical Perspective*. Morgan Kaufmann.
- Mukherjee, B. (2006). *Optical WDM Networks*. Springer.
- Winzer, P. & Neilson, D. (2017). "From Scaling Disparities to Integrated Parallelism: A Decathlon for a Decade". *Journal of Lightwave Technology*.

Research Papers
---------------

- Jinno, M., et al. (2009). "Spectrum-efficient and scalable elastic optical path network: architecture, benefits, and enabling technologies". *IEEE Communications Magazine*.
- Gerstel, O., et al. (2012). "Elastic optical networking: a new dawn for the optical layer?". *IEEE Communications Magazine*.

See Also
========

- :doc:`flex_grid_networks` - Detailed coverage of flexible-grid architecture
- :doc:`sdn_overview` - Software-defined networking for optical networks
- :doc:`modulation_formats` - Signal encoding techniques
- :doc:`rwa_problem` - Routing and wavelength assignment fundamentals
- :doc:`../guides/qot_estimation` - Quality of transmission in FUSION
