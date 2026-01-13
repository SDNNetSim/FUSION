=====================
Flex-Grid Networks
=====================

Introduction
============

Flexible-grid (flex-grid) optical networks, also known as **elastic optical networks (EON)**, represent a paradigm shift from traditional fixed-grid wavelength-division multiplexing (WDM) systems. By allowing flexible allocation of spectrum bandwidth tailored to each connection's requirements, flex-grid networks achieve dramatically improved spectrum efficiency and scalability.

This document provides a comprehensive technical overview of flex-grid architecture, covering the ITU-T standards, key concepts, constraints, challenges, and enabling technologies. Understanding these concepts is essential for working with FUSION, which is specifically designed to model and optimize elastic optical networks.

The Problem with Fixed-Grid WDM
================================

Traditional Wavelength Division Multiplexing
---------------------------------------------

In conventional WDM networks, the optical spectrum is divided into discrete wavelength channels with fixed spacing, typically 50 GHz or 100 GHz, as defined by the ITU-T G.694.1 fixed-grid standard.

::

    Fixed Grid (50 GHz spacing):

    Frequency →
    193.0 THz  193.05 THz  193.1 THz  193.15 THz  193.2 THz
         ├─────────┼─────────┼─────────┼─────────┤
         │ Channel │ Channel │ Channel │ Channel │
         │    1    │    2    │    3    │    4    │
         └─────────┴─────────┴─────────┴─────────┘
           50 GHz    50 GHz    50 GHz    50 GHz

**Key Characteristic**: All channels have the same fixed bandwidth regardless of actual data rate requirements.

Inefficiency and Limitations
-----------------------------

The fixed-grid approach suffers from several fundamental limitations:

**1. Spectrum Wastage**
    A 10 Gbps connection and a 100 Gbps connection both occupy a 50 GHz channel, wasting spectrum for lower-rate demands.

**2. Limited Scalability**
    As demand grows beyond the fixed channel capacity (e.g., 100 Gbps), multiple channels must be bonded together, creating management complexity.

**3. Heterogeneous Traffic Accommodation**
    Modern networks handle diverse traffic from 10 Gbps (access) to 1 Tbps (data center interconnect). Fixed grids cannot efficiently serve this range.

**4. Super-Channel Limitations**
    To create high-capacity connections (400 Gbps, 1 Tbps), multiple fixed channels must be combined, but they cannot be optimally packed.

**Example Scenario**:

Consider a fiber with 50 GHz fixed grid and 8 available channels:

- Demand 1: 10 Gbps → uses 1 channel (40 GHz wasted)
- Demand 2: 40 Gbps → uses 1 channel (10 GHz wasted)
- Demand 3: 100 Gbps → uses 1 channel (0 wasted)
- Demand 4: 200 Gbps → uses 2 channels (0 wasted)

Result: 4 demands served, 50 GHz wasted, 3 channels remaining (150 GHz)

With flex-grid, the same demands could use spectrum more precisely:

- Demand 1: 12.5 GHz (1 slot)
- Demand 2: 37.5 GHz (3 slots)
- Demand 3: 75 GHz (6 slots)
- Demand 4: 150 GHz (12 slots)

Result: 4 demands served using 275 GHz, with 125 GHz remaining for more connections.

The Flex-Grid Solution
=======================

Core Concept
------------

Flex-grid networks replace the fixed-grid paradigm with a **flexible, finer-granularity spectrum allocation** model:

- Spectrum divided into narrow **frequency slots** (typically 12.5 GHz)
- Each connection allocated the **exact number of slots needed**
- Bandwidth tailored to data rate and modulation format
- Spectrum utilization dramatically improved

::

    Flex Grid (12.5 GHz slot granularity):

    Frequency →
    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │  1  │  2  │     3     │       4       │   5   │  Connections
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
      1    1    2    2    3    3    3    2    2    2   Slots per connection
      slot slot slots slots slots slots slots slots slots slots

**Advantages**:
- Connection 1: 12.5 GHz (1 slot) for 10 Gbps
- Connection 3: 25 GHz (2 slots) for 25 Gbps
- Connection 4: 37.5 GHz (3 slots) for 50 Gbps
- No wasted spectrum, perfect fit to requirements

ITU-T G.694.1 Flexible Grid Standard
=====================================

Standardization History
-----------------------

The ITU-T G.694.1 standard, originally defining fixed grids, was amended in 2012 to include flexible-grid specifications. This standardization was critical for interoperability and commercial deployment.

**Timeline**:

- **2002**: ITU-T G.694.1 published (fixed grid only)
- **2012**: G.694.1 amended to include flexible grid
- **2020**: Further revisions and clarifications

Spectral Grid Definition
-------------------------

The flexible grid is defined by three parameters:

**Nominal Central Frequency**
    f = 193.1 THz + n × 0.00625 THz

    Where:
    - 193.1 THz is the anchor frequency (1552.52 nm)
    - n is a positive or negative integer
    - 0.00625 THz = 6.25 GHz is the frequency granularity

**Slot Width**
    Slot Width = 12.5 GHz × m

    Where m is a positive integer representing the number of slot units

**Example Calculations**:

Central frequency for n = 0:
    f = 193.1 THz + 0 × 6.25 GHz = 193.1 THz (anchor)

Central frequency for n = 8:
    f = 193.1 THz + 8 × 6.25 GHz = 193.15 THz

Slot width for m = 4:
    Width = 12.5 GHz × 4 = 50 GHz (equivalent to old fixed grid)

Slot width for m = 8:
    Width = 12.5 GHz × 8 = 100 GHz

Grid Structure
--------------

The flex-grid spectrum can be visualized as:

::

    Anchor (193.1 THz)
           ↓
    ...├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤...
       -4 -3 -2 -1  0  1  2  3  4  5  6  7     ← n values (slot indices)

       ←──────→ 6.25 GHz (frequency granularity)

       ←────────────→ 12.5 GHz (slot width unit)

**Important**: The 6.25 GHz frequency granularity allows central frequencies to be positioned precisely, while the 12.5 GHz slot width provides the minimum allocatable bandwidth unit.

Allowed Slot Widths
-------------------

While the slot width formula allows any positive integer m, practical systems typically support specific values:

- **m = 2**: 25 GHz (minimum practical width)
- **m = 3**: 37.5 GHz
- **m = 4**: 50 GHz (backward compatible with fixed grid)
- **m = 6**: 75 GHz
- **m = 8**: 100 GHz (backward compatible with fixed grid)

Some vendors support finer granularity (m = 1, 12.5 GHz), but guard bands typically require at least 12.5 GHz on each side of a channel, making m = 2 the practical minimum.

The Spectrum Slot Concept
==========================

Terminology
-----------

**Frequency Slot (FS)**
    A unit of spectrum with defined central frequency and width
    Characterized by (n, m) or (index, width) in FUSION

**Contiguous Slots**
    Adjacent frequency slots with no gaps
    Required for a single connection's spectrum

**Available Slots**
    Slots that are currently unallocated and available for use

**Occupied Slots**
    Slots currently assigned to existing connections

Slot Representation in FUSION
------------------------------

FUSION represents spectrum using an integer-indexed slot model:

::

    Slot Index:  0    1    2    3    4    5    6    7
                ├────┼────┼────┼────┼────┼────┼────┤
    Status:     │ A  │ A  │ O  │ O  │ O  │ A  │ A  │ A
                └────┴────┴────┴────┴────┴────┴────┘

    A = Available, O = Occupied

**Slot Index**: Integer index starting from 0
**Slot Width**: Each slot represents a fixed frequency width (e.g., 12.5 GHz)
**Central Frequency**: Computed from index and anchor frequency

Example: With 12.5 GHz slots and anchor at 193.1 THz:
- Slot 0: 193.10 THz
- Slot 1: 193.1125 THz
- Slot 2: 193.125 THz

Spectrum Assignment Notation
-----------------------------

A spectrum assignment is represented as:

**[starting_index, ending_index]** or **[start, start + width - 1]**

Examples:
- **[2, 4]**: Uses slots 2, 3, 4 (3 slots = 37.5 GHz)
- **[0, 7]**: Uses slots 0-7 (8 slots = 100 GHz)

In FUSION code:

.. code-block:: python

    # Spectrum assignment
    start_slot = 2
    num_slots = 3
    end_slot = start_slot + num_slots - 1  # = 4

    # This connection uses slots [2, 3, 4]

Contiguity and Continuity Constraints
======================================

Two fundamental constraints govern spectrum assignment in flex-grid networks:

Spectrum Contiguity Constraint
-------------------------------

**Definition**: All frequency slots assigned to a single connection must be **contiguous** (adjacent, with no gaps).

**Reason**: Optical transmitters and receivers are designed to process a continuous band of spectrum. Non-contiguous slots would require multiple independent transceivers.

**Valid Assignment**:

::

    ├────┼────┼────┼────┼────┤
    │ A  │ X  │ X  │ X  │ A  │  Connection uses slots [1, 2, 3]
    └────┴────┴────┴────┴────┘
         └─────────────┘
            Contiguous

**Invalid Assignment**:

::

    ├────┼────┼────┼────┼────┤
    │ X  │ A  │ X  │ A  │ X  │  Connection uses slots [0, 2, 4]
    └────┴────┴────┴────┴────┘
      ↑         ↑         ↑
      └─────────┴─────────┘
         Non-contiguous (INVALID)

Spectrum Continuity Constraint
-------------------------------

**Definition**: The same spectrum slots must be used on **every link** along the connection's path.

**Reason**: Wavelength converters (which would allow spectrum shifting between links) are expensive, complex, and not widely deployed. Most networks operate without wavelength conversion.

**Valid Assignment (Path: A → B → C)**:

::

    Link A-B:  ├────┼────┼────┼────┤
               │ X  │ X  │ X  │ A  │  Uses slots [0, 1, 2]
               └────┴────┴────┴────┘

    Link B-C:  ├────┼────┼────┼────┤
               │ X  │ X  │ X  │ A  │  Uses slots [0, 1, 2]
               └────┴────┴────┴────┘

               ✓ Same slots on both links

**Invalid Assignment (Path: A → B → C)**:

::

    Link A-B:  ├────┼────┼────┼────┤
               │ X  │ X  │ X  │ A  │  Uses slots [0, 1, 2]
               └────┴────┴────┴────┘

    Link B-C:  ├────┼────┼────┼────┤
               │ A  │ X  │ X  │ X  │  Uses slots [1, 2, 3]
               └────┴────┴────┴────┘

               ✗ Different slots (INVALID without wavelength conversion)

Combined Constraint: Routing and Spectrum Assignment (RSA)
-----------------------------------------------------------

The combination of routing a connection and assigning spectrum subject to contiguity and continuity is called the **Routing and Spectrum Assignment (RSA) problem**, the flex-grid equivalent of RWA (Routing and Wavelength Assignment).

RSA is NP-hard and central to FUSION's optimization algorithms (see :doc:`resource_allocation`).

Super-Channels
==============

Concept and Motivation
-----------------------

A **super-channel** is a high-capacity optical channel formed by combining multiple **subcarriers** that are:

- Closely spaced in frequency
- Jointly switched through the network (treated as a single entity)
- Transmitted and received by the same transponder pair

**Motivation**:

As data rates exceed 100 Gbps, challenges arise:

1. **Component limitations**: Electronics and optics struggle at very high symbol rates
2. **Nonlinear effects**: Higher power on a single carrier increases nonlinear impairments
3. **Dispersion tolerance**: Wider channels are more susceptible to chromatic dispersion

**Solution**: Instead of one 400 Gbps carrier, use four 100 Gbps subcarriers closely packed. This distributes the load while maintaining spectral efficiency.

Super-Channel Architecture
---------------------------

::

    Single Carrier (400 Gbps):
    ├────────────────────────────┤
    │    400 Gbps carrier        │  Wide, high symbol rate
    └────────────────────────────┘

    Super-Channel (4 × 100 Gbps):
    ├──────┼──────┼──────┼──────┤
    │ 100G │ 100G │ 100G │ 100G │  Four subcarriers
    └──────┴──────┴──────┴──────┘
       ↑      ↑      ↑      ↑
       └──────┴──────┴──────┘
         Jointly switched

Each subcarrier:
- Occupies a narrower spectrum slice
- Uses lower symbol rate (easier to implement)
- Experiences less nonlinear distortion

Subcarrier spacing can be very tight (e.g., orthogonal frequency division multiplexing, OFDM-like), achieving high spectral efficiency.

Spectrum Allocation for Super-Channels
---------------------------------------

In FUSION, a super-channel is modeled as a single demand requiring a wider contiguous spectrum block:

.. code-block:: python

    # 400 Gbps demand using super-channel
    data_rate = 400  # Gbps
    modulation = "16-QAM"
    num_subcarriers = 4

    # Each subcarrier: 100 Gbps
    # With 16-QAM, each might need ~25 GHz (2 slots)
    # Total: 4 × 25 GHz = 100 GHz (8 slots)
    required_slots = 8

**Contiguity requirement**: All 8 slots must be adjacent.

**Continuity requirement**: Same 8 slots on every link in the path.

Benefits of Super-Channels
---------------------------

**Spectral Efficiency**
    Tight subcarrier spacing uses spectrum efficiently

**Flexibility**
    Can adjust number and width of subcarriers based on distance and quality

**Scalability**
    Easier to scale to Tbps capacities (e.g., 10 × 100 Gbps)

**Reduced Nonlinearity**
    Lower power per subcarrier reduces nonlinear effects

**Simplified Management**
    Network treats super-channel as single unit (one route, one spectrum block)

Multi-Flow Transponders
========================

Concept
-------

Traditional transponders support a **single flow**: one source-destination pair per transponder.

**Multi-flow transponders** (also called **sliceable bandwidth-variable transponders, S-BVT**) can:

- Support multiple independent flows simultaneously
- Assign different spectrum slices to different destinations
- Share transmitter/receiver hardware efficiently

::

    Traditional Transponder:
    ┌─────────────────────────┐
    │   Transponder @ Node A  │
    │  ┌───────────────────┐  │
    │  │  Single Flow      │  │─────→ Node B (100 Gbps)
    │  │  (100 Gbps to B)  │  │
    │  └───────────────────┘  │
    └─────────────────────────┘

    Multi-Flow Transponder:
    ┌─────────────────────────┐
    │   Transponder @ Node A  │
    │  ┌───────────────────┐  │
    │  │  Flow 1: 40 Gbps  │  │─────→ Node B
    │  ├───────────────────┤  │
    │  │  Flow 2: 40 Gbps  │  │─────→ Node C
    │  ├───────────────────┤  │
    │  │  Flow 3: 20 Gbps  │  │─────→ Node D
    │  └───────────────────┘  │
    └─────────────────────────┘
    Total: 100 Gbps capacity shared among 3 flows

Advantages
----------

**Resource Efficiency**
    One transponder serves multiple demands, reducing hardware costs

**Flexibility**
    Dynamically allocate capacity among flows based on current needs

**Grooming**
    Combine small demands onto shared infrastructure

**Cost Reduction**
    Fewer transponders needed overall

Implementation in FUSION
-------------------------

FUSION supports multi-flow modeling through:

.. code-block:: python

    # Define a multi-flow transponder
    transponder_capacity = 100  # Gbps total

    # Multiple demands can share the transponder
    demands = [
        {"source": "A", "destination": "B", "rate": 40},
        {"source": "A", "destination": "C", "rate": 40},
        {"source": "A", "destination": "D", "rate": 20},
    ]

    # Total rate: 100 Gbps (fits within transponder capacity)

Each flow still requires its own:
- Routing (may take different paths)
- Spectrum assignment (contiguous + continuous per flow)

But they share:
- Transponder hardware
- Cost allocation

Challenges
----------

**Routing Complexity**
    Each flow may take a different path, increasing RSA problem complexity

**Synchronization**
    Provisioning and tearing down flows must coordinate at the shared transponder

**Spectrum Fragmentation**
    Flows of different sizes can create fragmented spectrum (see next section)

Spectrum Fragmentation
=======================

Definition and Impact
---------------------

**Spectrum fragmentation** occurs when available spectrum is divided into small, non-contiguous blocks that cannot accommodate new connection requests, even though the total available spectrum is sufficient.

::

    Heavily Fragmented Spectrum:
    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ O│A │O │A │A │O │A │O │O │A │O │A │A │A │O │O │
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

    O = Occupied, A = Available

    Largest contiguous block: 4 slots
    Total available: 8 slots

    Request: 5 contiguous slots → BLOCKED (even though 8 slots available!)

**Impact**:

- **Increased blocking probability**: Requests rejected despite available spectrum
- **Reduced network utilization**: Spectrum wasted in unusable fragments
- **Lower revenue**: Fewer connections served
- **Operational complexity**: Requires defragmentation interventions

Causes of Fragmentation
------------------------

**Dynamic Traffic Patterns**
    Connections established and torn down over time, leaving gaps

**Heterogeneous Bandwidth Demands**
    Mix of small (1-2 slots) and large (8-10 slots) connections

**First-Fit Spectrum Assignment**
    Greedy algorithms that allocate the first available block create gaps

**Multi-Flow Demands**
    Different-sized flows from the same transponder create irregular patterns

**No Preemption**
    Existing connections cannot be moved to consolidate free spectrum

Example Scenario
----------------

Initial state (empty fiber):

::

    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ A│ A│ A│ A│ A│ A│ A│ A│ A│ A│ A│ A│  12 slots available
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Establish Connection 1 (4 slots, using slots 0-3):

::

    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ O│ O│ O│ O│ A│ A│ A│ A│ A│ A│ A│ A│
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Establish Connection 2 (6 slots, using slots 4-9):

::

    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ O│ O│ O│ O│ O│ O│ O│ O│ O│ O│ A│ A│
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Tear down Connection 1 (slots 0-3 freed):

::

    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ A│ A│ A│ A│ O│ O│ O│ O│ O│ O│ A│ A│
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Establish Connection 3 (2 slots, using slots 0-1):

::

    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ O│ O│ A│ A│ O│ O│ O│ O│ O│ O│ A│ A│
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

**Current state**:
- Available slots: [2, 3, 10, 11] (4 slots total in 2 fragments)
- Largest contiguous block: 2 slots

**New request**: 3 slots → **BLOCKED** (despite 4 slots available)

Fragmentation Metrics
----------------------

Several metrics quantify fragmentation:

**External Fragmentation Ratio**
    EFR = 1 - (Largest Free Block Size / Total Free Spectrum)

    Higher values indicate worse fragmentation

**Number of Free Blocks**
    More blocks = more fragmentation

**Utilization Efficiency**
    Actual utilization vs. theoretical maximum given available spectrum

**Shannon Entropy**
    Measures randomness of spectrum occupancy pattern

Defragmentation Techniques
===========================

Defragmentation aims to consolidate free spectrum into larger contiguous blocks by rearranging existing connections.

Hitless Defragmentation
------------------------

**Concept**: Re-route connections to new spectrum without service interruption

**Approach**:
1. Identify target connections to move
2. Find alternative spectrum assignments
3. Establish new lightpaths alongside existing ones
4. "Make-before-break" switch to new paths
5. Tear down old lightpaths

**Advantages**:
- No service disruption
- Maintains SLA commitments

**Challenges**:
- Requires spare spectrum during transition
- Complex signaling and coordination
- May not be possible if network is heavily loaded

Example:

::

    Before:
    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ O│ O│ A│ A│ O│ O│ A│ A│ O│ O│ A│ A│
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
      Conn1   Conn2   Conn3

    Identify: Move Conn2 to consolidate slots 2-3 and 6-7

    During transition (both paths active):
    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ O│ O│ O│ O│ O│ O│ A│ A│ O│ O│ A│ A│  Conn2 uses new path @ slots 2-3
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

    After (old Conn2 @ slots 4-5 removed):
    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    │ O│ O│ O│ O│ A│ A│ A│ A│ O│ O│ A│ A│
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
      Conn1 Conn2   [4-slot contiguous block created]

Disruptive Defragmentation
---------------------------

**Concept**: Tear down and re-establish connections in better positions

**Approach**:
1. Select connections to rearrange
2. Tear down selected connections
3. Re-provision with optimized spectrum assignments
4. Restore service

**Advantages**:
- Simpler implementation
- More effective consolidation
- No spare spectrum needed

**Challenges**:
- Service interruption (typically milliseconds to seconds)
- May violate SLA requirements
- Risk of failed re-establishment

Proactive vs. Reactive Defragmentation
---------------------------------------

**Reactive**
    Triggered when a connection request is blocked
    Try to free up spectrum to accommodate the blocked request

**Proactive**
    Periodic defragmentation based on metrics (e.g., EFR threshold)
    Prevent fragmentation before it causes blocking

**Predictive**
    Use traffic patterns and forecasting to anticipate fragmentation
    Pre-emptively defragment during low-traffic periods

Spectrum Allocation Policies to Reduce Fragmentation
-----------------------------------------------------

Fragmentation can be mitigated through intelligent spectrum assignment:

**First-Fit**
    Allocate first available block
    Fast but creates fragmentation

**Best-Fit**
    Allocate smallest sufficient block
    Minimizes wasted space

**Last-Fit**
    Allocate from highest index slots
    Consolidates free spectrum at lower indices

**Exact-Fit**
    Prefer blocks exactly matching request size
    Avoids splitting larger blocks

**Most-Used**
    Allocate near existing connections
    Consolidates free spectrum into large contiguous regions

**Least-Used**
    Spread connections across spectrum
    Balances utilization but may fragment

FUSION implements various spectrum assignment policies (see :doc:`resource_allocation`).

Bandwidth Variable Transponders (BVT)
======================================

Concept and Capabilities
-------------------------

**Bandwidth Variable Transponders (BVT)**, also called **elastic transponders**, are flexible optical transceivers that can:

- **Adjust data rate**: Operate at different speeds (e.g., 100/200/400 Gbps)
- **Change modulation format**: Switch between BPSK, QPSK, 16-QAM, etc.
- **Adapt spectrum bandwidth**: Dynamically allocate spectrum width based on requirements
- **Adjust transmit power**: Optimize signal quality for different distances

BVTs are the key enabling technology for elastic optical networks.

Adaptive Modulation and Distance
---------------------------------

Different modulation formats have different **spectral efficiency** (bits per symbol) and **reach** (maximum distance):

::

    Modulation     Spectral         OSNR          Reach
    Format         Efficiency       Required      (km)
    ─────────────────────────────────────────────────
    BPSK           1 bit/symbol     ~9 dB         ~3000
    QPSK           2 bits/symbol    ~12 dB        ~2000
    8-QAM          3 bits/symbol    ~16 dB        ~1000
    16-QAM         4 bits/symbol    ~20 dB        ~600
    32-QAM         5 bits/symbol    ~24 dB        ~400
    64-QAM         6 bits/symbol    ~28 dB        ~200

**Trade-off**: Higher spectral efficiency requires higher signal quality (OSNR), limiting reach.

**BVT Advantage**: Select modulation format based on path distance and quality:

- **Short paths** (< 600 km): Use 16-QAM or 64-QAM for high efficiency
- **Medium paths** (600-2000 km): Use QPSK or 8-QAM
- **Long paths** (> 2000 km): Use BPSK for maximum reach

Example:

::

    Demand: 100 Gbps between nodes A and B

    Scenario 1 - Distance: 500 km
        Modulation: 16-QAM (4 bits/symbol)
        Symbol Rate: 25 GBaud
        Spectrum: ~37.5 GHz (3 slots)

    Scenario 2 - Distance: 1500 km
        Modulation: QPSK (2 bits/symbol)
        Symbol Rate: 50 GBaud
        Spectrum: ~75 GHz (6 slots)

    Same demand, same BVT, different spectrum usage based on distance!

Distance-Adaptive Spectrum Allocation
--------------------------------------

FUSION models BVT behavior by considering:

1. **Path distance** (sum of link lengths)
2. **Quality of Transmission (QoT)** estimation
3. **Modulation format selection** based on QoT
4. **Spectrum requirement** derived from modulation and data rate

.. code-block:: python

    def calculate_spectrum_slots(data_rate, path_distance):
        """Calculate required spectrum slots based on distance."""
        if path_distance < 600:
            modulation = "16-QAM"
            spectral_efficiency = 4  # bits/symbol
        elif path_distance < 1500:
            modulation = "8-QAM"
            spectral_efficiency = 3
        elif path_distance < 2500:
            modulation = "QPSK"
            spectral_efficiency = 2
        else:
            modulation = "BPSK"
            spectral_efficiency = 1

        # Symbol rate (GBaud) = data_rate (Gbps) / spectral_efficiency
        symbol_rate = data_rate / spectral_efficiency

        # Spectrum (GHz) ≈ symbol_rate × 1.2 (including guard bands)
        spectrum_ghz = symbol_rate * 1.2

        # Slots (12.5 GHz each)
        num_slots = int(np.ceil(spectrum_ghz / 12.5))

        return num_slots, modulation

See :doc:`modulation_formats` for FUSION's detailed signal quality modeling.

Software Control of BVTs
-------------------------

In SDN-controlled networks, the central controller can:

- Query BVT capabilities (supported modulations, max data rate)
- Configure modulation format for each connection
- Adjust parameters dynamically based on network conditions
- Optimize network-wide spectrum usage

This programmability is essential for FUSION's optimization algorithms (see :doc:`sdn_overview`).

Technical Implementation Considerations
========================================

Hardware Requirements
---------------------

**Wavelength-Selective Switches (WSS)**
    - Must support fine-granularity switching (12.5 GHz)
    - Reconfigurable passband width
    - Critical for ROADMs in flex-grid networks

**Bandwidth-Variable Transponders (BVT)**
    - Tunable lasers with wide range
    - Flexible digital signal processors
    - Adaptive modulation and coding

**Optical Spectrum Analyzers**
    - Monitor spectrum occupancy
    - Detect channel boundaries and gaps
    - Required for control and management

Signaling and Control Extensions
---------------------------------

Traditional control protocols (GMPLS) required extensions for flex-grid:

**RSVP-TE Extensions**
    - Signaling for variable-width channels
    - Spectrum availability advertisement

**OSPF-TE / ISIS-TE Extensions**
    - Traffic engineering with spectrum state
    - Link state advertisement for slot availability

**SDN / OpenFlow Extensions**
    - Flex-grid flow rules
    - Spectrum allocation commands

FUSION models SDN control, abstracting these protocol details (see :doc:`sdn_overview`).

Interoperability Challenges
----------------------------

**Fixed-Grid Legacy Networks**
    - Coexistence of fixed and flex-grid equipment
    - Migration strategies required

**Vendor Compatibility**
    - Different implementations of ITU-T standard
    - Testing and certification essential

**Management Systems**
    - OSS/BSS tools must support flex-grid concepts
    - Visualization of spectrum utilization

Research Directions
===================

Ongoing research in flex-grid networks includes:

Machine Learning for RSA
-------------------------

- Deep reinforcement learning for routing and spectrum assignment
- Traffic prediction and proactive optimization
- Anomaly detection and fault localization

Space-Division Multiplexing Integration
----------------------------------------

- Multi-core fiber combined with flex-grid
- Joint routing in space and spectrum
- Massive capacity scaling

Disaggregated Networks
-----------------------

- Open optical line systems
- Mix-and-match vendor components
- Software-defined control of heterogeneous equipment

Network Slicing
---------------

- Virtual network instances on shared infrastructure
- Spectrum as a resource for slices
- Isolation and performance guarantees

Energy Efficiency
-----------------

- Power-aware RSA algorithms
- Sleep modes for transponders and switches
- Green networking objectives

Connection to FUSION
=====================

FUSION is specifically designed to model, simulate, and optimize flex-grid elastic optical networks. The concepts covered in this document directly map to FUSION's functionality:

**Spectrum Model**
    FUSION represents spectrum as integer-indexed slots (see :doc:`../api/core`)

**RSA Algorithms**
    FUSION implements various routing and spectrum assignment heuristics (see :doc:`resource_allocation`)

**QoT Estimation**
    FUSION models signal quality to determine feasible modulations (see :doc:`modulation_formats`)

**Fragmentation**
    FUSION tracks and can optimize spectrum fragmentation (see :doc:`resource_allocation`)

**Super-Channels**
    FUSION supports multi-slot connections modeling super-channels

**BVT Modeling**
    FUSION selects modulation formats based on distance and QoT

To use FUSION effectively:

1. Understand flex-grid principles (this document)
2. Learn FUSION's network model (see :doc:`../api/core`)
3. Explore RSA algorithms (see :doc:`resource_allocation`)
4. Configure realistic scenarios (see :doc:`../getting_started/index`)

Further Reading
===============

Standards
---------

- ITU-T G.694.1: Spectral grids for WDM applications: DWDM frequency grid
- ITU-T G.872: Architecture for optical transport networks
- ITU-T G.8080: Architecture for the automatically switched optical network

Key Papers
----------

- Jinno, M., et al. (2009). "Spectrum-efficient and scalable elastic optical path network: architecture, benefits, and enabling technologies". *IEEE Communications Magazine*, 47(11), 66-73.

- Gerstel, O., et al. (2012). "Elastic optical networking: a new dawn for the optical layer?". *IEEE Communications Magazine*, 50(2), s12-s20.

- Christodoulopoulos, K., et al. (2011). "Routing and spectrum allocation in OFDM-based optical networks with elastic bandwidth allocation". *GLOBECOM*, 1-6.

- Chatterjee, B. C., et al. (2015). "Routing and spectrum allocation in elastic optical networks: A tutorial". *IEEE Communications Surveys & Tutorials*, 17(3), 1776-1800.

Books
-----

- Tomkos, I., et al. (Eds.). (2017). *Elastic Optical Networks: Architectures, Technologies, and Control*. Springer.

- Mukherjee, B. (2006). *Optical WDM Networks*. Springer. (Covers fixed-grid foundations)

See Also
========

- :doc:`optical_networking_basics` - Foundational concepts
- :doc:`sdn_overview` - Software-defined control for flex-grid networks
- :doc:`resource_allocation` - Routing and spectrum assignment algorithms
- :doc:`modulation_formats` - Modulation formats and their characteristics
- :doc:`network_topologies` - Network topology modeling in FUSION
