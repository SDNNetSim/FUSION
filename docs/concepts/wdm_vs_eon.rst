=====================================================
WDM vs. Elastic Optical Networks
=====================================================

Understanding the evolution from Wavelength Division Multiplexing (WDM) to Elastic
Optical Networks (EON) is crucial for appreciating modern optical networking.

.. contents:: Table of Contents
   :local:
   :depth: 3

The Evolution of Optical Networks
==================================

From Copper to Fiber
--------------------

In past decades, data was transmitted over copper wires. While functional, copper cables
had several significant drawbacks:

* **Security**: Could be tapped into, compromising data
* **Interference**: Prone to electromagnetic interference
* **Distance**: Signal degradation over long distances
* **Bandwidth**: Limited capacity for data transmission

The introduction of **fiber-optic cables** revolutionized telecommunications by addressing
all these issues:

* ✅ **Near light-speed transmission**: Data travels at ~200,000 km/s in fiber
* ✅ **Immune to EMI**: No electromagnetic interference
* ✅ **Secure**: Difficult to tap without detection
* ✅ **High bandwidth**: Enormous capacity for data
* ✅ **Long distance**: Can traverse thousands of kilometers with amplification

The WDM Revolution
------------------

Fiber-optic cables can do something remarkable: transmit multiple independent data streams
simultaneously using different wavelengths (colors) of light. This technology is called
**Wavelength Division Multiplexing (WDM)**.

Think of it like a prism splitting white light into a rainbow - each color can carry
different data!

Wavelength Division Multiplexing (WDM)
=======================================

How WDM Works
-------------

WDM divides the optical spectrum into fixed-width channels, each carrying a separate
data stream.

.. code-block:: text

   Wavelength Spectrum:
   |-------|-------|-------|-------|-------|-------|-------|
   | Ch 1  | Ch 2  | Ch 3  | Ch 4  | Ch 5  | Ch 6  | Ch 7  |
   | 50GHz | 50GHz | 50GHz | 50GHz | 50GHz | 50GHz | 50GHz |
   |-------|-------|-------|-------|-------|-------|-------|

**Key Characteristics:**

* **Fixed Grid**: Channels have predetermined, uniform spacing (e.g., 50 GHz)
* **Rigid Allocation**: Each connection gets one or more full channels
* **ITU-T Standard**: Follows ITU-T G.694.1 fixed frequency grid

WDM Channel Spacing
-------------------

Common WDM channel spacings:

* **DWDM (Dense WDM)**: 50 GHz or 100 GHz spacing
* **CWDM (Coarse WDM)**: 20 nm spacing (wider channels)

Example: 50 GHz spacing
   * 80 channels in C-band (1530-1565 nm)
   * Each channel: ~50 GHz wide
   * Total capacity: 80 × 100 Gbps = 8 Tbps

The WDM Limitation
==================

Spectrum Inefficiency
---------------------

Consider a request for 25 Gbps of bandwidth in a 50 GHz WDM system:

.. code-block:: text

   50 GHz Channel (can carry ~100 Gbps):
   ╔═══════════════════════════════════════════════╗
   ║ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ║
   ║  25 Gbps used      |    25 Gbps WASTED      ║
   ╚═══════════════════════════════════════════════╝

**Problem**: If a connection only needs 25 Gbps, the remaining 25 Gbps in the
50 GHz channel is reserved but unused - wasted!

* ❌ 50% spectrum efficiency in this example
* ❌ Cannot subdivide channels for smaller requests
* ❌ Cannot combine channels for requests > 100 Gbps (without multiple transponders)
* ❌ Poor adaptation to varying traffic demands

The Capacity Crunch
-------------------

As internet traffic grows exponentially:

* **Video Streaming**: 4K/8K video requires massive bandwidth
* **Cloud Computing**: Data center interconnection
* **5G Networks**: Ultra-high-speed mobile access
* **IoT**: Billions of connected devices

WDM's fixed grid becomes increasingly inadequate because:

1. **Underutilization**: Small requests waste spectrum
2. **Limited Flexibility**: Cannot adapt to diverse traffic
3. **Spectral Limits**: Approaching Shannon capacity limits
4. **Inefficient Upgrades**: Must add entire channels, not just what's needed

.. important::
   With traffic growing 30-40% annually, we're rapidly approaching the limits
   of available optical spectrum in WDM systems!

Elastic Optical Networks (EON)
===============================

The EON Solution
----------------

Elastic Optical Networks solve WDM's inefficiency by using a **flexible grid**
that can adapt to the exact bandwidth requirements of each connection.

.. code-block:: text

   Flex-Grid Spectrum (12.5 GHz slots):
   |---|---|---|---|---|---|---|---|---|---|---|---|
   |░░░|░░░|XXX|XXX|   |   |XXX|XXX|XXX|   |   |░░░|
   |Request 1  |  Free |  Request 2  | Free  |Req 3|

Instead of fixed 50 GHz channels, EON uses fine-granularity slots (e.g., 12.5 GHz),
and allocates exactly what each connection needs!

Key Innovations
---------------

**1. Flexible Spectrum Allocation**

* Allocate any number of contiguous slots
* Request for 25 Gbps? Use 2 slots (25 GHz)
* Request for 150 Gbps? Use 12 slots (150 GHz)

**2. Bandwidth Variable Transponders (BVT)**

* Can operate at different data rates
* Adapt to connection requirements
* More cost-effective than fixed-rate transponders

**3. Distance-Adaptive Modulation**

* Short distances: Use high-order modulation (e.g., 16QAM) for high capacity
* Long distances: Use robust modulation (e.g., QPSK) for reliability
* Optimize spectrum efficiency based on path length

**4. Super-Channels**

* Group multiple optical carriers with narrow spacing
* Create ultra-high-capacity connections (400G, 1T+)
* All carriers of a super-channel traverse same path

WDM vs. EON Comparison
======================

Visual Comparison
-----------------

Consider the same 5 connection requests on WDM vs. EON:

**WDM (Fixed 50 GHz channels):**

.. code-block:: text

   |-------|-------|-------|-------|-------|-------|-------|-------|
   |███ 40%|███ 60%|███100%|███ 30%|███ 80%|       |       |       |
   |-------|-------|-------|-------|-------|-------|-------|-------|
     Used     Used     Used     Used     Used    Free     Free     Free

   Utilization: 62%
   Wasted in used channels: 38%
   Blocked: 1 request (couldn't fit)

**EON (Flexible 12.5 GHz slots):**

.. code-block:: text

   |XXX|XXX|XXX|XXX|XXX|XXX|XXX|XXX|XXX|XXX|XXX|   |   |   |   |   |
   |Request 1  |Req2|Request 3  |Req4|Req5    | Free spectrum        |

   Utilization: 68%
   Wasted: 0%
   Blocked: 0 requests

.. note::
   **Credit**: Adapted from P. Afsharlar, "Resource Allocation Algorithms for
   Elastic Optical Networks," PhD. dissertation, Dept. Elect. and Comp. Eng.,
   Univ. of Mass. Lowell, Lowell, MA, United States, 2021.

Efficiency Comparison
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Metric
     - WDM
     - EON
   * - **Spectrum Granularity**
     - 50/100 GHz
     - 6.25/12.5 GHz
   * - **Flexibility**
     - Fixed channels
     - Elastic allocation
   * - **Spectrum Efficiency**
     - 40-60%
     - 70-90%
   * - **Traffic Adaptability**
     - Poor
     - Excellent
   * - **Blocking Probability**
     - Higher
     - Lower
   * - **Equipment Cost**
     - Multiple transponders
     - BVT (more flexible)
   * - **Network Complexity**
     - Lower (simpler)
     - Higher (more control)

Performance Benefits
--------------------

EON provides significant improvements:

* **30-50% Better Spectrum Efficiency**: More connections with same resources
* **Lower Blocking**: Can accommodate more diverse traffic
* **Better QoS**: Right-sized bandwidth allocation
* **Future-Proof**: Easier to upgrade to higher data rates
* **Cost-Effective**: Fewer stranded resources

When WDM Is Still Used
-----------------------

Despite EON advantages, WDM remains relevant:

* **Legacy Networks**: Existing WDM infrastructure
* **Simpler Requirements**: When flexibility isn't critical
* **Cost**: WDM equipment can be less expensive
* **Mature Technology**: Well-understood, stable
* **Specific Use Cases**: Point-to-point long-haul links

The Flex-Grid Standard
=======================

ITU-T G.694.1
-------------

The ITU-T standardized the flexible grid for EON:

* **Slot Width**: 12.5 GHz nominal (can be 6.25 GHz)
* **Central Frequency**: 193.1 THz + n × 6.25 GHz
* **Minimum Spacing**: One slot (12.5 GHz)
* **Backward Compatible**: Can emulate fixed WDM grid

Spectrum Slot Notation
-----------------------

A connection is specified by its slot allocation:

* **First Slot**: Index of first occupied slot
* **Last Slot**: Index of last occupied slot
* **Number of Slots**: (Last - First + 1)

Example:
   * First Slot: 10
   * Last Slot: 13
   * Occupies: 4 slots (50 GHz)
   * Capacity: ~100-200 Gbps depending on modulation

Technical Challenges in EON
============================

While EON offers significant advantages, it introduces new challenges:

Routing and Spectrum Assignment (RSA)
--------------------------------------

More complex than WDM's Routing and Wavelength Assignment (RWA):

* **Spectrum Continuity**: Same slots must be available on entire path
* **Spectrum Contiguity**: Allocated slots must be adjacent (no gaps)
* **Variable Sizing**: Requests have different slot requirements
* **Fragmentation**: Free slots become scattered, reducing efficiency

Control Plane Complexity
-------------------------

* **Dynamic Reconfiguration**: Real-time spectrum management
* **Multi-Domain**: Coordination across network operators
* **Signaling Overhead**: More information to exchange
* **Computation**: More complex algorithms needed

Fragmentation Management
------------------------

As connections come and go, spectrum becomes fragmented:

.. code-block:: text

   Fragmented Spectrum:
   |XXX|   |XXX|   |   |XXX|XXX|   |   |XXX|   |   |
    Used  F  Used  Free   Used   Free   Used  Free

   Cannot allocate 3 contiguous slots, even though 6 slots are free!

Solutions:
   * **Defragmentation**: Periodically rearrange connections
   * **Smart Assignment**: Algorithms that minimize fragmentation
   * **Fragmentation-Aware Routing**: Consider current spectrum state

Physical Layer Impairments
---------------------------

* **Flexible Modulation**: More complexity in transponders
* **Nonlinear Effects**: Vary with channel spacing
* **Crosstalk**: Adjacent channels can interfere
* **Filter Requirements**: More precise optical filters needed

EON in FUSION
==============

FUSION simulates EON with:

* **Flexible Spectrum Grid**: Configurable slot width (typically 12.5 GHz)
* **Variable Slot Allocation**: Connections use required number of slots
* **Distance-Adaptive Modulation**: Automatic modulation selection
* **RSA Algorithms**: Multiple routing and spectrum assignment strategies
* **Fragmentation Modeling**: Realistic spectrum state evolution
* **Performance Metrics**: Blocking probability, spectrum utilization, etc.

Configuration Example
---------------------

.. code-block:: ini

   [network_settings]
   # Enable EON with flex-grid
   num_spectrum_slots = 320     # Total slots available
   slot_width = 12.5            # GHz per slot
   guard_band = 1               # Slots between connections

   [spectrum_settings]
   algorithm = first_fit
   modulation_selection = distance_adaptive

   [request_settings]
   min_bandwidth = 25           # Gbps
   max_bandwidth = 400          # Gbps

Real-World Deployments
======================

EON Technology Adoption
-----------------------

Major telecommunications providers have deployed EON:

* **AT&T**: Long-haul backbone network
* **Verizon**: Metro and regional networks
* **NTT**: Japanese nationwide network
* **China Telecom**: Large-scale deployment

Reported Benefits:

* 40-50% reduction in required spectrum
* 25-30% lower blocking probability
* Better support for diverse services (5G, cloud, video)
* More economical network expansion

Research and Development
------------------------

Active research areas:

* **Machine Learning**: AI-driven RSA optimization
* **Space-Division Multiplexing**: Combining EON with multi-core fibers
* **Quantum Communication**: EON for quantum key distribution
* **Optical Edge Computing**: EON for distributed computing
* **6G Networks**: Next-generation access and backhaul

Summary and Key Takeaways
==========================

.. important::
   **Why EON Matters**

   * **Spectrum Crisis**: Internet traffic growth requires better efficiency
   * **Flexibility**: EON adapts to diverse, changing traffic demands
   * **Cost**: Better utilization means fewer resources needed
   * **Future**: Foundation for next-generation optical networks

**Key Differences:**

.. list-table::
   :widths: 50 50

   * - **WDM**
     - **EON**
   * - Fixed channels (50/100 GHz)
     - Flexible slots (12.5 GHz)
   * - One size fits all
     - Right-sized allocation
   * - Simple, rigid
     - Complex, adaptive
   * - Lower spectrum efficiency
     - Higher spectrum efficiency
   * - Easier control
     - More sophisticated control
   * - Legacy technology
     - Modern, future-proof

**Bottom Line:**

EON is the future of optical networking, offering the flexibility and efficiency
needed to meet growing bandwidth demands. Understanding WDM vs. EON is essential
for appreciating why modern optical network simulators like FUSION focus on
elastic optical networks.

Next Steps
==========

Continue your learning:

* :doc:`flex_grid_networks` - Deep dive into flex-grid architecture
* :doc:`resource_allocation` - Learn about RSA algorithms
* :doc:`modulation_formats` - Understand distance-adaptive modulation
* :doc:`sdn_overview` - See how SDN enables EON

Additional Resources
====================

**Seminal Papers:**

* Jinno et al., "Spectrum-efficient and scalable elastic optical path network," IEEE Communications Magazine, 2009
* Gerstel et al., "Elastic optical networking: a new dawn for the optical layer?" IEEE Communications Magazine, 2012

**Review Articles:**

* Chatterjee et al., "Routing and spectrum allocation in elastic optical networks: A tutorial," IEEE Communications Surveys & Tutorials, 2015
* Tomkos et al., "A survey on elastic optical networking," IEEE Communications Surveys & Tutorials, 2014

**Further Reading:**

* :doc:`../reference/bibliography` - Complete reference list
* :doc:`../reference/helpful_links` - Online resources

.. seealso::

   **Related Concepts:**

   * :doc:`optical_networking_basics` - Fundamentals
   * :doc:`flex_grid_networks` - Technical details
   * :doc:`network_topologies` - Network structures
