==================
Modulation Formats
==================

Understanding optical modulation formats is essential for efficient spectrum usage
and reliable communication in elastic optical networks.

.. contents:: Table of Contents
   :local:
   :depth: 3

What is Modulation?
===================

The Basics
----------

**Modulation** is the process of encoding digital data onto an optical carrier signal (light).
Think of it like morse code, but instead of dots and dashes, we vary properties of light
to represent 1s and 0s.

In optical communications, we can modulate:

* **Amplitude**: Brightness of light
* **Phase**: Timing/position of light waves
* **Frequency**: Color of light
* **Polarization**: Orientation of light waves

Modern optical systems primarily use **phase modulation** and **amplitude modulation** combined.

Why Modulation Matters
----------------------

Different modulation formats provide different trade-offs:

**Spectral Efficiency** (bits/s/Hz)
   How much data fits in given spectrum

**Reach** (km)
   Maximum transmission distance

**Complexity** (cost)
   Equipment sophistication required

**Robustness** (reliability)
   Tolerance to noise and impairments

**The Fundamental Trade-off:**

.. code-block:: text

   High Data Rate ←→ Long Distance

   You can't have both!

   High-order modulation: More data, shorter reach
   Low-order modulation: Less data, longer reach

Constellation Diagrams
======================

Understanding Constellations
-----------------------------

A **constellation diagram** shows how data symbols are represented:

* Each point = one symbol
* Symbol = group of bits
* More points = more bits per symbol = higher data rate

**BPSK (Binary Phase Shift Keying):**

.. code-block:: text

   1 bit per symbol

      0°        180°
       ●---------●
       0          1

   2 points = 2¹ = 1 bit/symbol

**QPSK (Quadrature Phase Shift Keying):**

.. code-block:: text

   2 bits per symbol

          01
           ●
      11 ●   ● 00
           ●
          10

   4 points = 2² = 2 bits/symbol

**16QAM (16 Quadrature Amplitude Modulation):**

.. code-block:: text

   4 bits per symbol

     ● ● ● ●
     ● ● ● ●
     ● ● ● ●
     ● ● ● ●

   16 points = 2⁴ = 4 bits/symbol

**Key Insight:**

More points = More bits per symbol = Higher data rate

BUT: Points closer together = Harder to distinguish = More errors

Common Modulation Formats
==========================

BPSK - Binary Phase Shift Keying
---------------------------------

**Characteristics:**

* 1 bit per symbol
* 2 constellation points
* Simplest format
* Most robust
* Longest reach

**Specifications:**

.. list-table::
   :widths: 30 70

   * - **Bits/Symbol**
     - 1
   * - **Reach**
     - 6000+ km
   * - **SNR Required**
     - 6.8 dB
   * - **Spectral Efficiency**
     - ~1 bit/s/Hz
   * - **Use Cases**
     - Ultra-long-haul, submarine cables

**When to Use:**

* ✓ Very long distances (> 4000 km)
* ✓ Degraded signal conditions
* ✓ When reliability is critical

**FUSION Configuration:**

.. code-block:: ini

   [spectrum_settings]
   modulation_selection = fixed
   modulation_format = BPSK

QPSK - Quadrature Phase Shift Keying
-------------------------------------

**Characteristics:**

* 2 bits per symbol
* 4 constellation points
* Good balance of reach and efficiency
* Most widely used
* Industry standard

**Specifications:**

.. list-table::
   :widths: 30 70

   * - **Bits/Symbol**
     - 2
   * - **Reach**
     - ~4000 km
   * - **SNR Required**
     - 9.8 dB
   * - **Spectral Efficiency**
     - ~2 bit/s/Hz
   * - **Use Cases**
     - Long-haul, regional networks

**When to Use:**

* ✓ Long distances (1000-4000 km)
* ✓ Standard backbone connections
* ✓ General purpose

**Data Rate Example:**

.. code-block:: text

   Symbol Rate: 32 GBaud
   Bits/Symbol: 2
   Data Rate: 32 × 2 × 2 (polarizations) = 128 Gbps

8QAM - 8-ary Quadrature Amplitude Modulation
---------------------------------------------

**Characteristics:**

* 3 bits per symbol
* 8 constellation points
* Medium reach
* Good spectral efficiency

**Specifications:**

.. list-table::
   :widths: 30 70

   * - **Bits/Symbol**
     - 3
   * - **Reach**
     - ~2000 km
   * - **SNR Required**
     - 12.6 dB
   * - **Spectral Efficiency**
     - ~3 bit/s/Hz
   * - **Use Cases**
     - Metro, regional networks

**When to Use:**

* ✓ Medium distances (500-2000 km)
* ✓ When 16QAM reach insufficient
* ✓ Balanced performance

16QAM - 16-ary Quadrature Amplitude Modulation
-----------------------------------------------

**Characteristics:**

* 4 bits per symbol
* 16 constellation points
* Popular for metro networks
* High spectral efficiency

**Specifications:**

.. list-table::
   :widths: 30 70

   * - **Bits/Symbol**
     - 4
   * - **Reach**
     - ~1000 km
   * - **SNR Required**
     - 14.8 dB
   * - **Spectral Efficiency**
     - ~4 bit/s/Hz
   * - **Use Cases**
     - Metro, data center interconnect

**When to Use:**

* ✓ Medium distances (250-1000 km)
* ✓ Data center interconnection
* ✓ High-capacity metro

**Data Rate Example:**

.. code-block:: text

   Symbol Rate: 32 GBaud
   Bits/Symbol: 4
   Data Rate: 32 × 4 × 2 = 256 Gbps

32QAM - 32-ary Quadrature Amplitude Modulation
-----------------------------------------------

**Characteristics:**

* 5 bits per symbol
* 32 constellation points
* Short-medium reach
* Very high spectral efficiency

**Specifications:**

.. list-table::
   :widths: 30 70

   * - **Bits/Symbol**
     - 5
   * - **Reach**
     - ~500 km
   * - **SNR Required**
     - 17.1 dB
   * - **Spectral Efficiency**
     - ~5 bit/s/Hz
   * - **Use Cases**
     - Metro, short-haul

**When to Use:**

* ✓ Short-medium distances (125-500 km)
* ✓ Maximum spectral efficiency needed
* ✓ Well-maintained fiber

64QAM - 64-ary Quadrature Amplitude Modulation
-----------------------------------------------

**Characteristics:**

* 6 bits per symbol
* 64 constellation points
* Short reach only
* Maximum spectral efficiency
* Most demanding

**Specifications:**

.. list-table::
   :widths: 30 70

   * - **Bits/Symbol**
     - 6
   * - **Reach**
     - ~250 km
   * - **SNR Required**
     - 19.9 dB
   * - **Spectral Efficiency**
     - ~6 bit/s/Hz
   * - **Use Cases**
     - Metro, intra-city

**When to Use:**

* ✓ Short distances (< 250 km)
* ✓ High-quality fiber
* ✓ Maximum capacity needed

**Data Rate Example:**

.. code-block:: text

   Symbol Rate: 32 GBaud
   Bits/Symbol: 6
   Data Rate: 32 × 6 × 2 = 384 Gbps

Complete Comparison
===================

Format Summary Table
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 20 20

   * - Format
     - Bits/Sym
     - Reach (km)
     - SNR (dB)
     - Data Rate*
     - Use Case
   * - **BPSK**
     - 1
     - 6000+
     - 6.8
     - 64 Gbps
     - Ultra-long-haul
   * - **QPSK**
     - 2
     - 4000
     - 9.8
     - 128 Gbps
     - Long-haul
   * - **8QAM**
     - 3
     - 2000
     - 12.6
     - 192 Gbps
     - Regional
   * - **16QAM**
     - 4
     - 1000
     - 14.8
     - 256 Gbps
     - Metro
   * - **32QAM**
     - 5
     - 500
     - 17.1
     - 320 Gbps
     - Metro/Short-haul
   * - **64QAM**
     - 6
     - 250
     - 19.9
     - 384 Gbps
     - Intra-city

\*Assuming 32 GBaud symbol rate, dual polarization

Visual Trade-off
----------------

.. code-block:: text

   Modulation vs. Reach:

   Data Rate (Gbps)
      ↑
   384 |                    ●64QAM
   320 |              ●32QAM
   256 |        ●16QAM
   192 |   ●8QAM
   128 | ●QPSK
    64 |●BPSK
      |________________________→ Reach (km)
       0   1000  2000  3000 4000+

   Trade-off: Higher data rate = Shorter reach

Distance-Adaptive Modulation
=============================

The Concept
-----------

Instead of using one modulation for all connections, **choose the best format
for each connection based on its distance**.

**Goal:** Maximize spectral efficiency while ensuring successful transmission.

How It Works
------------

**Step 1:** Calculate path length

.. code-block:: python

   def calculate_path_length(path, network):
       total_length = 0
       for i in range(len(path) - 1):
           link = (path[i], path[i+1])
           total_length += network[link]['length_km']
       return total_length

**Step 2:** Select modulation based on length

.. code-block:: python

   def select_modulation(path_length_km):
       if path_length_km <= 250:
           return '64QAM', 6
       elif path_length_km <= 500:
           return '32QAM', 5
       elif path_length_km <= 1000:
           return '16QAM', 4
       elif path_length_km <= 2000:
           return '8QAM', 3
       elif path_length_km <= 4000:
           return 'QPSK', 2
       else:
           return 'BPSK', 1

**Step 3:** Calculate spectrum slots needed

.. code-block:: python

   def calculate_slots_needed(bandwidth_gbps, bits_per_symbol, slot_width_ghz=12.5):
       # Account for modulation efficiency
       symbol_rate = bandwidth_gbps / (bits_per_symbol * 2)  # 2 polarizations

       # Convert to spectrum width
       bandwidth_ghz = symbol_rate * 1.2  # 20% overhead

       # Round up to slots
       slots = math.ceil(bandwidth_ghz / slot_width_ghz)

       return slots

Example
-------

**Scenario:** 400 Gbps request, path = 750 km

.. code-block:: python

   # Step 1: Path length
   path_length = 750  # km

   # Step 2: Select modulation
   modulation, bits_per_symbol = select_modulation(750)
   # Result: modulation = '16QAM', bits_per_symbol = 4

   # Step 3: Calculate slots
   slots_needed = calculate_slots_needed(400, 4)
   # Result: slots_needed = 12

**Alternative path:** 400 Gbps request, path = 1500 km

.. code-block:: python

   modulation, bits_per_symbol = select_modulation(1500)
   # Result: modulation = '8QAM', bits_per_symbol = 3

   slots_needed = calculate_slots_needed(400, 3)
   # Result: slots_needed = 16

**Key Point:** Longer path requires more slots for same data rate!

Benefits
--------

* ✓ **Optimal Efficiency**: Use spectrum efficiently where possible
* ✓ **Automatic**: No manual configuration needed
* ✓ **Fairness**: Long paths don't unfairly consume resources
* ✓ **Realistic**: Models real-world transponders

FUSION Configuration
--------------------

.. code-block:: ini

   [spectrum_settings]
   modulation_selection = distance_adaptive

   # Optional: Customize distance thresholds
   [modulation_thresholds]
   64qam_reach_km = 250
   32qam_reach_km = 500
   16qam_reach_km = 1000
   8qam_reach_km = 2000
   qpsk_reach_km = 4000

SNR-Based Modulation Selection
===============================

More Accurate Approach
----------------------

Instead of using just distance, calculate actual signal quality (SNR).

**Advantages:**

* ✓ Accounts for fiber quality
* ✓ Considers accumulated impairments
* ✓ More accurate than distance-based
* ✓ Handles non-uniform fiber

**Process:**

1. Calculate optical SNR for path
2. Compare to modulation thresholds
3. Select highest modulation with sufficient SNR

SNR Calculation
---------------

Simplified model:

.. code-block:: python

   def calculate_snr(path, network):
       # Initialize signal power
       power_dbm = 0  # Transmit power

       # Traverse path
       for link in path:
           # Fiber attenuation
           length = network[link]['length_km']
           attenuation = 0.2 * length  # dB
           power_dbm -= attenuation

           # Amplifier (every 80 km)
           if link_has_amplifier(link):
               power_dbm += 20  # Amplifier gain
               # Add ASE noise
               noise_figure = 5  # dB
               # Noise accumulates...

       # Calculate SNR
       snr_db = power_dbm - total_noise_dbm
       return snr_db

**Modulation Selection with SNR:**

.. code-block:: python

   def snr_based_modulation(snr_db):
       if snr_db >= 19.9:
           return '64QAM'
       elif snr_db >= 17.1:
           return '32QAM'
       elif snr_db >= 14.8:
           return '16QAM'
       elif snr_db >= 12.6:
           return '8QAM'
       elif snr_db >= 9.8:
           return 'QPSK'
       elif snr_db >= 6.8:
           return 'BPSK'
       else:
           return None  # Cannot establish connection

FUSION Configuration
--------------------

.. code-block:: ini

   [spectrum_settings]
   modulation_selection = snr_based

   [snr_settings]
   enable_snr = true
   snr_model = gaussian_noise  # or 'ase_noise', 'nonlinear'
   fiber_attenuation_db_km = 0.2
   amplifier_spacing_km = 80
   amplifier_noise_figure_db = 5.0

Spectrum Usage Impact
=====================

How Modulation Affects Spectrum
--------------------------------

Different modulations require different spectrum for same data rate:

**Example: 400 Gbps Connection**

.. code-block:: text

   BPSK (1 bit/sym):  48 slots  |████████████████████████████████████████████████|
   QPSK (2 bit/sym):  24 slots  |████████████████████████|
   8QAM (3 bit/sym):  16 slots  |████████████████|
   16QAM (4 bit/sym): 12 slots  |████████████|
   32QAM (5 bit/sym): 10 slots  |██████████|
   64QAM (6 bit/sym):  8 slots  |████████|

**Key Insight:** Higher modulation = Less spectrum = More connections possible

Network Capacity Analysis
-------------------------

**Scenario:** 20 links × 320 slots = 6400 total slots

**All QPSK (400 Gbps each, 24 slots):**

* Connections possible: 6400 / 24 = 266 connections
* Total capacity: 266 × 400 = 106.4 Tbps

**All 16QAM (400 Gbps each, 12 slots):**

* Connections possible: 6400 / 12 = 533 connections
* Total capacity: 533 × 400 = 213.2 Tbps

**2× capacity improvement** from better modulation!

Practical Considerations
========================

Modulation in Real Networks
----------------------------

**Coherent Detection:**

* Modern optical networks use coherent receivers
* Can detect phase and amplitude
* Enables high-order modulation
* Digital signal processing (DSP) compensates impairments

**Transponder Limitations:**

* Not all transponders support all modulations
* Bandwidth variable transponders (BVT) more flexible
* Cost increases with capability

**Operational Challenges:**

* SNR estimation accuracy
* Margin requirements (safety buffer)
* Aging fiber degrades performance
* Temperature variations affect signal

Safety Margins
--------------

In practice, add margin to thresholds:

.. code-block:: python

   # Add 2-3 dB margin for safety
   SAFETY_MARGIN_DB = 3.0

   def select_modulation_with_margin(snr_db):
       snr_available = snr_db - SAFETY_MARGIN_DB

       if snr_available >= 19.9:
           return '64QAM'
       # ...

**Why Margins:**

* Aging components
* Temperature variations
* Estimation errors
* Maintenance activities
* Unexpected impairments

FUSION Configuration Options
=============================

Fixed Modulation
----------------

Use same format for all connections:

.. code-block:: ini

   [spectrum_settings]
   modulation_selection = fixed
   modulation_format = QPSK

**Use When:**

* Simple scenarios
* Comparing algorithms
* Known network characteristics

Distance-Adaptive
-----------------

Automatic selection based on distance:

.. code-block:: ini

   [spectrum_settings]
   modulation_selection = distance_adaptive

**Use When:**

* General simulations
* Diverse path lengths
* Realistic scenarios

SNR-Based
---------

Selection based on signal quality:

.. code-block:: ini

   [spectrum_settings]
   modulation_selection = snr_based

   [snr_settings]
   enable_snr = true
   snr_model = gaussian_noise

**Use When:**

* Physical layer matters
* Research on signal quality
* Validating margin requirements

Custom Thresholds
-----------------

Override default values:

.. code-block:: ini

   [modulation_thresholds]
   # SNR thresholds (dB)
   bpsk_snr_db = 6.8
   qpsk_snr_db = 9.8
   8qam_snr_db = 12.6
   16qam_snr_db = 14.8
   32qam_snr_db = 17.1
   64qam_snr_db = 19.9

   # Reach thresholds (km)
   bpsk_reach_km = 6000
   qpsk_reach_km = 4000
   8qam_reach_km = 2000
   16qam_reach_km = 1000
   32qam_reach_km = 500
   64qam_reach_km = 250

Summary
=======

Key Takeaways
-------------

* ✓ **Trade-off Exists**: Data rate vs. reach
* ✓ **Distance-Adaptive**: Best for realistic simulations
* ✓ **Higher Order = More Efficient**: But shorter reach
* ✓ **SNR-Based More Accurate**: But more computation
* ✓ **Modulation Choice Matters**: Significantly affects blocking

Quick Reference
---------------

.. tip::
   **For beginners**: Use distance-adaptive modulation

.. tip::
   **For metro networks**: 16QAM or 32QAM

.. tip::
   **For long-haul**: QPSK or 8QAM

.. tip::
   **For maximum capacity**: SNR-based with highest viable format

.. tip::
   **For research**: Try multiple approaches and compare

Next Steps
==========

* :doc:`resource_allocation` - How modulation affects RSA
* :doc:`flex_grid_networks` - Spectrum allocation details
* :doc:`optical_networking_basics` - Physical layer fundamentals
* :doc:`../user_guide/running_simulations` - Configure modulation in FUSION
* :doc:`../examples/basic_simulation` - See modulation in action

References
==========

**Standards:**

* ITU-T G.698.2: Amplified multichannel DWDM applications
* ITU-T G.709: Interfaces for the optical transport network

**Further Reading:**

* :doc:`../reference/bibliography` - Research papers on modulation

.. seealso::

   * :doc:`optical_networking_basics` - Fundamentals
   * :doc:`wdm_vs_eon` - Evolution of optical networks
   * :doc:`resource_allocation` - How modulation affects resource allocation
