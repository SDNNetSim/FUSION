==============
SDN Overview
==============

Introduction
============

Software-Defined Networking (SDN) represents a fundamental architectural shift in how networks are designed, controlled, and managed. By separating the network control logic from the underlying forwarding hardware, SDN enables centralized, programmable control of network resources, dramatically improving flexibility, efficiency, and innovation potential.

In the context of optical networks—and specifically elastic optical networks modeled by FUSION—SDN principles provide the control framework necessary to dynamically manage spectrum, routing, and quality of service in response to changing traffic demands and network conditions.

This document provides a comprehensive overview of SDN principles, architecture, benefits, and applications to optical networking, helping you understand the control paradigm that FUSION implements.

What is Software-Defined Networking?
=====================================

Core Concept
------------

Traditional networks embed control logic directly into each network device. Routers and switches independently make forwarding decisions based on distributed protocols (e.g., OSPF, BGP) running locally.

**Software-Defined Networking (SDN)** decouples the control plane (decision-making logic) from the data plane (packet forwarding), centralizing control in a software-based controller.

::

    Traditional Network Architecture:

    ┌─────────────────────────────────────────────┐
    │             Management Plane                │  Human operators
    └────────────┬────────────────────────────────┘
                 │ (Manual configuration)
    ┌────────────▼────────────────────────────────┐
    │  ┌────────┐  ┌────────┐  ┌────────┐        │
    │  │ Router │  │ Router │  │ Router │        │
    │  │   +    │  │   +    │  │   +    │        │  Each device has
    │  │Control │  │Control │  │Control │        │  integrated control
    │  └────┬───┘  └────┬───┘  └────┬───┘        │
    │       │           │           │             │
    │  ┌────▼───┐  ┌────▼───┐  ┌────▼───┐        │
    │  │  Data  │──│  Data  │──│  Data  │        │  Data plane
    │  │ Plane  │  │ Plane  │  │ Plane  │        │
    │  └────────┘  └────────┘  └────────┘        │
    └─────────────────────────────────────────────┘


    SDN Architecture:

    ┌─────────────────────────────────────────────┐
    │         Application Plane                   │  Network apps
    └────────────┬────────────────────────────────┘
                 │ (Northbound API)
    ┌────────────▼────────────────────────────────┐
    │        SDN Controller                       │  Centralized
    │    (Centralized Control Logic)              │  control brain
    └────────────┬────────────────────────────────┘
                 │ (Southbound API - e.g., OpenFlow)
    ┌────────────▼────────────────────────────────┐
    │  ┌────────┐  ┌────────┐  ┌────────┐        │
    │  │ Switch │  │ Switch │  │ Switch │        │  Simple forwarding
    │  │ (Data) │──│ (Data) │──│ (Data) │        │  devices
    │  └────────┘  └────────┘  └────────┘        │
    └─────────────────────────────────────────────┘

**Key Difference**: In SDN, switches/routers become simple forwarding devices following instructions from a central controller, rather than making independent decisions.

Control Plane vs. Data Plane Separation
----------------------------------------

**Control Plane**
    The "brain" of the network that makes decisions about:
    - Where traffic should be sent (routing)
    - How to handle different types of packets (policies)
    - When to update forwarding rules
    - How to respond to failures

**Data Plane**
    The "hands" of the network that execute decisions:
    - Forwarding packets according to flow tables
    - Buffering and queuing
    - Basic packet processing
    - Performance at line rate

**Separation Benefits**:

- **Centralized view**: Controller sees entire network topology and state
- **Simplified devices**: Switches/routers become commodity hardware
- **Programmability**: Control logic written in software, easily updated
- **Vendor independence**: Open interfaces replace proprietary protocols

Historical Context
------------------

**Pre-SDN Era (1980s-2000s)**
    - Proprietary device control
    - Distributed control protocols
    - Manual configuration
    - Limited programmability

**Early SDN Concepts (2000s)**
    - Separation of control and data (4D project, RCP)
    - OpenFlow protocol (Stanford, 2008)
    - Initial academic research

**SDN Emergence (2010s)**
    - Google's B4 network (2012): First large-scale SDN deployment
    - Open Networking Foundation (ONF) formed
    - Commercial SDN controllers (OpenDaylight, ONOS, Cisco ACI)
    - Widespread industry adoption

**Current Era (2020s)**
    - SDN in data centers, WANs, optical networks
    - Integration with NFV, cloud orchestration
    - AI/ML-driven control
    - Intent-based networking

Why SDN Matters
================

Limitations of Traditional Networks
------------------------------------

Traditional distributed control architectures face significant challenges:

**Complexity**
    - Each device configured individually
    - Hundreds of protocols and features
    - Difficult to understand network-wide behavior

**Ossification**
    - New features require vendor support
    - Slow standardization process
    - Innovation constrained by hardware update cycles

**Inflexibility**
    - Static configurations
    - Manual provisioning (days to weeks)
    - Difficult to adapt to changing demands

**Suboptimal Performance**
    - Local decisions without global view
    - Inefficient resource utilization
    - Difficult to implement network-wide optimization

Benefits of SDN
---------------

SDN addresses these limitations through:

**Centralized Control**
    - Global network view enables optimal decisions
    - Consistent policy enforcement
    - Simplified troubleshooting (single point of visibility)

**Programmability**
    - Network behavior defined in software
    - Rapid deployment of new features
    - Automation of complex operations

**Agility**
    - Dynamic resource allocation
    - Fast provisioning (seconds to minutes)
    - Responsive to traffic changes

**Vendor Independence**
    - Open interfaces (OpenFlow, NETCONF)
    - Mix and match equipment
    - Reduced vendor lock-in

**Innovation**
    - Rapid prototyping of new protocols
    - Experimental deployments coexist with production
    - Research to production pipeline shortened

**Cost Reduction**
    - Commodity switching hardware
    - Reduced operational overhead (automation)
    - Better resource utilization

SDN Architecture
================

The Three-Layer Model
----------------------

SDN architecture is typically described in three layers:

::

    ┌─────────────────────────────────────────────────────────┐
    │                   APPLICATION LAYER                     │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │   Traffic    │  │    Routing   │  │   Security   │  │
    │  │ Engineering  │  │ Optimization │  │  Management  │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    └───────────────────────┬─────────────────────────────────┘
                            │ Northbound API (REST, etc.)
    ┌───────────────────────▼─────────────────────────────────┐
    │                    CONTROL LAYER                        │
    │         ┌─────────────────────────────────┐             │
    │         │       SDN Controller            │             │
    │         │  - Topology Discovery           │             │
    │         │  - Path Computation             │             │
    │         │  - Flow Management              │             │
    │         │  - State Management             │             │
    │         └─────────────────────────────────┘             │
    └───────────────────────┬─────────────────────────────────┘
                            │ Southbound API (OpenFlow, etc.)
    ┌───────────────────────▼─────────────────────────────────┐
    │                  INFRASTRUCTURE LAYER                   │
    │  ┌────────┐       ┌────────┐       ┌────────┐          │
    │  │ Switch │───────│ Switch │───────│ Switch │          │
    │  └────────┘       └────────┘       └────────┘          │
    │        │               │               │                │
    │  ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐        │
    │  │   Host    │   │   Host    │   │   Host    │        │
    │  └───────────┘   └───────────┘   └───────────┘        │
    └─────────────────────────────────────────────────────────┘

Application Layer
-----------------

The **application layer** contains network applications that define high-level behavior and policies:

**Traffic Engineering Applications**
    - Compute optimal routes based on network-wide state
    - Load balancing across multiple paths
    - QoS enforcement and bandwidth reservation

**Security Applications**
    - DDoS detection and mitigation
    - Access control and authentication
    - Intrusion detection and prevention

**Monitoring and Analytics**
    - Network telemetry collection
    - Performance analysis
    - Anomaly detection

**Orchestration**
    - Service provisioning
    - Resource allocation
    - Multi-domain coordination

Applications communicate with the controller via the **northbound API** (typically RESTful interfaces) to:
- Query network state (topology, statistics, flows)
- Install policies and configurations
- Receive event notifications

Control Layer
-------------

The **control layer** (SDN controller) is the network's "brain":

**Core Functions**:

**Topology Discovery**
    - Detect network devices and links
    - Maintain graph representation
    - Track changes dynamically

**State Management**
    - Collect device statistics
    - Maintain flow tables
    - Store configuration data

**Path Computation**
    - Compute shortest paths
    - Multi-constraint routing
    - Backup path calculation

**Flow Management**
    - Install flow rules in switches
    - Monitor flow statistics
    - Remove expired flows

**Event Processing**
    - React to link failures
    - Handle new device connections
    - Process packet-in messages

The controller exposes:
- **Northbound API**: For applications to interact with network
- **Southbound API**: For communication with infrastructure devices

Infrastructure Layer
--------------------

The **infrastructure layer** (data plane) consists of:

**SDN Switches**
    - Execute forwarding based on flow tables
    - Report statistics to controller
    - Forward unknown packets to controller (packet-in)

**Optical Devices** (in optical SDN)
    - ROADMs (Reconfigurable Optical Add-Drop Multiplexers)
    - Bandwidth-Variable Transponders (BVT)
    - Optical cross-connects

**Hosts/Endpoints**
    - Servers, storage, user devices
    - Generate and consume traffic

Devices communicate with controller via the **southbound API** (e.g., OpenFlow, NETCONF).

SDN Interfaces and Protocols
=============================

Southbound APIs
---------------

Southbound APIs enable controller-to-device communication:

**OpenFlow**
    Most prominent SDN protocol
    Defines flow table structure and message types
    Originally designed for packet switching (Layer 2/3)
    Extended for optical networks (see below)

**NETCONF/YANG**
    Configuration management protocol
    YANG models define device data structures
    More general than OpenFlow, supports legacy devices

**OVSDB**
    Open vSwitch Database Management Protocol
    Manages virtual switch configurations
    Complements OpenFlow

**BGP-LS (Link State)**
    Exports topology information
    Used for multi-domain and hybrid deployments

**PCEP (Path Computation Element Protocol)**
    Client-server protocol for path computation
    SDN controller acts as PCE server

Northbound APIs
---------------

Northbound APIs enable application-to-controller communication:

**RESTful APIs**
    Most common approach
    HTTP-based, JSON or XML payloads
    Easy integration with applications

**Intent-Based APIs**
    High-level, declarative specifications
    Controller translates intent to low-level configurations
    Example: "Provide 100 Gbps connectivity between DC1 and DC2"

**GraphQL**
    Query language for flexible data retrieval
    Client specifies exactly what data is needed

**gRPC**
    High-performance RPC framework
    Used for streaming telemetry and fast operations

East-West APIs
--------------

For multi-controller and multi-domain scenarios:

**Inter-Controller Communication**
    Synchronization of state between controllers
    Coordination of cross-domain paths

**Federation**
    Hierarchical control across administrative domains
    Abstraction of intra-domain details

OpenFlow Protocol
=================

Overview
--------

**OpenFlow** is the most widely adopted SDN southbound protocol, originally developed at Stanford University in 2008.

**Core Concept**: Switches maintain **flow tables** with entries specifying:
- **Match fields**: Criteria to identify packets (e.g., source/dest IP, port, VLAN)
- **Actions**: What to do with matching packets (forward, drop, modify, etc.)
- **Statistics**: Counters for matched packets/bytes

The controller populates and manages these flow tables remotely.

Flow Table Structure
--------------------

Each flow entry contains:

::

    ┌────────────────────────────────────────────────────────┐
    │  Match Fields          │  Priority  │  Counters        │
    ├────────────────────────────────────────────────────────┤
    │  Actions / Instructions                                │
    ├────────────────────────────────────────────────────────┤
    │  Timeouts              │  Cookie                       │
    └────────────────────────────────────────────────────────┘

**Match Fields**: Header field values to match (IP address, MAC, VLAN, etc.)

**Priority**: Used when multiple entries match (highest priority wins)

**Actions**: Operations to perform (output to port, modify headers, drop, etc.)

**Counters**: Statistics (packets matched, bytes matched, duration)

**Timeouts**: Idle timeout (remove if unused) and hard timeout (absolute expiration)

**Cookie**: Opaque identifier set by controller for tracking

OpenFlow Messages
-----------------

Communication between controller and switches uses OpenFlow messages:

**Controller-to-Switch Messages**:

- **Flow-Mod**: Add, modify, or delete flow entries
- **Packet-Out**: Send packet out specific switch port
- **Barrier**: Ensure previous messages are processed

**Switch-to-Controller Messages**:

- **Packet-In**: Send unmatched packet to controller for decision
- **Flow-Removed**: Notify when flow entry is removed
- **Port-Status**: Notify of port state changes (up/down)

**Symmetric Messages** (both directions):

- **Hello**: Establish connection and negotiate version
- **Echo**: Keep-alive and latency measurement
- **Error**: Report problems

Typical OpenFlow Workflow
--------------------------

1. **Connection Establishment**
   - Switch connects to controller (TCP/TLS)
   - Handshake and version negotiation

2. **Topology Discovery**
   - Controller sends LLDP (Link Layer Discovery Protocol) packets
   - Learns switch-to-switch connections

3. **Packet Arrival at Switch**
   - Switch checks flow table for match
   - If matched: Execute actions
   - If not matched: Send Packet-In to controller

4. **Controller Decision**
   - Controller determines appropriate path
   - Installs flow entries along the path (Flow-Mod messages)

5. **Subsequent Packets**
   - Forwarded by switches based on installed flows
   - No controller involvement (fast path)

6. **Flow Removal**
   - Timeout expires or controller sends delete
   - Switch notifies controller (Flow-Removed)

Example Scenario
----------------

**Topology**: Three switches (S1, S2, S3) in a line, controller C

::

    Host A ──── S1 ──── S2 ──── S3 ──── Host B
                 │       │       │
                 └───────┼───────┘
                         │
                    Controller C

**Scenario**: Host A sends packet to Host B for the first time

1. Packet arrives at S1, no matching flow entry
2. S1 sends Packet-In to controller C
3. C computes path: S1 → S2 → S3
4. C installs flows:
   - S1: Match (src=A, dst=B) → Action: Forward to S2
   - S2: Match (src=A, dst=B) → Action: Forward to S3
   - S3: Match (src=A, dst=B) → Action: Forward to Host B
5. C sends Packet-Out to S1 to forward original packet
6. Subsequent packets from A to B forwarded by switches without controller involvement

Benefits for Optical Networks
==============================

Centralized Control of Complex Resources
-----------------------------------------

Optical networks involve intricate physical layer phenomena (dispersion, nonlinearity, OSNR, etc.) that are difficult to manage with distributed control.

**SDN Advantage**: Centralized controller with complete network state can:

- Compute routes considering physical constraints
- Optimize spectrum allocation across entire network
- Balance multiple objectives (blocking, energy, cost)
- React to physical layer alarms and reroute connections

Programmability and Automation
-------------------------------

Optical network provisioning traditionally requires manual configuration of multiple devices (transponders, ROADMs, amplifiers).

**SDN Advantage**: Automated provisioning workflow:

1. Application requests connection (source, destination, bandwidth)
2. Controller computes route and spectrum assignment
3. Controller configures all devices along path
4. Connection established in seconds (vs. days manually)

This enables:
- On-demand bandwidth services
- Dynamic reconfiguration
- Self-healing upon failures

Dynamic Spectrum Management
----------------------------

Flex-grid elastic optical networks require fine-grained spectrum control (see :doc:`flex_grid_networks`).

**SDN Advantage**: Controller maintains real-time spectrum occupancy state:

- Tracks which slots are used on each link
- Computes contiguous and continuous spectrum assignments
- Implements defragmentation algorithms
- Optimizes spectrum utilization network-wide

Multi-Layer Optimization
-------------------------

Modern networks have multiple layers (IP, optical, Ethernet).

**SDN Advantage**: Unified control across layers:

- Coordinate IP routing with optical lightpath provisioning
- Optimize jointly (e.g., minimize IP router ports by creating direct optical paths)
- Implement multi-layer protection and restoration

Quality of Service (QoS) Guarantees
------------------------------------

Applications have diverse requirements (latency, bandwidth, reliability).

**SDN Advantage**: Controller enforces QoS policies:

- Prioritize critical traffic
- Reserve spectrum bandwidth for guaranteed services
- Ensure Quality of Transmission (QoT) meets thresholds
- Monitor SLA compliance

Rapid Innovation and Experimentation
-------------------------------------

New optical technologies (modulation formats, coding schemes, switching techniques) emerge frequently.

**SDN Advantage**: Software-based control enables:

- Rapid prototyping of new algorithms
- A/B testing of control strategies
- Gradual rollout of new features
- Research-to-production pipeline acceleration

SDN Controllers for Optical Networks
=====================================

General-Purpose SDN Controllers
-------------------------------

**OpenDaylight (ODL)**
    - Open-source, modular Java-based controller
    - Extensive plugin ecosystem
    - OpenFlow, NETCONF, BGP-LS support
    - Used in many optical SDN research projects

**ONOS (Open Network Operating System)**
    - Open-source, distributed controller
    - Designed for carrier-grade deployments
    - High availability and scalability
    - Intent-based northbound API

**Ryu**
    - Lightweight Python-based controller
    - Component-based architecture
    - Easy to extend and customize
    - Popular in research and prototyping

**Floodlight**
    - Open-source Java controller
    - REST API for applications
    - Performance-optimized
    - Used in commercial products

Optical-Specific Controllers and Platforms
-------------------------------------------

**ONOS with ODTN (Open Disaggregated Transport Network)**
    - Extension of ONOS for optical networks
    - Supports disaggregated optical components
    - OpenConfig and NETCONF interfaces
    - Deployed by AT&T, NTT, etc.

**OpenROADM**
    - ROADM-specific control framework
    - Defines device models and interfaces
    - Interoperability across vendors
    - Integrated with OpenDaylight

**Cassini**
    - SDN controller for multi-layer optical networks
    - Path computation engine (PCE)
    - GMPLS and OpenFlow integration

**Commercial Controllers**
    - Cisco's Crosswork Optimization Engine
    - Nokia's NSP (Network Services Platform)
    - Ciena's Blue Planet
    - Huawei's iMaster NCE

Extensions for Optical Networks
--------------------------------

Standard OpenFlow is designed for packet networks (Layer 2/3). Optical networks required extensions:

**OpenFlow Extensions for Optical (OFO)**
    - Flow entries match optical parameters (wavelength, spectrum slots)
    - Actions include configuring transponders, tuning lasers, switching wavelengths

**NETCONF/YANG Models**
    - Standardized data models for optical devices
    - OpenROADM YANG models
    - ONF Transport API

**TAPI (Transport API)**
    - ONF standard for transport network control
    - Service-oriented, technology-agnostic
    - Supports Ethernet, OTN, optical

Software-Defined Elastic Optical Networks (SD-EON)
===================================================

Concept
-------

**Software-Defined Elastic Optical Networks (SD-EON)** combine two paradigms:

- **Elastic Optical Networks (EON)**: Flex-grid spectrum allocation (see :doc:`flex_grid_networks`)
- **Software-Defined Networking (SDN)**: Centralized, programmable control

**Result**: Dynamic, efficient, programmable optical networks that adapt to traffic demands in real-time.

::

    ┌──────────────────────────────────────────────┐
    │         SDN Applications                     │
    │  - Traffic Engineering                       │
    │  - Spectrum Defragmentation                  │
    │  - Restoration Management                    │
    └──────────────────┬───────────────────────────┘
                       │ Northbound API
    ┌──────────────────▼───────────────────────────┐
    │         SDN Controller                       │
    │  ┌─────────────────────────────────────┐    │
    │  │  - RSA (Routing & Spectrum          │    │
    │  │    Assignment) Algorithms           │    │
    │  │  - Spectrum State Database          │    │
    │  │  - QoT Estimation Engine            │    │
    │  │  - Defragmentation Logic            │    │
    │  └─────────────────────────────────────┘    │
    └──────────────────┬───────────────────────────┘
                       │ Southbound API (OpenFlow, NETCONF)
    ┌──────────────────▼───────────────────────────┐
    │      Elastic Optical Network                 │
    │  ┌────────┐      ┌────────┐      ┌────────┐ │
    │  │  BVT   │──────│ ROADM  │──────│  BVT   │ │
    │  │(Node A)│      │        │      │(Node B)│ │
    │  └────────┘      └────────┘      └────────┘ │
    │       │                               │      │
    │    ┌──▼──┐                         ┌──▼──┐  │
    │    │Host │                         │Host │  │
    │    └─────┘                         └─────┘  │
    └──────────────────────────────────────────────┘

SD-EON Architecture Components
-------------------------------

**Controller Functions**:

**Topology and Spectrum State Management**
    - Maintain network graph with fiber links
    - Track spectrum slot availability on each link
    - Update state as connections established/torn down

**Routing and Spectrum Assignment (RSA)**
    - Compute routes considering physical constraints (distance, QoT)
    - Assign contiguous and continuous spectrum slots
    - Optimize objectives (blocking, fragmentation, energy)

**Quality of Transmission (QoT) Estimation**
    - Model signal degradation (OSNR, dispersion, nonlinearity)
    - Select appropriate modulation format based on path
    - Ensure signal quality meets receiver thresholds

**Modulation Format Selection**
    - Choose format based on distance and QoT
    - Configure bandwidth-variable transponders (BVT)
    - Balance spectral efficiency and reach

**Defragmentation**
    - Detect spectrum fragmentation
    - Rearrange existing connections to consolidate free spectrum
    - Execute hitless or disruptive defragmentation

**Protection and Restoration**
    - Compute backup paths
    - Detect failures via monitoring
    - Trigger fast restoration upon failure

**Device Configuration**:

The controller configures optical devices:

**Bandwidth-Variable Transponders (BVT)**
    - Central frequency (wavelength)
    - Modulation format (BPSK, QPSK, 16-QAM, etc.)
    - Symbol rate and bandwidth
    - Transmit power

**ROADMs (Reconfigurable Optical Add-Drop Multiplexers)**
    - Which wavelengths/spectrum slots to add, drop, or pass through
    - Spectrum slot boundaries
    - Attenuation per channel

**Optical Amplifiers**
    - Gain levels
    - Tilt (frequency-dependent gain)

RSA in SD-EON
-------------

The **Routing and Spectrum Assignment (RSA)** problem is central to SD-EON:

**Input**:
- Network topology (nodes, links, spectrum slots per link)
- Connection request (source, destination, data rate)
- Current spectrum occupancy state

**Output**:
- Route (sequence of nodes and links)
- Spectrum assignment (contiguous slot range, continuous across route)
- Modulation format

**Constraints**:
- Contiguity: Slots must be adjacent
- Continuity: Same slots on all links
- Availability: Slots must be free
- QoT: Signal quality must meet threshold

**Objectives** (optimization goals):
- Minimize blocking probability
- Minimize spectrum usage
- Minimize fragmentation
- Minimize energy consumption
- Balance load across links

The controller implements RSA algorithms (see :doc:`rsa_problem` for details).

Spectrum Defragmentation
-------------------------

As connections dynamically arrive and depart, spectrum becomes fragmented (see :doc:`flex_grid_networks`).

**SD-EON Advantage**: Controller can proactively defragment:

**Monitoring**
    Controller tracks fragmentation metrics (e.g., external fragmentation ratio)

**Triggering**
    When fragmentation exceeds threshold, initiate defragmentation

**Planning**
    Compute new spectrum assignments that consolidate free spectrum

**Execution**
    - Hitless: Establish new paths, switch traffic, tear down old paths
    - Disruptive: Tear down and re-establish connections

**Validation**
    Verify fragmentation reduced and no QoT violations

Dynamic Lightpath Provisioning
-------------------------------

**Workflow** for establishing a connection in SD-EON:

1. **Request Arrival**
   - Application requests connection (src, dst, rate)
   - Controller receives request via northbound API

2. **Route Computation**
   - Controller computes candidate routes (k-shortest paths)
   - Considers link costs, distance, available spectrum

3. **QoT Estimation**
   - For each candidate route, estimate signal quality
   - Determine feasible modulation formats
   - Calculate required spectrum slots

4. **Spectrum Assignment**
   - Search for available contiguous slots on each link
   - Ensure continuity across route
   - Apply spectrum assignment policy (first-fit, best-fit, etc.)

5. **Resource Reservation**
   - Mark assigned slots as occupied in state database
   - Prevent double-allocation

6. **Device Configuration**
   - Configure source BVT (frequency, modulation, power)
   - Configure intermediate ROADMs (add/drop/pass-through)
   - Configure destination BVT (receive parameters)

7. **Validation**
   - Verify signal quality via monitors
   - Confirm connection established
   - Return success to application

8. **Teardown** (when connection no longer needed)
   - Application requests release
   - Controller tears down connection
   - Frees spectrum slots
   - Updates state database

Practical Example
=================

Scenario: Data Center Interconnect
-----------------------------------

Consider a service provider operating an SD-EON connecting multiple data centers:

**Network**:
- 10 nodes (data centers and intermediate sites)
- 15 fiber links (metro and regional distances)
- C-band spectrum: 320 slots (12.5 GHz each = 4 THz total)

**Traffic**:
- 50+ active connections at any time
- Data rates: 10 Gbps to 400 Gbps
- Dynamic traffic: connections established/torn down throughout day

**SDN Controller Deployment**:

The controller is deployed as a centralized service with:

- **High availability**: Primary and backup controller instances
- **Scalability**: Handles 100+ connection requests per hour
- **Interfaces**:
  - Southbound: NETCONF/YANG to optical devices
  - Northbound: REST API for orchestration system

Use Case 1: On-Demand Connection
---------------------------------

**Scenario**: Data center DC1 needs to transfer 100 TB to DC2 (backup job)

**Traditional Approach**:
- Manual ticket created
- Engineer plans path and spectrum
- Multiple devices configured individually
- Takes hours to days

**SD-EON Approach**:
1. Orchestration system requests 400 Gbps connection (DC1 → DC2)
2. Controller computes route considering current load
3. Distance: 800 km → Select QPSK modulation
4. Required spectrum: 200 GHz (16 slots)
5. Finds available slots [50-65] on all links of chosen route
6. Configures BVTs and ROADMs automatically
7. Connection established in seconds
8. After transfer complete (6 hours later), connection automatically torn down
9. Spectrum released for other uses

**Benefits**:
- Fast provisioning (seconds vs. hours)
- Automated workflow
- Efficient spectrum utilization
- On-demand capacity

Use Case 2: Failure Restoration
--------------------------------

**Scenario**: Fiber cut between nodes N3 and N4

**Detection**:
- ROADMs detect loss of signal
- Send alarms to controller
- Controller marks link N3-N4 as failed

**Impact**:
- 5 active connections traverse N3-N4

**SD-EON Response**:
1. Controller identifies affected connections
2. For each connection:
   - Compute backup route avoiding failed link
   - Find available spectrum on backup route
   - Configure devices along new route
   - Switch traffic to backup path
3. Total restoration time: < 100 milliseconds (hitless)
4. When fiber repaired, revert to original paths if desired

**Benefits**:
- Fast restoration (sub-second)
- Automated recovery
- No manual intervention
- Service continuity maintained

Use Case 3: Spectrum Defragmentation
-------------------------------------

**Scenario**: Over time, spectrum becomes fragmented

**Monitoring**:
- Controller calculates external fragmentation ratio (EFR)
- EFR exceeds threshold (e.g., 0.4)

**Impact**:
- Several 200 Gbps requests blocked despite sufficient total free spectrum

**SD-EON Response**:
1. Controller triggers proactive defragmentation
2. Analyzes current connections and identifies rearrangement opportunities
3. Selects 10 connections to rearrange (chosen to maximize consolidation)
4. Computes new spectrum assignments that pack connections tightly
5. Executes hitless defragmentation:
   - Establish new paths alongside existing
   - Switch connections to new paths
   - Tear down old paths
6. Fragmentation reduced, large contiguous blocks available
7. Subsequent 200 Gbps requests successfully accommodated

**Benefits**:
- Improved spectrum utilization
- Reduced blocking
- Proactive management
- Automated optimization

How FUSION Models SDN
======================

FUSION is designed to simulate and analyze SDN-controlled elastic optical networks. Key aspects:

Controller Model
----------------

FUSION represents a centralized SDN controller that:

- Maintains network topology and state
- Receives connection requests
- Computes routes using various algorithms (shortest path, k-shortest paths, etc.)
- Assigns spectrum using different policies (first-fit, best-fit, etc.)
- Estimates QoT and selects modulation formats
- Tracks spectrum occupancy and fragmentation

See :doc:`../api/core/network` for implementation details.

Control Workflows
-----------------

FUSION simulates the controller's operational workflows:

**Connection Establishment**:

.. code-block:: python

    # Example FUSION code for provisioning connection
    from fusion.core.network import Network
    from fusion.rsa.algorithms import k_shortest_path_rsa

    network = Network.from_file("topology.json")
    request = {"source": "A", "destination": "B", "rate": 100}

    # Controller computes route and assigns spectrum
    result = k_shortest_path_rsa(network, request)

    if result.success:
        # Update network state
        network.establish_connection(result.route, result.spectrum)
        print(f"Connection established: {result}")
    else:
        print(f"Request blocked: {result.reason}")

**Dynamic Traffic Simulation**:

.. code-block:: python

    # Simulate dynamic traffic over time
    from fusion.simulation import TrafficSimulator

    simulator = TrafficSimulator(network)
    simulator.load_traffic_matrix("traffic.csv")
    simulator.run(duration=3600)  # 1 hour simulation

    # Analyze results
    print(f"Blocking probability: {simulator.blocking_probability}")
    print(f"Average spectrum utilization: {simulator.avg_utilization}")

See :doc:`../tutorials/dynamic_simulation` for detailed examples.

RSA Algorithms
--------------

FUSION implements various RSA algorithms that an SDN controller might use:

- K-shortest path with first-fit spectrum assignment
- Least-loaded routing
- Fragmentation-aware spectrum assignment
- Defragmentation heuristics

See :doc:`rsa_problem` and :doc:`../api/rsa/algorithms` for details.

QoT Estimation
--------------

FUSION models physical layer impairments and QoT estimation:

- OSNR degradation from amplifier noise
- Chromatic dispersion accumulation
- Nonlinear effects (simplified models)
- Modulation format selection based on estimated QoT

See :doc:`../guides/qot_estimation` for FUSION's QoT model.

Spectrum Management
-------------------

FUSION tracks spectrum state:

- Slot availability per link
- Contiguity and continuity enforcement
- Fragmentation metrics
- Spectrum occupancy visualization

See :doc:`spectrum_assignment` for spectrum management in FUSION.

Performance Metrics
-------------------

FUSION collects metrics relevant to SDN control:

- Blocking probability (percentage of requests rejected)
- Spectrum utilization (percentage of slots occupied)
- Fragmentation metrics (EFR, number of free blocks)
- Average connection holding time
- Provisioning success rate

These metrics evaluate controller algorithms and policies.

Limitations and Simplifications
--------------------------------

FUSION is a **simulator**, not a real SDN controller. Some simplifications:

**No Actual Device Control**
    FUSION models device behavior, doesn't configure real equipment

**Instantaneous Provisioning**
    FUSION doesn't model signaling delays or configuration time

**Perfect Information**
    Controller has complete, accurate network state (no measurement errors)

**Simplified QoT**
    Detailed physical layer effects use approximate models

**Single Controller**
    No multi-controller or distributed control modeled

Despite simplifications, FUSION provides valuable insights into SDN control strategies for optical networks.

Challenges and Future Directions
=================================

Current Challenges
------------------

**Scalability**
    Managing networks with thousands of nodes and millions of connections

**Latency**
    Controller response time impacts restoration speed and user experience

**Reliability**
    Controller failure can disrupt entire network (requires high availability)

**Security**
    Centralized control is attractive attack target (needs robust security)

**Interoperability**
    Multi-vendor environments require standardized interfaces

**Legacy Integration**
    Coexistence with non-SDN devices and protocols

**Complexity**
    Sophisticated algorithms can be computationally expensive

Future Research Directions
---------------------------

**Machine Learning Integration**
    - Traffic prediction and proactive provisioning
    - Intelligent routing and spectrum assignment
    - Anomaly detection and security

**Intent-Based Networking**
    - High-level service specifications
    - Automated policy translation
    - Self-driving networks

**Multi-Domain SDN**
    - Inter-controller coordination
    - Cross-domain path computation
    - Scalable federated control

**Network Slicing**
    - Virtual network instances on shared infrastructure
    - Resource isolation and QoS guarantees
    - 5G and beyond applications

**Disaggregated Networks**
    - Open optical line systems
    - Mix-and-match vendor components
    - Software-defined control of heterogeneous equipment

**Energy-Aware Control**
    - Power-efficient routing and spectrum assignment
    - Dynamic device sleep modes
    - Green networking objectives

**Quantum-Ready Networks**
    - Quantum key distribution (QKD) integration
    - Control of quantum channels
    - Co-optimization with classical traffic

Connection to FUSION
=====================

Understanding SDN principles is essential for using FUSION effectively:

**Control Paradigm**
    FUSION models centralized SDN control, not distributed GMPLS

**Global Optimization**
    FUSION algorithms leverage global network view for optimization

**Dynamic Management**
    FUSION simulates dynamic provisioning and teardown

**Programmability**
    FUSION allows implementing custom RSA algorithms (controller logic)

**Evaluation**
    FUSION measures metrics relevant to SDN control performance

To dive deeper:

- Understand flex-grid networks: :doc:`flex_grid_networks`
- Learn RSA problem and algorithms: :doc:`rsa_problem`
- Explore FUSION's network model: :doc:`../api/core/network`
- Run dynamic simulations: :doc:`../tutorials/dynamic_simulation`

Further Reading
===============

Books
-----

- Feamster, N., Rexford, J., & Zegura, E. (2014). "The Road to SDN: An Intellectual History of Programmable Networks". *ACM SIGCOMM Computer Communication Review*.

- Kreutz, D., et al. (2015). "Software-Defined Networking: A Comprehensive Survey". *Proceedings of the IEEE*, 103(1), 14-76.

- Azodolmolky, S. (2013). *Software Defined Networking with OpenFlow*. Packt Publishing.

- Goransson, P., Black, C., & Culver, T. (2016). *Software Defined Networks: A Comprehensive Approach*. Morgan Kaufmann.

Standards
---------

- ONF OpenFlow Specification v1.5
- ONF SDN Architecture
- IETF RFC 7426: Software-Defined Networking (SDN) - Layers and Architecture Terminology
- ITU-T Y.3300: Framework of software-defined networking

Key Papers
----------

- McKeown, N., et al. (2008). "OpenFlow: Enabling Innovation in Campus Networks". *ACM SIGCOMM Computer Communication Review*.

- Jain, S., et al. (2013). "B4: Experience with a globally-deployed software defined WAN". *ACM SIGCOMM*.

- Liu, L., et al. (2015). "Field trial of an OpenFlow-based unified control plane for multilayer multigranularity optical switching networks". *IEEE/OSA Journal of Lightwave Technology*.

- Channegowda, M., et al. (2013). "Experimental demonstration of an OpenFlow based software-defined optical network employing packet, fixed and flexible DWDM grid technologies". *Optics Express*.

Optical SDN Resources
---------------------

- ONF: Open Disaggregated Transport Network (ODTN) project
- TIP (Telecom Infra Project): Open Optical & Packet Transport group
- OIF (Optical Internetworking Forum): Flex Ethernet and FlexO specifications

See Also
========

- :doc:`optical_networking_basics` - Foundational optical concepts
- :doc:`flex_grid_networks` - Elastic optical network architecture
- :doc:`rsa_problem` - Routing and spectrum assignment problem
- :doc:`../api/core/network` - FUSION's network model
- :doc:`../tutorials/dynamic_simulation` - Simulating SDN-controlled networks
- :doc:`../guides/qot_estimation` - Quality of transmission estimation
