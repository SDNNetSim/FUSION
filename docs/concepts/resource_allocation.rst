===========================
Resource Allocation in EONs
===========================

Introduction
============

Resource allocation in elastic optical networks (EONs) is one of the most critical and challenging problems in modern optical networking. At its core, the problem involves determining **how to route connection requests** through the network and **how to assign spectrum resources** to those connections in a way that maximizes network efficiency while satisfying physical constraints.

This document provides comprehensive coverage of routing and spectrum assignment algorithms, their complexity, implementation strategies, and how FUSION tackles these challenges.

The Fundamental Problem
=======================

What is Resource Allocation?
-----------------------------

When a connection request arrives in an optical network, the controller must make several interconnected decisions:

1. **Routing**: Which physical path should the connection take through the network?
2. **Spectrum Assignment**: Which frequency slots should be allocated to this connection?
3. **Modulation Selection**: Which modulation format should be used (affects reach and spectrum)?
4. **Core Assignment**: In multi-core fiber networks, which fiber core should be used?

These decisions must satisfy multiple constraints while optimizing network-wide objectives.

**Example Scenario**:

::

    Network:    A -------- B -------- C
                 \                   /
                  \-------- D -------/

    Request: 100 Gbps from A to C

    Routing Options:
        Path 1: A → B → C (2 hops)
        Path 2: A → D → C (2 hops)

    For Path 1 (distance: 800 km):
        Modulation: QPSK (for reach)
        Spectrum: 6 slots needed

    For Path 2 (distance: 1200 km):
        Modulation: QPSK
        Spectrum: 6 slots needed
        But: Path 2 might have less available spectrum

    Decision: Choose path with best spectrum availability,
              considering current network state

Problem Variants
----------------

The resource allocation problem has evolved with network technology:

**RWA (Routing and Wavelength Assignment)**
    Traditional WDM networks with fixed-grid channels

    - Route a lightpath through the network
    - Assign a wavelength (channel) to the connection
    - Each channel has fixed capacity
    - Wavelength continuity constraint applies

**RSA (Routing and Spectrum Assignment)**
    Elastic optical networks with flex-grid spectrum

    - Route a lightpath through the network
    - Assign contiguous spectrum slots
    - Variable bandwidth per connection
    - Spectrum continuity and contiguity constraints
    - **Primary focus of FUSION**

**RMSA (Routing, Modulation, and Spectrum Assignment)**
    RSA with adaptive modulation selection

    - Select modulation format based on path distance and QoT
    - Modulation choice affects spectrum requirement
    - Distance-adaptive transmission
    - Trade-off between reach and spectral efficiency

**RMCSA (Routing, Modulation, Core, and Spectrum Assignment)**
    RMSA extended to multi-core fiber (space-division multiplexing)

    - Additional dimension: which fiber core to use
    - Inter-core crosstalk considerations
    - Massive capacity scaling
    - **Supported in FUSION for SDM networks**

Problem Complexity
==================

Why This Problem is Hard
-------------------------

The resource allocation problem in EONs is **NP-hard**, meaning there is no known polynomial-time algorithm that guarantees optimal solutions for all instances.

**Computational Complexity**:

Consider a network with:
- N = 20 nodes
- Average degree = 3 (3 links per node)
- Total links ≈ 30

For a single connection request from node A to node B:
- Possible paths: Can be exponential in network size
- For each path: Must check spectrum availability
- Spectrum slots per link: 320 (typical C-band with 12.5 GHz slots)
- Cores per link: 4-7 (for multi-core fiber)
- Modulation formats: 4-6 options

**Search space**: Millions of possible allocations for a single request!

For **static network design** (offline problem):
- Multiple demands (hundreds or thousands)
- Must jointly optimize all assignments
- Combinatorial explosion of possibilities
- ILP formulations can have millions of variables

For **dynamic networks** (online problem):
- Decisions made in real-time
- Limited time for computation (milliseconds)
- Current allocation affects future requests
- Cannot guarantee global optimality

Fundamental Constraints
-----------------------

Resource allocation algorithms must satisfy several critical constraints:

**Spectrum Continuity**
    The same spectrum slots must be available on every link along the path

    Without wavelength conversion (typical case), a connection using slots [5, 6, 7]
    on the first link must use the same slots [5, 6, 7] on all subsequent links.

**Spectrum Contiguity**
    Allocated spectrum slots must be adjacent (no gaps)

    A connection requiring 3 slots must use [5, 6, 7], not [5, 7, 9]

    Reason: Optical transmitters/receivers process continuous spectrum bands.

**Spectrum Non-Overlap**
    Different connections on the same fiber link cannot use the same spectrum slots

    Fundamental physics: signals on overlapping spectrum interfere with each other.

**Quality of Transmission (QoT)**
    Signal quality at destination must exceed minimum threshold

    - Depends on path length, amplifier noise, dispersion, nonlinear effects
    - Determines which modulation formats are feasible
    - Longer paths require more robust (lower spectral efficiency) modulations

**Guard Bands**
    Small spectral gaps between adjacent channels prevent inter-channel interference

    Typical: 1 slot (12.5 GHz) on each side of a channel

**Physical Layer**
    - Maximum optical path length
    - Amplifier spacing and gain
    - Dispersion compensation limits
    - Nonlinear effects (cross-phase modulation, four-wave mixing)

Optimization Objectives
-----------------------

Resource allocation algorithms aim to optimize various objectives:

**Primary Objectives**:

**Minimize Blocking Probability**
    Maximize the number of accepted requests

    Blocking occurs when no valid path+spectrum combination exists

**Maximize Spectrum Utilization**
    Use available spectrum efficiently

    Higher utilization = more revenue-generating traffic

**Minimize Spectrum Fragmentation**
    Avoid creating small, unusable spectrum gaps

    Fragmentation causes future blocking even when total free spectrum is sufficient

**Balance Network Load**
    Distribute traffic across links to avoid congestion

    Prevents hotspots and improves fault tolerance

**Secondary Objectives**:

- Minimize path length (reduce latency)
- Minimize number of hops (reduce switching complexity)
- Minimize energy consumption (fewer active devices)
- Maximize revenue (prioritize high-value requests)
- Ensure fairness (avoid starvation of long-distance requests)

**Multi-Objective Optimization**: Often these objectives conflict, requiring trade-offs and weighted optimization.

Routing Algorithms
==================

Routing determines the physical path a lightpath takes through the network. Different algorithms make different trade-offs between optimality, computation time, and network state awareness.

Shortest Path (Dijkstra's Algorithm)
-------------------------------------

**Concept**: Find the single shortest path from source to destination based on a metric.

**Algorithm**:

1. Assign each link a weight (e.g., physical distance, hop count)
2. Use Dijkstra's algorithm to find minimum-weight path
3. Return this single path

**Metrics**:
- **Hop count**: Minimize number of links traversed
- **Distance**: Minimize physical path length
- **Cost**: Minimize network operator's cost
- **Custom**: Any link property (congestion, available spectrum, etc.)

**Advantages**:
- Simple and fast (O(E + V log V) with Fibonacci heap)
- Guaranteed optimal for single-objective shortest path
- Deterministic and easy to debug

**Disadvantages**:
- Returns only one path (no alternatives if spectrum unavailable)
- May not consider current network state (spectrum availability)
- Can create network congestion hotspots
- No load balancing

**FUSION Implementation**:

.. code-block:: python

    from fusion.modules.routing import KShortestPath

    # Dijkstra is k=1 case of k-shortest paths
    routing_alg = KShortestPath(
        engine_props={'k_paths': 1, 'routing_weight': 'length'},
        sdn_props=sdn_props
    )

    path = routing_alg.route(source='A', destination='C', request=request)
    # Returns single shortest path

K-Shortest Paths (Yen's Algorithm)
-----------------------------------

**Concept**: Find the k-shortest paths, then try them in order until one has available spectrum.

**Algorithm** (Yen's algorithm):

1. Find shortest path using Dijkstra
2. For each deviation from previous paths:

   a. Remove one edge from previous path
   b. Find shortest path in modified graph
   c. Add to candidate paths

3. Select k-shortest unique paths
4. Return paths sorted by length/weight

**Advantages**:
- Multiple backup options if first path unavailable
- Still efficient: O(kN(E + N log N))
- Balances optimality and flexibility
- Commonly used in production networks

**Disadvantages**:
- More computation than single shortest path
- Paths may share many links (less diversity)
- Still doesn't directly consider spectrum state
- All paths evaluated sequentially (no parallelism)

**FUSION Implementation**:

.. code-block:: python

    from fusion.modules.routing import KShortestPath

    routing_alg = KShortestPath(
        engine_props={'k_paths': 3, 'routing_weight': 'length'},
        sdn_props=sdn_props
    )

    # Returns list of 3 shortest paths
    paths = routing_alg.get_paths(source='A', destination='C', k=3)

    # Try each path until spectrum assignment succeeds
    for path in paths:
        spectrum_assignment = spectrum_alg.assign(path, request)
        if spectrum_assignment is not None:
            # Success!
            break

**Typical k values**: 3-5 paths provide good balance

**Choosing Weight Metric**:

.. code-block:: python

    # By hop count (minimize number of links)
    routing_alg = KShortestPath(
        engine_props={'k_paths': 3, 'routing_weight': None},  # None = hop count
        sdn_props=sdn_props
    )

    # By distance (minimize physical length)
    routing_alg = KShortestPath(
        engine_props={'k_paths': 3, 'routing_weight': 'length'},
        sdn_props=sdn_props
    )

    # By custom weight (e.g., congestion, available spectrum)
    # First, set link weights in topology
    for u, v in topology.edges():
        topology[u][v]['congestion'] = calculate_congestion(u, v)

    routing_alg = KShortestPath(
        engine_props={'k_paths': 3, 'routing_weight': 'congestion'},
        sdn_props=sdn_props
    )

Adaptive Routing Algorithms
----------------------------

Adaptive algorithms adjust routing decisions based on current network state.

**Least Congested Path**

Route connections to avoid congested links.

**Concept**: Assign weights to links based on current spectrum utilization, then find shortest path.

.. code-block:: python

    from fusion.modules.routing import LeastCongested

    routing_alg = LeastCongested(
        engine_props=engine_props,
        sdn_props=sdn_props
    )

    # Dynamically updates link weights based on utilization
    path = routing_alg.route(source='A', destination='C', request=request)

**Advantages**:
- Balances load across network
- Reduces congestion hotspots
- Improves blocking probability

**Disadvantages**:
- May select longer paths
- Increased latency
- Requires real-time state information

**Fragmentation-Aware Routing**

Route connections to minimize spectrum fragmentation.

**Concept**: Consider not just whether spectrum is available, but whether using it will fragment remaining spectrum.

.. code-block:: python

    from fusion.modules.routing import FragmentationAware

    routing_alg = FragmentationAware(
        engine_props=engine_props,
        sdn_props=sdn_props
    )

    path = routing_alg.route(source='A', destination='C', request=request)

**Fragmentation Metrics**:
- External fragmentation ratio
- Number of free spectrum blocks
- Largest contiguous free block size

**Crosstalk-Aware Routing (Multi-Core Fiber)**

For space-division multiplexing networks, avoid inter-core crosstalk.

.. code-block:: python

    from fusion.modules.routing import XtAware

    routing_alg = XtAware(
        engine_props=engine_props,
        sdn_props=sdn_props
    )

    # Considers crosstalk between cores on same fiber
    path = routing_alg.route(source='A', destination='C', request=request)

**Nonlinear Impairment-Aware Routing**

Consider nonlinear effects (XPM, FWM) when routing.

.. code-block:: python

    from fusion.modules.routing import NliAware

    routing_alg = NliAware(
        engine_props=engine_props,
        sdn_props=sdn_props
    )

Spectrum Assignment Algorithms
===============================

Once a path is determined, the spectrum assignment algorithm allocates specific frequency slots to the connection.

First-Fit
---------

**Concept**: Allocate the first available contiguous block of spectrum slots that satisfies the request.

**Algorithm**:

1. For each core and band (C-band, L-band, etc.):

   a. Scan spectrum slots from lowest to highest index
   b. Find first contiguous block of sufficient size
   c. Check continuity along entire path
   d. If successful, allocate and return

2. If no suitable block found, return failure

**Example**:

::

    Spectrum state on path (each slot is 12.5 GHz):
    Link 1: [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    Link 2: [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

    Request: 3 contiguous slots

    Scan for common availability:
    Slots [0, 1, 2]: Link 1 ✓ (free), Link 2 ✓ (free) → First-fit selects [0, 1, 2]

**Advantages**:
- Very fast: O(S × L) where S = slots, L = links in path
- Simple implementation
- Low computational overhead
- Works well for low-to-moderate network load

**Disadvantages**:
- Can cause fragmentation by always using low-index slots
- No consideration of future requests
- May waste large contiguous blocks for small requests
- Greedy approach (no look-ahead)

**FUSION Implementation**:

.. code-block:: python

    from fusion.modules.spectrum import FirstFitSpectrum

    spectrum_alg = FirstFitSpectrum(
        engine_props=engine_props,
        sdn_props=sdn_props,
        route_props=route_props
    )

    assignment = spectrum_alg.assign(path=path, request=request)

    if assignment:
        print(f"Assigned slots {assignment['start_slot']}-{assignment['end_slot']}")
        print(f"Core: {assignment['core_number']}, Band: {assignment['band']}")

Best-Fit
--------

**Concept**: Allocate the smallest available contiguous block that can accommodate the request.

**Algorithm**:

1. Find all contiguous free blocks on the path
2. Filter blocks that are large enough for the request
3. Select the smallest sufficient block
4. Allocate and return

**Example**:

::

    Available blocks on path:
        Block A: slots [0, 1, 2, 3] (4 slots)
        Block B: slots [7, 8, 9, 10, 11, 12] (6 slots)
        Block C: slots [18, 19, 20] (3 slots)

    Request: 3 slots

    First-fit would choose: Block A (slots [0, 1, 2])
    Best-fit chooses: Block C (slots [18, 19, 20])

    Reason: Exact fit minimizes wasted space and preserves larger blocks

**Advantages**:
- Reduces spectrum fragmentation
- Preserves large contiguous blocks for future large requests
- Better spectrum efficiency under heterogeneous traffic
- Still relatively fast

**Disadvantages**:
- More computation than First-Fit (must scan all blocks)
- May spread allocations across spectrum (less consolidation)
- Can create many small fragments

**FUSION Implementation**:

.. code-block:: python

    from fusion.modules.spectrum import BestFitSpectrum

    spectrum_alg = BestFitSpectrum(
        engine_props=engine_props,
        sdn_props=sdn_props,
        route_props=route_props
    )

Last-Fit
--------

**Concept**: Allocate spectrum starting from the highest slot indices and working backward.

**Algorithm**:

1. Scan spectrum from highest index to lowest
2. Find first (from top) contiguous block
3. Allocate and return

**Advantages**:
- Consolidates free spectrum at lower indices
- Creates large contiguous blocks at bottom of spectrum
- Good for defragmentation strategies

**Disadvantages**:
- May allocate spectrum farther from ITU grid center
- Less efficient than best-fit for fragmentation

**FUSION Implementation**:

.. code-block:: python

    from fusion.modules.spectrum import LastFitSpectrum

Exact-Fit
---------

**Concept**: Prefer spectrum blocks that exactly match the request size.

Only allocate from larger blocks if no exact match exists.

**Advantages**:
- Minimizes fragmentation
- Optimal for predictable traffic patterns

**Disadvantages**:
- Higher blocking if exact matches rare
- More complex implementation

Random-Fit
----------

**Concept**: Randomly select from available blocks.

**Advantages**:
- Spreads allocations across spectrum
- Avoids systematic biases
- Good for research and comparison baselines

**Disadvantages**:
- No optimization objective
- Unpredictable performance
- Not used in production

Modulation and Core Assignment
===============================

Modulation Format Selection
----------------------------

Modulation format determines how many bits are encoded per symbol, affecting both **spectral efficiency** and **reach** (maximum transmission distance).

**Common Modulation Formats**:

::

    Format    Bits/Symbol    Spectral Eff.    Typical Reach    OSNR Required
    ------    -----------    -------------    -------------    -------------
    BPSK           1             1 b/s/Hz         ~3000 km          ~9 dB
    QPSK           2             2 b/s/Hz         ~2000 km         ~12 dB
    8-QAM          3             3 b/s/Hz         ~1000 km         ~16 dB
    16-QAM         4             4 b/s/Hz          ~600 km         ~20 dB
    32-QAM         5             5 b/s/Hz          ~400 km         ~24 dB
    64-QAM         6             6 b/s/Hz          ~200 km         ~28 dB

**Trade-off**: Higher spectral efficiency ↔ Lower reach

**Distance-Adaptive Modulation**:

Select modulation format based on path length:

.. code-block:: python

    def select_modulation(path_length_km: float) -> str:
        """Select modulation format based on path length."""
        if path_length_km < 600:
            return "16-QAM"  # High efficiency for short paths
        elif path_length_km < 1000:
            return "8-QAM"
        elif path_length_km < 2000:
            return "QPSK"
        else:
            return "BPSK"  # Maximum reach for long paths

**Example**:

::

    Request: 100 Gbps, Path length: 800 km

    Selected modulation: 8-QAM (3 bits/symbol)
    Symbol rate: 100 Gbps / 3 = 33.33 GBaud
    Spectrum needed: 33.33 GHz × 1.2 (guard + FEC) ≈ 40 GHz
    Slots needed: ceil(40 / 12.5) = 4 slots

    Alternative (if forced QPSK):
    Symbol rate: 100 Gbps / 2 = 50 GBaud
    Spectrum needed: 60 GHz ≈ 5 slots

    8-QAM saves 1 slot (12.5 GHz) of spectrum!

**QoT-Aware Modulation Selection**:

FUSION estimates signal quality (QoT) along the path and selects the most spectrally efficient modulation that meets SNR requirements.

See :doc:`../guides/qot_estimation` for FUSION's detailed QoT model.

Core Assignment (Multi-Core Fiber)
-----------------------------------

In space-division multiplexing (SDM) networks with multi-core fiber, an additional decision is required: **which fiber core** to use.

**Constraints**:
- Inter-core crosstalk (signals on adjacent cores interfere)
- Spectrum continuity within selected core
- Crosstalk depends on spectrum overlap on adjacent cores

**Strategies**:

**First-Fit Core**
    Try cores in order (core 0, core 1, ...) until assignment succeeds

**Least-Used Core**
    Assign to core with most available spectrum

    Balances load across cores

**Crosstalk-Aware Core**
    Consider crosstalk from adjacent cores

    .. code-block:: python

        def calculate_crosstalk(core_num, adjacent_cores, spectrum_range):
            """Calculate expected crosstalk on this core."""
            xt_sum = 0.0
            for adj_core in adjacent_cores:
                overlap = calculate_spectrum_overlap(
                    core_num, adj_core, spectrum_range
                )
                xt_sum += crosstalk_coefficient * overlap
            return xt_sum

**Joint Routing, Core, and Spectrum Assignment (RCSA)**:

Some advanced algorithms jointly optimize all decisions:

.. code-block:: python

    # Crosstalk-aware routing considers cores during path selection
    from fusion.modules.routing import XtAware

    routing_alg = XtAware(
        engine_props=engine_props,
        sdn_props=sdn_props
    )

    # Returns path AND preferred core assignments
    result = routing_alg.route(source='A', destination='C', request=request)

Algorithm Families
==================

Beyond specific algorithms, resource allocation approaches can be categorized into broader families.

Greedy Heuristics
-----------------

**Concept**: Make locally optimal decisions at each step without reconsidering past choices.

**Examples**:
- First-fit spectrum assignment
- Shortest path routing
- K-shortest paths with sequential trial

**Characteristics**:
- Fast (polynomial time)
- Simple implementation
- No optimality guarantee
- May get trapped in local optima

**Performance**:
- Works well for low-to-moderate network load
- Degrades under high load or fragmentation
- Typically 10-30% higher blocking than optimal (in small networks)

**Use Cases**:
- Real-time dynamic networks
- Large-scale networks where optimal algorithms intractable
- Initial solutions for metaheuristics

Integer Linear Programming (ILP)
---------------------------------

**Concept**: Formulate resource allocation as mathematical optimization problem with integer decision variables.

**Formulation Example (Simplified RSA)**:

::

    Decision Variables:
        x[p, s] ∈ {0, 1}  # 1 if demand uses path p with starting slot s
        y[l, s] ∈ {0, 1}  # 1 if link l uses slot s

    Objective:
        Minimize: Σ(demands) blocking_penalty[d] * (1 - served[d])

    Constraints:
        # Each demand uses exactly one path and starting slot (if served)
        Σ(p, s) x[p, s] ≤ 1  for each demand

        # Spectrum continuity: same slots on all links in path
        # Spectrum non-overlap: each slot on each link used by ≤1 connection
        # Contiguity: allocated slots must be adjacent

        # Physical constraints (QoT, reach, etc.)

**Advantages**:
- Optimal or near-optimal solutions (if solved to optimality)
- Can incorporate complex constraints
- Guarantees feasibility if solution exists

**Disadvantages**:
- Computationally expensive: NP-hard, exponential worst-case time
- Only practical for small networks (< 20 nodes) or offline planning
- Requires specialized solvers (CPLEX, Gurobi)
- Not suitable for real-time decisions in large networks

**FUSION Usage**:

ILP formulations are primarily used for:
- Benchmarking heuristic algorithms
- Offline network planning
- Small-scale scenario optimization

Metaheuristic Algorithms
-------------------------

**Concept**: Sophisticated search strategies that explore solution space intelligently.

**Genetic Algorithms (GA)**

Inspired by biological evolution:

1. Generate initial population of solutions (random or heuristic)
2. Evaluate fitness (e.g., blocking probability)
3. Selection: Choose best-performing solutions
4. Crossover: Combine pairs of solutions
5. Mutation: Introduce random changes
6. Repeat until convergence or time limit

**Simulated Annealing**

Inspired by metallurgical annealing:

1. Start with initial solution
2. Generate neighbor solution (small random change)
3. If better, accept
4. If worse, accept with probability depending on "temperature"
5. Gradually decrease temperature (reduce randomness)

**Ant Colony Optimization (ACO)**

Inspired by ant foraging behavior:

1. Virtual "ants" explore paths through network
2. Successful paths deposited with "pheromone"
3. Future ants biased toward high-pheromone paths
4. Pheromones evaporate over time
5. Emergent optimal or near-optimal paths

**Advantages**:
- Can find high-quality solutions for large problems
- Escape local optima
- Flexible: can optimize complex objectives

**Disadvantages**:
- No optimality guarantee
- Slower than greedy heuristics
- Many parameters to tune
- Stochastic (results vary across runs)

**Use Cases**:
- Offline network planning and design
- Periodic network optimization (defragmentation)
- Research and benchmarking

Machine Learning-Based Algorithms
----------------------------------

Modern networks leverage ML to learn optimal allocation strategies from data.

See :doc:`machine_learning_optical` for comprehensive coverage.

**Supervised Learning**

Learn routing/spectrum decisions from labeled training data:

.. code-block:: python

    # Decision tree classifier for routing selection
    from sklearn.tree import DecisionTreeClassifier

    # Features: src, dst, bandwidth, network state
    # Label: best path index (from k-shortest paths)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # At runtime: predict best path
    best_path_index = model.predict(network_features)

**Advantages**:
- Fast inference at runtime
- Learn from historical data
- Can capture complex patterns

**Disadvantages**:
- Requires large labeled dataset
- May not generalize to unseen scenarios
- Training data bias

**Reinforcement Learning (RL)**

Learn allocation policy through trial-and-error interaction:

.. code-block:: python

    # Q-learning for spectrum assignment

    State: network spectrum state, request parameters
    Actions: choose spectrum block
    Reward: +1 if request succeeds, -1 if blocked

    # Agent learns Q(state, action) = expected future reward

**Advantages**:
- No labeled data required
- Adapts to changing traffic patterns
- Can optimize long-term objectives

**Disadvantages**:
- Requires extensive training
- Exploration vs exploitation trade-off
- Convergence not guaranteed

**Deep Reinforcement Learning (DRL)**

Combine deep neural networks with RL:

.. code-block:: python

    from stable_baselines3 import PPO
    from fusion.modules.rl.envs import OpticalNetworkEnv

    # Create OpenAI Gym environment
    env = OpticalNetworkEnv(config)

    # Train DRL agent
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000000)

    # Use trained agent for spectrum assignment
    obs = env.reset()
    action, _states = model.predict(obs)

**FUSION ML/RL Support**:

FUSION provides:
- Integration with scikit-learn for supervised learning
- Custom OpenAI Gym environments for RL
- StableBaselines3 integration for DRL
- Training data generation and export
- Pre-trained model deployment

Performance Metrics and Trade-offs
===================================

Evaluating Algorithms
---------------------

Resource allocation algorithms are evaluated using multiple metrics:

**Blocking Probability (BP)**
    Fraction of requests that cannot be accommodated

    .. math::

        BP = \frac{\text{Blocked Requests}}{\text{Total Requests}}

    Lower is better. Target: < 1% for production networks.

**Bandwidth Blocking Ratio (BBR)**
    Fraction of requested bandwidth that is blocked

    .. math::

        BBR = \frac{\sum \text{Blocked Bandwidth}}{\sum \text{Requested Bandwidth}}

    Accounts for request size (large requests contribute more).

**Spectrum Utilization**
    Percentage of spectrum currently allocated

    .. math::

        Utilization = \frac{\text{Used Slots}}{\text{Total Available Slots}}

    Higher utilization is better, but 100% means no spare capacity.

**Spectrum Efficiency**
    Total throughput per unit spectrum

    .. math::

        Efficiency = \frac{\sum \text{Data Rates (Gbps)}}{\text{Total Spectrum (GHz)}}

    Measures bits/s/Hz. Higher is better.

**Fragmentation Ratio**
    Measure of spectrum fragmentation

    .. math::

        Fragmentation = 1 - \frac{\text{Largest Free Block}}{\text{Total Free Spectrum}}

    Higher fragmentation = more wasted spectrum.

**Average Path Length**
    Mean distance/hops of established connections

    Shorter paths: lower latency, fewer resources used.

**Computation Time**
    Time to make routing and spectrum assignment decision

    Critical for dynamic networks. Target: < 10 ms for real-time.

**Fairness**
    Distribution of blocking across connection types

    Ensure short and long distance requests have equitable service.

Algorithm Comparison
--------------------

**Typical Performance (Dynamic Network, Moderate Load)**:

::

    Algorithm              Blocking    Computation    Fragmentation
                          Probability     Time
    ------------------------------------------------------------------
    First-Fit + Shortest      12%        0.5 ms          High

    First-Fit + K-SP          8%         2 ms            Medium

    Best-Fit + K-SP           7%         5 ms            Low

    Fragmentation-Aware       6%         8 ms            Low

    ILP (Small Network)       4%        1000 ms          Very Low

    GA Metaheuristic          5%        500 ms           Low

    DRL (Trained)             6%         3 ms            Low

**Key Trade-offs**:

**Optimality vs Speed**
    Optimal algorithms (ILP) too slow for large dynamic networks

    Heuristics sacrifice optimality for speed

**Simplicity vs Performance**
    First-fit is simplest but worst blocking

    Advanced algorithms improve performance at cost of complexity

**Current State Awareness vs Overhead**
    Adaptive algorithms reduce blocking but require state tracking

    State management adds computational and memory overhead

**Load Balancing vs Path Length**
    Spreading load may require longer paths

    Increases latency and resource usage

Static vs Dynamic Allocation
-----------------------------

**Static (Offline) Allocation**:

- All demands known in advance
- Allocate all connections simultaneously
- Can use sophisticated optimization (ILP, metaheuristics)
- Used for network planning and design

**Dynamic (Online) Allocation**:

- Requests arrive and depart over time
- Allocate in real-time without future knowledge
- Must use fast heuristics or trained ML models
- Reflects real-world network operation
- **Primary focus of FUSION simulations**

**Semi-Static**:

- Periodic re-optimization (e.g., nightly)
- Defragmentation during low-traffic periods
- Balance between static and dynamic

How FUSION Implements These Algorithms
=======================================

Architecture Overview
---------------------

FUSION uses a modular, pluggable architecture for resource allocation:

::

    Request → Routing Module → Spectrum Module → Core Module → Allocation
                   ↓                ↓                 ↓
              [Algorithm]      [Algorithm]       [Algorithm]
              k-shortest-path   first-fit         first-fit-core
              congestion-aware  best-fit          crosstalk-aware
              fragmentation-aware last-fit        ...
              xt-aware          ...
              nli-aware
              ...

Each module implements a standard interface, allowing easy swapping and comparison.

Routing Module
--------------

All routing algorithms inherit from ``AbstractRoutingAlgorithm``:

.. code-block:: python

    from fusion.interfaces.router import AbstractRoutingAlgorithm

    class CustomRouting(AbstractRoutingAlgorithm):
        def route(self, source, destination, request):
            """Find a path from source to destination."""
            # Your routing logic here
            return path  # List of node IDs

**Built-in Routing Algorithms**:

- ``KShortestPath``: Dijkstra, k-shortest paths (Yen's algorithm)
- ``LeastCongested``: Load-aware routing
- ``FragmentationAware``: Spectrum-fragmentation-aware routing
- ``XtAware``: Crosstalk-aware routing for SDM
- ``NliAware``: Nonlinear-impairment-aware routing

**Example Configuration**:

.. code-block:: python

    engine_props = {
        'routing_algorithm': 'k_shortest_path',
        'k_paths': 3,
        'routing_weight': 'length',
    }

Spectrum Module
---------------

All spectrum algorithms inherit from ``AbstractSpectrumAssigner``:

.. code-block:: python

    from fusion.interfaces.spectrum import AbstractSpectrumAssigner

    class CustomSpectrum(AbstractSpectrumAssigner):
        def assign(self, path, request):
            """Assign spectrum along the path."""
            # Your spectrum assignment logic here
            return {
                'start_slot': start,
                'end_slot': end,
                'core_number': core,
                'band': band
            }

**Built-in Spectrum Algorithms**:

- ``FirstFitSpectrum``: First-fit allocation
- ``BestFitSpectrum``: Best-fit allocation
- ``LastFitSpectrum``: Last-fit allocation
- ``LightPathSlicing``: Dynamic bandwidth slicing

**Example Configuration**:

.. code-block:: python

    engine_props = {
        'spectrum_algorithm': 'first_fit',
    }

Real FUSION Example
-------------------

Complete example showing resource allocation in FUSION:

.. code-block:: python

    from fusion import Fusion
    from fusion.modules.routing import KShortestPath
    from fusion.modules.spectrum import FirstFitSpectrum

    # Configure network
    engine_props = {
        'topology_file': 'NSFNet.json',
        'slots_per_link': 320,
        'slot_size_ghz': 12.5,
        'cores_per_link': 1,
        'k_paths': 3,
        'routing_algorithm': 'k_shortest_path',
        'routing_weight': 'length',
        'spectrum_algorithm': 'first_fit',
    }

    # Initialize FUSION engine
    fusion_engine = Fusion(engine_props)

    # Create traffic demand
    request = {
        'source': 0,  # Node ID
        'destination': 5,
        'bandwidth': 100,  # Gbps
        'holding_time': 3600,  # seconds
    }

    # Process request
    result = fusion_engine.process_request(request)

    if result['success']:
        print(f"Connection established!")
        print(f"Path: {result['path']}")
        print(f"Slots: {result['start_slot']}-{result['end_slot']}")
        print(f"Modulation: {result['modulation']}")
        print(f"Spectrum: {result['spectrum_ghz']} GHz")
    else:
        print(f"Request blocked: {result['reason']}")

**Comparing Algorithms**:

.. code-block:: python

    algorithms = ['first_fit', 'best_fit', 'last_fit']

    results = {}
    for alg in algorithms:
        engine_props['spectrum_algorithm'] = alg
        fusion_engine = Fusion(engine_props)

        # Run simulation
        stats = fusion_engine.run_simulation(num_requests=1000)

        results[alg] = {
            'blocking': stats['blocking_probability'],
            'utilization': stats['spectrum_utilization'],
            'fragmentation': stats['fragmentation_ratio'],
        }

    # Compare results
    for alg, metrics in results.items():
        print(f"{alg}:")
        print(f"  Blocking: {metrics['blocking']:.2%}")
        print(f"  Utilization: {metrics['utilization']:.2%}")
        print(f"  Fragmentation: {metrics['fragmentation']:.3f}")

**Advanced: Custom Algorithm**:

.. code-block:: python

    from fusion.interfaces.router import AbstractRoutingAlgorithm

    class MySmartRouting(AbstractRoutingAlgorithm):
        """Custom routing that considers spectrum AND distance."""

        def route(self, source, destination, request):
            # Get k-shortest paths
            k_paths = self.get_k_shortest_paths(source, destination, k=5)

            best_path = None
            best_score = float('inf')

            for path in k_paths:
                # Calculate path score
                distance = self.calculate_path_distance(path)
                fragmentation = self.calculate_path_fragmentation(path)
                available_spectrum = self.calculate_available_spectrum(path)

                # Weighted combination
                score = (
                    0.3 * distance +
                    0.5 * fragmentation -
                    0.2 * available_spectrum
                )

                if score < best_score:
                    best_score = score
                    best_path = path

            return best_path

    # Register and use custom algorithm
    from fusion.modules.routing.registry import register_routing_algorithm

    register_routing_algorithm('my_smart_routing', MySmartRouting)

    engine_props['routing_algorithm'] = 'my_smart_routing'

Research and Future Directions
===============================

Open Problems
-------------

Despite decades of research, several challenges remain:

**Online Optimization with Learning**
    Can we learn optimal online policies that match offline optimal solutions?

**Joint Optimization at Scale**
    How to jointly optimize routing, spectrum, modulation, and core for large networks in real-time?

**Fragmentation Prevention**
    Can we prevent fragmentation rather than just reacting to it?

**AI Explainability**
    ML/RL algorithms are black boxes. How to interpret and trust their decisions?

**Multi-Objective Optimization**
    How to balance conflicting objectives (blocking, latency, energy, cost) fairly?

Advanced Topics
---------------

**Sliceable Transponders**

Multi-flow transponders that support multiple connections:

- Routing and spectrum for each flow
- Aggregate capacity constraint at transponder
- Traffic grooming opportunities

**Elastic Bandwidth Adjustment**

Dynamically adjust allocated bandwidth during connection lifetime:

- Scale up when demand increases
- Scale down when demand decreases
- Requires make-before-break provisioning

**Survivability and Protection**

Ensure network resilience to failures:

- Dedicated protection (backup path always reserved)
- Shared protection (backup paths share resources)
- Restoration (find new path after failure)
- Joint optimization of working and protection paths

**Network Slicing**

Partition network into virtual networks:

- Each slice has isolated resources
- Different SLAs and QoS per slice
- Resource allocation within and across slices

Further Reading
===============

Foundational Papers
-------------------

**Routing and Wavelength Assignment (RWA)**:

- Zang, H., Jue, J. P., & Mukherjee, B. (2000). "A review of routing and wavelength assignment approaches for wavelength-routed optical WDM networks". *Optical Networks Magazine*, 1(1), 47-60.

**Routing and Spectrum Assignment (RSA)**:

- Christodoulopoulos, K., Tomkos, I., & Varvarigos, E. A. (2011). "Elastic bandwidth allocation in flexible OFDM-based optical networks". *Journal of Lightwave Technology*, 29(9), 1354-1366.

- Chatterjee, B. C., Sarma, N., & Oki, E. (2015). "Routing and spectrum allocation in elastic optical networks: A tutorial". *IEEE Communications Surveys & Tutorials*, 17(3), 1776-1800.

**RMSA and Adaptive Modulation**:

- Klinkowski, M., & Walkowiak, K. (2011). "Routing and spectrum assignment in spectrum sliced elastic optical path network". *IEEE Communications Letters*, 15(8), 884-886.

**ILP Formulations**:

- Walkowiak, K., Klinkowski, M., Rabiega, B., & Goścień, R. (2014). "Routing and spectrum allocation algorithms for elastic optical networks with dedicated path protection". *Optical Switching and Networking*, 13, 63-75.

**Machine Learning for RSA**:

- Chen, X., Li, B., Proietti, R., et al. (2018). "DeepRMSA: A deep reinforcement learning framework for routing, modulation and spectrum assignment in elastic optical networks". *Journal of Lightwave Technology*, 37(16), 4155-4163.

Books
-----

- Tomkos, I., Azodolmolky, S., Sole-Pareta, J., Careglio, D., & Palkopoulou, E. (Eds.). (2017). *Elastic Optical Networks: Architectures, Technologies, and Control*. Springer.

- Mukherjee, B. (2006). *Optical WDM Networks*. Springer. (Chapters on RWA)

- Zervas, G., & Simeonidou, D. (2012). "Cognitive optical networks: Need, enabling technologies and applications". *ICTON*, We.B1.1.

See Also
========

Related FUSION Documentation:

- :doc:`flex_grid_networks` - Flexible spectrum allocation fundamentals
- :doc:`modulation_formats` - Modulation format characteristics and selection
- :doc:`machine_learning_optical` - ML/RL approaches to resource allocation
- :doc:`network_topologies` - How topology affects routing
- :doc:`../guides/qot_estimation` - Quality of transmission modeling in FUSION
- :doc:`../tutorials/getting_started` - Hands-on FUSION tutorial
- :doc:`../api/modules/routing` - Routing algorithm API reference
- :doc:`../api/modules/spectrum` - Spectrum assignment API reference
