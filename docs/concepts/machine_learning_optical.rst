======================================
Machine Learning in Optical Networks
======================================

Introduction
============

Machine learning (ML) and artificial intelligence (AI) are transforming optical networking, enabling intelligent, adaptive, and efficient network operation. As optical networks grow in complexity and traffic patterns become increasingly dynamic, traditional algorithmic approaches struggle to achieve optimal performance. ML offers a data-driven alternative that can learn from experience, adapt to changing conditions, and discover patterns that human designers might miss.

This document explores how machine learning is applied to optical networks, with particular focus on routing, spectrum assignment, quality of transmission (QoT) prediction, and failure management. We cover the theoretical foundations, practical applications, and how FUSION enables ML/RL-based network optimization.

Why Machine Learning for Optical Networks?
===========================================

The Challenge of Traditional Approaches
----------------------------------------

Classical optimization algorithms for optical networks face several fundamental challenges:

**Computational Complexity**
    Resource allocation problems (RSA, RMCSA) are NP-hard. Optimal solutions require exponential time, making them impractical for large networks or real-time decisions.

**Dynamic Uncertainty**
    Traffic patterns change over time. Algorithms designed for one traffic distribution may perform poorly under different conditions.

**Multi-Objective Trade-offs**
    Networks must balance blocking probability, spectrum utilization, energy consumption, latency, and fairness. Weighting these objectives manually is difficult.

**Physical Layer Complexity**
    Accurately modeling signal quality (QoT) requires considering many interacting impairments: attenuation, dispersion, nonlinear effects, crosstalk. Analytical models are complex and may not capture all effects.

**Lack of Adaptability**
    Fixed heuristics (e.g., First-Fit, k-shortest paths) cannot adapt to network state or learn from past performance.

What Machine Learning Offers
-----------------------------

ML addresses these challenges by:

**Learning from Data**
    Instead of hand-crafting algorithms, ML learns optimal policies from historical data or simulated experience.

**Handling Complexity**
    Neural networks can approximate complex functions (QoT, crosstalk) without explicit mathematical models.

**Adaptation**
    ML models can adapt to changing traffic patterns, network topology, and failure scenarios.

**Speed**
    Once trained, ML models make decisions quickly (milliseconds), suitable for real-time network control.

**Discovering Patterns**
    ML can identify patterns and correlations in network data that humans might not recognize.

**Example**: A reinforcement learning agent learns that, during peak hours, routing through the network's center causes congestion and fragmentation. It adapts by preferring edge paths during these periods, reducing blocking by 15% compared to static k-shortest paths.

Real-World Impact
-----------------

ML is being adopted by network operators and equipment vendors:

- **Google**: Uses ML for traffic engineering and QoT prediction in optical WANs
- **AT&T**: Employs ML for failure prediction and network optimization
- **Huawei**: Develops AI-driven SDN controllers for optical networks
- **Facebook/Meta**: Applies ML to data center optical interconnects
- **Academic Research**: Hundreds of papers annually on ML for optical networks

Machine Learning Approaches
============================

Machine learning encompasses several paradigms, each suited to different networking problems.

Supervised Learning
-------------------

**Concept**: Learn a mapping from inputs to outputs using labeled training data.

**Training Process**:

1. Collect dataset of (input, output) pairs
2. Train model to minimize prediction error
3. Validate on unseen data
4. Deploy model for inference

**Input Features** (for optical networks):
- Network topology (node/link features)
- Traffic demand parameters (source, destination, bandwidth)
- Current network state (spectrum utilization, active connections)
- Historical statistics (blocking rate, average path length)

**Output Labels**:
- Best routing path (classification)
- Spectrum slots to allocate (regression)
- Modulation format (classification)
- QoT estimate (regression)

**Common Algorithms**:

**Decision Trees**
    Tree-based models that partition feature space

    - Advantages: Interpretable, fast training, handles non-linear relationships
    - Disadvantages: Can overfit, unstable
    - Use case: Routing path selection

**Random Forests**
    Ensemble of decision trees

    - Advantages: More robust than single trees, good generalization
    - Disadvantages: Less interpretable, slower inference
    - Use case: QoT prediction

**Support Vector Machines (SVM)**
    Find optimal hyperplane separating classes

    - Advantages: Effective in high-dimensional spaces
    - Disadvantages: Slow training for large datasets
    - Use case: Modulation format classification

**Neural Networks**
    Multi-layer networks of artificial neurons

    - Advantages: Can approximate any function, powerful feature learning
    - Disadvantages: Requires large data, "black box"
    - Use case: Complex QoT estimation, traffic prediction

**Example: Routing Path Selection with Decision Tree**:

.. code-block:: python

    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    # Training data: features and labels
    # Features: [src, dst, bandwidth, load_link1, load_link2, ...]
    X_train = np.array([
        [0, 5, 100, 0.3, 0.5, 0.2],  # Sample 1
        [0, 5, 100, 0.8, 0.4, 0.3],  # Sample 2
        # ... more samples
    ])

    # Labels: which path was best (0 = path 0, 1 = path 1, etc.)
    y_train = np.array([0, 1, ...])

    # Train classifier
    model = DecisionTreeClassifier(max_depth=10)
    model.fit(X_train, y_train)

    # At runtime: predict best path
    new_request = np.array([[0, 5, 100, 0.6, 0.7, 0.2]])
    best_path_index = model.predict(new_request)[0]

**FUSION Integration**:

.. code-block:: python

    # Configure ML-based routing in FUSION
    engine_props = {
        'ml_settings': {
            'ml_training': False,  # Using pre-trained model
            'deploy_model': True,
            'ml_model': 'decision_tree',
            'model_path': 'models/routing_classifier.pkl',
        }
    }

Unsupervised Learning
---------------------

**Concept**: Find patterns and structure in data without labeled outputs.

**Use Cases in Optical Networks**:

**Clustering**
    Group similar traffic demands or network states

    - Identify traffic classes (small/large, short/long distance)
    - Detect anomalies (unusual traffic patterns)
    - Network segmentation (group nodes by function)

**Dimensionality Reduction**
    Compress high-dimensional network state into lower dimensions

    - Principal Component Analysis (PCA)
    - Autoencoders
    - Useful for visualization and feature extraction

**Anomaly Detection**
    Identify unusual network behavior

    - Predict failures before they occur
    - Detect security threats (DDoS, intrusion)
    - Quality degradation alerts

**Example: Traffic Pattern Clustering**:

.. code-block:: python

    from sklearn.cluster import KMeans

    # Features: demand characteristics
    demands = np.array([
        [10, 500],   # [bandwidth (Gbps), distance (km)]
        [100, 500],
        [10, 2000],
        [100, 2000],
        # ... more demands
    ])

    # Cluster into 4 classes
    kmeans = KMeans(n_clusters=4)
    clusters = kmeans.fit_predict(demands)

    # Each cluster might represent:
    # 0: Small bandwidth, short distance
    # 1: Large bandwidth, short distance
    # 2: Small bandwidth, long distance
    # 3: Large bandwidth, long distance

    # Use different strategies for each cluster
    for cluster_id in range(4):
        cluster_demands = demands[clusters == cluster_id]
        strategy = design_strategy_for_cluster(cluster_demands)

Reinforcement Learning (RL)
----------------------------

**Concept**: An agent learns to make decisions by interacting with an environment, receiving rewards for good actions and penalties for bad ones.

**Key Components**:

**Agent**
    The decision maker (e.g., SDN controller)

**Environment**
    The optical network (topology, spectrum state, traffic)

**State**
    Current network situation (available spectrum, active connections, etc.)

**Action**
    Decision to make (choose path, assign spectrum, select modulation)

**Reward**
    Feedback signal (positive for successful allocation, negative for blocking)

**Policy**
    Strategy mapping states to actions (what the agent learns)

**RL Process**:

::

    Initialize agent with random or heuristic policy

    for each episode:
        Reset environment (empty network)

        for each time step:
            Observe current state
            Agent selects action based on policy
            Environment responds: new state and reward
            Agent updates policy based on experience

        Episode ends (all requests processed)

    After training: Deploy learned policy in network

**Why RL for Optical Networks?**

- No labeled data required (learns from interaction)
- Optimizes long-term objectives (not just immediate reward)
- Adapts to changing conditions
- Handles sequential decision making (routing affects future requests)

**RL Paradigms**:

**Model-Free RL**
    Agent learns policy directly from experience without modeling environment dynamics

    - Q-Learning, SARSA, Policy Gradients
    - Simpler, more general
    - Most common for optical networks

**Model-Based RL**
    Agent learns model of environment, then plans using model

    - Can be more sample-efficient
    - Requires accurate environment model
    - Less common in networking

**On-Policy vs Off-Policy**
    - On-policy: Learn about policy currently being followed
    - Off-policy: Learn about optimal policy while following exploratory policy

Deep Learning for Optical Networks
-----------------------------------

**Deep Learning** uses multi-layer neural networks to learn hierarchical representations.

**Architectures**:

**Deep Neural Networks (DNN)**
    Fully-connected multi-layer networks

    - Use case: QoT prediction, traffic forecasting
    - Input: Network state, demand parameters
    - Output: QoT estimate, predicted traffic

**Convolutional Neural Networks (CNN)**
    Networks with convolutional layers, good for spatial patterns

    - Use case: Process spectrum utilization matrices
    - Treat spectrum state as 2D image (links × slots)
    - Learn spatial patterns of fragmentation

**Recurrent Neural Networks (RNN) / LSTM**
    Networks with memory, good for sequential data

    - Use case: Time-series traffic prediction
    - Model temporal dependencies in traffic
    - Predict future demand arrivals

**Graph Neural Networks (GNN)**
    Networks that operate on graph-structured data

    - Use case: Network topology processing
    - Nodes = network nodes, Edges = links
    - Learn topology-aware features
    - Naturally suited to network problems

**Example: GNN for Routing**:

.. code-block:: python

    import torch
    import torch_geometric

    class NetworkGNN(torch.nn.Module):
        """Graph Neural Network for network topology processing."""

        def __init__(self, node_features, edge_features, hidden_dim):
            super().__init__()
            self.conv1 = torch_geometric.nn.GCNConv(node_features, hidden_dim)
            self.conv2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim)
            self.output = torch.nn.Linear(hidden_dim, 1)  # Path score

        def forward(self, graph_data):
            x, edge_index = graph_data.x, graph_data.edge_index

            # Graph convolutions
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))

            # Output path score
            return self.output(x)

    # Use for routing: score each candidate path
    gnn = NetworkGNN(node_features=10, edge_features=5, hidden_dim=64)
    path_scores = gnn(network_graph)
    best_path = torch.argmax(path_scores)

Use Cases in Optical Networks
==============================

Routing and Spectrum Assignment (RSA)
--------------------------------------

ML can learn intelligent RSA policies.

**Supervised Learning Approach**:

1. **Generate Training Data**: Run optimal ILP solver (offline) on many network scenarios
2. **Extract Features**: Network state, demand parameters
3. **Train Model**: Learn to predict ILP's decisions
4. **Deploy**: Use model for real-time decisions

**Advantages**: Fast inference (milliseconds) with near-optimal quality

**Reinforcement Learning Approach**:

1. **Define Environment**: Network simulator (e.g., FUSION)
2. **Define State**: Spectrum utilization, demand parameters
3. **Define Actions**: Choose path, spectrum slots, modulation
4. **Define Reward**: +1 for successful allocation, -10 for blocking
5. **Train Agent**: RL agent learns policy maximizing cumulative reward

**Advantages**: Adapts to traffic patterns, no need for optimal baseline

**Example: Q-Learning for Spectrum Assignment**:

.. code-block:: python

    import numpy as np

    class QLearningSpectrum:
        """Q-Learning for spectrum assignment."""

        def __init__(self, num_slots, learning_rate=0.1, discount=0.95, epsilon=0.1):
            self.num_slots = num_slots
            self.lr = learning_rate
            self.gamma = discount
            self.epsilon = epsilon
            # Q-table: Q[state, action] = expected reward
            self.q_table = {}

        def get_state(self, network):
            """Extract state representation from network."""
            # State: tuple of available spectrum blocks
            return tuple(network.get_free_blocks())

        def choose_action(self, state):
            """Choose spectrum block using epsilon-greedy policy."""
            if np.random.random() < self.epsilon:
                # Explore: random action
                return np.random.choice(len(state))
            else:
                # Exploit: best action
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(len(state))
                return np.argmax(self.q_table[state])

        def update(self, state, action, reward, next_state):
            """Update Q-value based on experience."""
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(state))
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(len(next_state))

            # Q-learning update
            current_q = self.q_table[state][action]
            max_next_q = np.max(self.q_table[next_state])
            new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state][action] = new_q

**FUSION RL Configuration**:

.. code-block:: python

    engine_props = {
        'rl_settings': {
            'is_training': True,
            'path_algorithm': 'q_learning',
            'spectrum_algorithm': 'q_learning',
            'learn_rate': 0.01,
            'discount_factor': 0.95,
            'epsilon_start': 0.2,
            'epsilon_end': 0.05,
            'reward': 1,
            'penalty': -100,
        }
    }

Quality of Transmission (QoT) Prediction
-----------------------------------------

Predicting signal quality is critical for modulation selection and feasibility checking.

**Traditional Approach**: Analytical models (Gaussian Noise model, etc.)
- Requires detailed knowledge of impairments
- Complex equations
- May not capture all effects

**ML Approach**: Learn QoT from measurements or simulations

**Supervised Learning**:

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor

    # Training data: features and measured QoT
    features = [
        # [path_length, num_hops, num_amplifiers, traffic_load, ...]
        [500, 3, 5, 0.6],
        [1200, 5, 12, 0.4],
        # ...
    ]
    qot_values = [28.5, 22.3, ...]  # OSNR in dB

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(features, qot_values)

    # Predict QoT for new path
    new_path_features = [[800, 4, 8, 0.5]]
    predicted_qot = model.predict(new_path_features)[0]

    # Select modulation based on QoT
    if predicted_qot > 28:
        modulation = "64-QAM"
    elif predicted_qot > 20:
        modulation = "16-QAM"
    elif predicted_qot > 12:
        modulation = "QPSK"
    else:
        modulation = "BPSK"

**Advantages**:
- More accurate than simplified analytical models
- Can capture complex interactions
- Fast inference

**Challenges**:
- Requires training data (measurements or detailed simulations)
- May not generalize to unseen scenarios
- Needs retraining as network changes

Traffic Prediction and Forecasting
-----------------------------------

Predicting future traffic helps with proactive resource allocation.

**Time-Series Forecasting**:

Use historical traffic data to predict future arrivals:

.. code-block:: python

    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    # Historical traffic data: requests per hour
    traffic_history = [120, 135, 140, ...]  # Time series

    # Prepare sequences for LSTM
    X, y = create_sequences(traffic_history, lookback=24)  # 24-hour lookback

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(24, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X, y, epochs=100)

    # Predict next hour's traffic
    predicted_traffic = model.predict(last_24_hours)

**Proactive Optimization**:

With traffic predictions:
- Pre-provision resources during expected demand spikes
- Trigger defragmentation during predicted low-traffic periods
- Adjust protection schemes based on expected load

Failure Detection and Management
---------------------------------

ML can predict failures before they occur and optimize recovery.

**Anomaly Detection**:

Detect unusual patterns indicating impending failure:

.. code-block:: python

    from sklearn.ensemble import IsolationForest

    # Normal network metrics: OSNR, BER, power levels, etc.
    normal_data = [
        [28.5, 1e-12, 0.5],  # [OSNR, BER, power]
        [28.3, 1.2e-12, 0.48],
        # ...
    ]

    # Train anomaly detector
    model = IsolationForest(contamination=0.01)  # Expect 1% anomalies
    model.fit(normal_data)

    # Monitor in real-time
    current_metrics = [22.1, 5e-9, 0.45]  # Degraded signal
    is_anomaly = model.predict([current_metrics])[0]  # Returns -1 if anomaly

    if is_anomaly == -1:
        alert("Potential failure detected!")
        trigger_proactive_rerouting()

**Failure Prediction**:

Predict remaining useful life of components:

- Train on historical failure data
- Features: component age, utilization, environmental conditions
- Output: probability of failure in next N hours
- Take preventive action before failure

**RL for Dynamic Restoration**:

Learn optimal restoration policies:

.. code-block:: python

    # RL agent learns to:
    # 1. Detect failure quickly
    # 2. Select best backup path
    # 3. Minimize disrupted traffic
    # 4. Optimize resource usage

    # State: network topology, failed links, active connections
    # Actions: choose restoration path for each disrupted connection
    # Reward: -penalty for disrupted traffic, -cost for resources used

Supervised Learning for RSA
============================

Detailed Example: Decision Tree for Routing
--------------------------------------------

Let's build a complete supervised learning system for routing path selection.

**Step 1: Data Collection**

Run simulations with optimal algorithm (e.g., ILP or exhaustive search) and collect:

.. code-block:: python

    import pandas as pd

    # Collect training data
    training_data = []

    for scenario in scenarios:
        network = create_network(topology)

        for request in requests:
            # Get k candidate paths
            k_paths = get_k_shortest_paths(request.source, request.dest, k=5)

            # For each path, extract features
            for i, path in enumerate(k_paths):
                features = extract_features(network, path, request)

                # Optimal algorithm determines best path
                is_best = (i == optimal_algorithm(network, k_paths, request))

                training_data.append({
                    **features,
                    'label': 1 if is_best else 0
                })

    df = pd.DataFrame(training_data)
    df.to_csv('routing_training_data.csv')

**Step 2: Feature Engineering**

Design informative features:

.. code-block:: python

    def extract_features(network, path, request):
        """Extract features for routing decision."""
        return {
            # Path characteristics
            'path_length': calculate_path_length(path),
            'num_hops': len(path) - 1,
            'path_index': path.index,  # Which k-path (0=shortest)

            # Request characteristics
            'bandwidth': request.bandwidth,
            'source': request.source,
            'destination': request.destination,

            # Network state (spectrum availability)
            'avg_available_slots': calculate_avg_available(network, path),
            'min_available_slots': calculate_min_available(network, path),
            'max_available_slots': calculate_max_available(network, path),

            # Congestion
            'avg_link_utilization': calculate_avg_utilization(network, path),
            'max_link_utilization': calculate_max_utilization(network, path),

            # Fragmentation
            'avg_fragmentation': calculate_avg_fragmentation(network, path),

            # Topology features
            'path_diversity': calculate_diversity_from_existing(network, path),
        }

**Step 3: Model Training**

.. code-block:: python

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Load data
    df = pd.read_csv('routing_training_data.csv')

    # Split features and labels
    feature_cols = ['path_length', 'num_hops', 'bandwidth', 'avg_available_slots', ...]
    X = df[feature_cols]
    y = df['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train decision tree
    model = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")

    # Feature importance
    importances = model.feature_importances_
    for feat, imp in zip(feature_cols, importances):
        print(f"{feat}: {imp:.3f}")

**Step 4: Deployment in FUSION**

.. code-block:: python

    import pickle
    from fusion.modules.routing import AbstractRoutingAlgorithm

    class MLRouting(AbstractRoutingAlgorithm):
        """ML-based routing using trained decision tree."""

        def __init__(self, engine_props, sdn_props):
            super().__init__(engine_props, sdn_props)

            # Load trained model
            model_path = engine_props['ml_settings']['model_path']
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

        def route(self, source, destination, request):
            # Get k candidate paths
            k_paths = self.get_k_shortest_paths(source, destination, k=5)

            # Extract features for each path
            features_list = []
            for path in k_paths:
                features = self.extract_features(path, request)
                features_list.append(features)

            # Predict best path
            predictions = self.model.predict_proba(features_list)
            best_path_idx = np.argmax(predictions[:, 1])  # Probability of being best

            return k_paths[best_path_idx]

    # Configure FUSION to use ML routing
    engine_props = {
        'routing_algorithm': 'ml_routing',
        'ml_settings': {
            'deploy_model': True,
            'model_path': 'models/decision_tree_routing.pkl',
        }
    }

Reinforcement Learning for Dynamic Networks
============================================

RL is particularly well-suited for dynamic optical networks where traffic arrives and departs over time.

Bandit Algorithms
-----------------

Multi-armed bandits are simple RL algorithms for stateless problems.

**Epsilon-Greedy Bandit**

Balance exploration and exploitation:

.. code-block:: python

    class EpsilonGreedyBandit:
        """Epsilon-greedy bandit for routing path selection."""

        def __init__(self, num_arms, epsilon=0.1):
            self.num_arms = num_arms  # Number of paths
            self.epsilon = epsilon
            self.q_values = np.zeros(num_arms)  # Estimated value of each arm
            self.counts = np.zeros(num_arms)  # Number of times each arm pulled

        def select_action(self):
            """Select path using epsilon-greedy policy."""
            if np.random.random() < self.epsilon:
                return np.random.randint(self.num_arms)  # Explore
            else:
                return np.argmax(self.q_values)  # Exploit

        def update(self, action, reward):
            """Update Q-value based on reward."""
            self.counts[action] += 1
            n = self.counts[action]

            # Incremental average
            self.q_values[action] += (reward - self.q_values[action]) / n

**Upper Confidence Bound (UCB)**

Optimistic exploration based on uncertainty:

.. code-block:: python

    class UCBBandit:
        """UCB bandit for routing path selection."""

        def __init__(self, num_arms, c=2.0):
            self.num_arms = num_arms
            self.c = c  # Exploration parameter
            self.q_values = np.zeros(num_arms)
            self.counts = np.zeros(num_arms)
            self.total_counts = 0

        def select_action(self):
            """Select path using UCB policy."""
            # Initialize: try each arm once
            if self.total_counts < self.num_arms:
                return self.total_counts

            # UCB formula
            ucb_values = self.q_values + self.c * np.sqrt(
                np.log(self.total_counts) / (self.counts + 1e-5)
            )
            return np.argmax(ucb_values)

        def update(self, action, reward):
            """Update Q-value based on reward."""
            self.counts[action] += 1
            self.total_counts += 1
            n = self.counts[action]
            self.q_values[action] += (reward - self.q_values[action]) / n

**FUSION Bandit Configuration**:

.. code-block:: python

    engine_props = {
        'rl_settings': {
            'path_algorithm': 'ucb_bandit',  # or 'epsilon_greedy_bandit'
            'epsilon': 0.1,  # For epsilon-greedy
            'ucb_c': 2.0,  # For UCB
        }
    }

Temporal Difference Learning (Q-Learning)
------------------------------------------

Q-learning learns the value of state-action pairs.

**Algorithm**:

::

    Initialize Q(s, a) arbitrarily for all states and actions

    for each episode:
        Initialize state s

        for each step in episode:
            Choose action a using epsilon-greedy policy from Q(s, ·)
            Take action a, observe reward r and next state s'

            # Q-learning update
            Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]

            s ← s'

        until terminal state reached

**Implementation for Spectrum Assignment**:

.. code-block:: python

    class QLearningSpectrum:
        """Q-Learning for spectrum assignment."""

        def __init__(self, num_slots, alpha=0.1, gamma=0.95, epsilon=0.1):
            self.num_slots = num_slots
            self.alpha = alpha  # Learning rate
            self.gamma = gamma  # Discount factor
            self.epsilon = epsilon  # Exploration rate
            self.q_table = {}  # Q(state, action)

        def get_state(self, network, path):
            """Discretize network state."""
            # State: available spectrum blocks on path
            blocks = network.get_free_blocks(path)
            return tuple(sorted(blocks))  # Hashable state representation

        def get_actions(self, state):
            """Get possible actions (spectrum blocks to assign)."""
            return list(state)  # Each free block is an action

        def select_action(self, state):
            """Epsilon-greedy action selection."""
            actions = self.get_actions(state)

            if np.random.random() < self.epsilon:
                return np.random.choice(actions)  # Explore
            else:
                # Exploit: choose best action
                q_values = [self.q_table.get((state, a), 0.0) for a in actions]
                return actions[np.argmax(q_values)]

        def update(self, state, action, reward, next_state):
            """Q-learning update."""
            # Current Q-value
            q_current = self.q_table.get((state, action), 0.0)

            # Max Q-value for next state
            if next_state is None:  # Terminal state
                q_max_next = 0.0
            else:
                next_actions = self.get_actions(next_state)
                q_values_next = [
                    self.q_table.get((next_state, a), 0.0) for a in next_actions
                ]
                q_max_next = max(q_values_next) if q_values_next else 0.0

            # Q-learning update rule
            q_new = q_current + self.alpha * (
                reward + self.gamma * q_max_next - q_current
            )
            self.q_table[(state, action)] = q_new

**FUSION Q-Learning Configuration**:

.. code-block:: python

    engine_props = {
        'rl_settings': {
            'is_training': True,
            'path_algorithm': 'q_learning',
            'spectrum_algorithm': 'q_learning',
            'learn_rate': 0.01,
            'discount_factor': 0.95,
            'epsilon_start': 0.2,
            'epsilon_end': 0.05,
            'decay_factor': 0.01,
            'reward': 1,
            'penalty': -100,
        }
    }

Deep Reinforcement Learning (DRL)
----------------------------------

Combine deep neural networks with RL to handle high-dimensional state spaces.

**Deep Q-Networks (DQN)**

Use neural network to approximate Q-function:

.. code-block:: python

    import torch
    import torch.nn as nn

    class DQN(nn.Module):
        """Deep Q-Network for spectrum assignment."""

        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )

        def forward(self, state):
            """Return Q-values for all actions."""
            return self.network(state)

    # Training loop
    dqn = DQN(state_dim=100, action_dim=10)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()

        for step in range(max_steps):
            # Select action
            with torch.no_grad():
                q_values = dqn(torch.FloatTensor(state))
            action = epsilon_greedy_select(q_values)

            # Take action
            next_state, reward, done = env.step(action)

            # Compute target Q-value
            with torch.no_grad():
                next_q = dqn(torch.FloatTensor(next_state)).max()
            target_q = reward + gamma * next_q * (1 - done)

            # Update network
            predicted_q = dqn(torch.FloatTensor(state))[action]
            loss = loss_fn(predicted_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            if done:
                break

**Proximal Policy Optimization (PPO)**

State-of-the-art policy gradient algorithm:

.. code-block:: python

    from stable_baselines3 import PPO
    from fusion.modules.rl.envs import OpticalNetworkEnv

    # Create FUSION environment
    env = OpticalNetworkEnv(config={
        'topology': 'NSFNet',
        'num_slots': 320,
        'traffic_load': 150,
    })

    # Train PPO agent
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)

    # Save trained model
    model.save('ppo_optical_network')

    # Deploy for inference
    model = PPO.load('ppo_optical_network')
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break

**FUSION DRL Configuration**:

.. code-block:: python

    engine_props = {
        'rl_settings': {
            'is_training': False,  # Using pre-trained model
            'spectrum_algorithm': 'ppo',
            'spectrum_model': 'models/ppo/NSFNet/model.zip',
            'device': 'cuda',  # Use GPU if available
            'render_mode': None,
        }
    }

Deep Learning Architectures for Networks
=========================================

Graph Neural Networks (GNN)
----------------------------

GNNs are naturally suited to network problems because networks are graphs.

**RouteNet Architecture**

Proposed by Google for network modeling:

.. code-block:: python

    import torch_geometric as pyg

    class RouteNet(torch.nn.Module):
        """RouteNet-style GNN for network modeling."""

        def __init__(self, hidden_dim=64):
            super().__init__()

            # Link embedding layers
            self.link_encoder = nn.Sequential(
                nn.Linear(link_features, hidden_dim),
                nn.ReLU()
            )

            # Path embedding layers
            self.path_encoder = nn.Sequential(
                nn.Linear(path_features, hidden_dim),
                nn.ReLU()
            )

            # Message passing between links and paths
            self.link_update = pyg.nn.GCNConv(hidden_dim, hidden_dim)
            self.path_update = pyg.nn.GCNConv(hidden_dim, hidden_dim)

            # Output: predicted QoT, delay, etc.
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, graph_data):
            # Embed links and paths
            link_embed = self.link_encoder(graph_data.link_features)
            path_embed = self.path_encoder(graph_data.path_features)

            # Message passing
            for _ in range(num_iterations):
                link_embed = self.link_update(link_embed, graph_data.link_edge_index)
                path_embed = self.path_update(path_embed, graph_data.path_edge_index)

            # Predict outputs
            return self.output(path_embed)

**Advantages**:
- Captures network topology structure
- Generalizes across different topologies
- Interpretable message passing

Feature Engineering for Optical Networks
=========================================

Effective ML requires good features. For optical networks:

Network Topology Features
--------------------------

- Node degree (number of links per node)
- Network diameter (longest shortest path)
- Clustering coefficient
- Betweenness centrality
- Shortest path distance between nodes

Link Features
-------------

- Physical length
- Current utilization (occupied slots / total slots)
- Available spectrum blocks
- Fragmentation metric
- Number of active connections

Path Features
-------------

- Path length (physical distance)
- Number of hops
- Average link utilization along path
- Minimum available slots on any link
- Path diversity (overlap with existing paths)

Request Features
----------------

- Bandwidth demand
- Source and destination nodes
- Request arrival time
- Expected holding time

Network State Features
----------------------

- Total network utilization
- Number of active connections
- Blocking rate (recent history)
- Fragmentation (network-wide)

Training Data and Labels
========================

Obtaining Training Data
-----------------------

**Simulation-Based**:

Generate training data from FUSION simulations:

.. code-block:: python

    # Configure FUSION to output training data
    engine_props = {
        'ml_settings': {
            'output_train_data': True,
            'train_data_path': 'data/training_data.csv',
        }
    }

    # Run simulation
    fusion_engine = Fusion(engine_props)
    fusion_engine.run_simulation(num_requests=10000)

    # Training data saved to CSV with features and labels

**Real Network Measurements**:

Collect data from operational networks:
- Network state snapshots
- Request arrivals and characteristics
- Allocation decisions and outcomes
- QoT measurements

**Challenges**:
- Privacy and proprietary concerns
- Limited availability of public datasets
- Need for large, diverse datasets

Labeling Strategies
-------------------

**Optimal Labels** (for supervised learning):

Run optimal algorithm (ILP) to label best decisions:

.. code-block:: python

    def generate_optimal_labels(network, requests):
        """Generate training data with optimal labels."""
        training_data = []

        for request in requests:
            # Get candidate solutions
            candidates = get_candidate_solutions(network, request)

            # Run ILP to find optimal
            optimal_solution = ilp_solve(network, request, candidates)

            # Label candidates
            for candidate in candidates:
                features = extract_features(network, candidate, request)
                label = 1 if candidate == optimal_solution else 0
                training_data.append((features, label))

        return training_data

**Heuristic Labels**:

If optimal solutions infeasible, use high-quality heuristics:

.. code-block:: python

    # Use best-performing heuristic as teacher
    teacher_algorithm = BestFitSpectrum()  # Known good algorithm

    for request in requests:
        teacher_solution = teacher_algorithm.solve(network, request)
        # Label this as "good" solution

**Reward-Based** (for RL):

Design reward function:

.. code-block:: python

    def calculate_reward(action_result):
        """Calculate reward for RL agent."""
        if action_result.success:
            reward = 1.0  # Request accepted

            # Bonus for efficient allocation
            reward += 0.1 * (1.0 - action_result.fragmentation)

            # Bonus for short path
            reward += 0.05 * (1.0 - action_result.path_length_normalized)
        else:
            reward = -10.0  # Request blocked (large penalty)

        return reward

Online vs Offline Learning
===========================

Offline Learning
----------------

**Training Phase** (offline):
- Collect large dataset
- Train model on historical data
- Validate and test performance

**Deployment Phase** (online):
- Use fixed trained model for inference
- No further learning during operation

**Advantages**:
- Controlled training environment
- Can ensure convergence before deployment
- No risk of degradation during operation

**Disadvantages**:
- Cannot adapt to changes (new traffic patterns, failures)
- Requires periodic retraining

Online Learning
---------------

**Concept**: Continue learning during deployment.

**Approaches**:

**Incremental Learning**:

Update model as new data arrives:

.. code-block:: python

    # Start with pre-trained model
    model = load_pretrained_model()

    # During operation
    for request in request_stream:
        # Make prediction
        action = model.predict(request)

        # Observe outcome
        outcome = execute_action(action)

        # Update model
        model.incremental_update(request, action, outcome)

**Advantages**:
- Adapts to changing conditions
- Improves over time with experience

**Challenges**:
- Risk of catastrophic forgetting (forget old knowledge)
- Exploration vs exploitation in live network
- Stability concerns

**Hybrid Approach**:

Combine offline pre-training with online fine-tuning:

.. code-block:: python

    # Phase 1: Offline pre-training (safe, controlled)
    model = train_offline(large_dataset)

    # Phase 2: Online fine-tuning (adaptive, with safeguards)
    while True:
        # Use model for decisions
        action = model.predict(state)

        # Collect experience
        experience = execute_and_observe(action)

        # Periodically fine-tune (e.g., nightly during low traffic)
        if time_for_update():
            model.fine_tune(recent_experience)

Real-World Applications and Challenges
=======================================

Industry Adoption
-----------------

**Google's Orion SDN Controller**:
- Uses ML for traffic engineering in B4 network
- Predicts bandwidth demand and QoT
- Reduced manual tuning by 90%

**AT&T**:
- ML-based failure prediction (95% accuracy, 24-hour advance warning)
- Reinforcement learning for resource allocation
- Network-wide optimization

**Facebook/Meta**:
- GNNs for data center network optimization
- Traffic prediction and proactive routing
- Express Backbone (EBB) optical WAN

Research Highlights
-------------------

**DeepRMSA** (Chen et al., 2018):
- Deep RL for routing, modulation, and spectrum assignment
- Outperforms heuristics by 30% in blocking probability

**RouteNet** (Rusek et al., 2020):
- GNN for network modeling and delay prediction
- Generalizes across different topologies

**Cognitive Optical Networks** (Mata et al., 2018):
- ML for self-optimization and self-healing
- Autonomous network operation

Practical Challenges
--------------------

**Data Scarcity**:
- Limited public datasets
- Difficulty obtaining real network data
- Domain shift (sim-to-real gap)

**Explainability**:
- Neural networks are black boxes
- Network operators need to understand decisions
- Safety and trust concerns

**Generalization**:
- Models may not generalize to unseen topologies or traffic patterns
- Requires diverse training data

**Computational Resources**:
- Training large models is expensive (time, energy, GPUs)
- Inference must be fast (milliseconds)

**Deployment Barriers**:
- Risk aversion in operational networks
- Need for extensive testing and validation
- Integration with existing systems

**Safety and Stability**:
- ML models can make unpredictable mistakes
- Need failsafes and fallback mechanisms
- Ensuring stability during online learning

FUSION's ML/RL Capabilities
============================

Supported ML Algorithms
-----------------------

**Supervised Learning**:
- Decision tree classifiers (sklearn)
- Random forests
- Support vector machines
- Neural networks (TensorFlow, PyTorch)

**Reinforcement Learning**:
- Q-Learning
- SARSA
- Epsilon-greedy bandit
- Upper Confidence Bound (UCB) bandit

**Deep Reinforcement Learning**:
- Deep Q-Networks (DQN)
- Proximal Policy Optimization (PPO)
- A2C, A3C
- Custom architectures

Training Environment
--------------------

FUSION provides Gymnasium-compatible environments:

.. code-block:: python

    from fusion.modules.rl.envs import OpticalNetworkEnv

    # Create environment
    env = OpticalNetworkEnv(config={
        'topology': 'NSFNet',
        'num_slots': 320,
        'cores_per_link': 1,
        'traffic_load': 150,
        'episode_length': 1000,
    })

    # Standard Gym interface
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

**Environment Features**:
- Configurable network topology
- Realistic traffic models
- Physical layer constraints (QoT, modulation)
- Multi-core fiber support
- Detailed metrics and logging

Integration with StableBaselines3
----------------------------------

FUSION integrates with StableBaselines3 for DRL:

.. code-block:: python

    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.callbacks import EvalCallback
    from fusion.modules.rl.envs import OpticalNetworkEnv

    # Create environment
    env = OpticalNetworkEnv(config)

    # Train PPO agent
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log='./logs/'
    )

    # Add evaluation callback
    eval_env = OpticalNetworkEnv(config)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        eval_freq=10000,
    )

    # Train
    model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback
    )

    # Save
    model.save('ppo_optical_network')

How to Use ML in FUSION Simulations
====================================

Example 1: Training a Decision Tree for Routing
------------------------------------------------

.. code-block:: python

    from fusion import Fusion

    # Step 1: Generate training data
    engine_props = {
        'topology_file': 'NSFNet.json',
        'routing_algorithm': 'k_shortest_path',
        'k_paths': 5,
        'spectrum_algorithm': 'first_fit',
        'ml_settings': {
            'output_train_data': True,
            'train_file_path': 'data/training_nsfnet.csv',
        }
    }

    fusion = Fusion(engine_props)
    fusion.run_simulation(num_requests=10000)
    print("Training data generated.")

    # Step 2: Train decision tree
    engine_props['ml_settings'] = {
        'ml_training': True,
        'ml_model': 'decision_tree',
        'train_file_path': 'data/training_nsfnet.csv',
        'test_size': 0.3,
        'model_save_path': 'models/dt_routing.pkl',
    }

    fusion = Fusion(engine_props)
    fusion.train_ml_model()
    print("Model trained and saved.")

    # Step 3: Deploy trained model
    engine_props['ml_settings'] = {
        'deploy_model': True,
        'ml_model': 'decision_tree',
        'model_path': 'models/dt_routing.pkl',
    }

    fusion = Fusion(engine_props)
    results = fusion.run_simulation(num_requests=5000)

    print(f"Blocking Probability: {results['blocking_probability']:.2%}")
    print(f"Spectrum Utilization: {results['spectrum_utilization']:.2%}")

Example 2: Training a DRL Agent
--------------------------------

.. code-block:: python

    from stable_baselines3 import PPO
    from fusion.modules.rl.envs import OpticalNetworkEnv

    # Create environment
    env_config = {
        'topology': 'NSFNet',
        'num_slots': 320,
        'traffic_load': 150,
        'episode_length': 1000,
    }
    env = OpticalNetworkEnv(env_config)

    # Create and train PPO agent
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1_000_000)

    # Save model
    model.save('models/ppo_nsfnet')

    # Deploy in FUSION
    engine_props = {
        'rl_settings': {
            'is_training': False,
            'spectrum_algorithm': 'ppo',
            'spectrum_model': 'models/ppo_nsfnet.zip',
        }
    }

    fusion = Fusion(engine_props)
    results = fusion.run_simulation(num_requests=5000)

Example 3: Q-Learning for Spectrum Assignment
----------------------------------------------

.. code-block:: python

    from fusion import Fusion

    # Train Q-learning agent
    engine_props = {
        'topology_file': 'NSFNet.json',
        'rl_settings': {
            'is_training': True,
            'spectrum_algorithm': 'q_learning',
            'learn_rate': 0.01,
            'discount_factor': 0.95,
            'epsilon_start': 0.3,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995,
            'reward': 1,
            'penalty': -100,
        }
    }

    fusion = Fusion(engine_props)

    # Train over multiple episodes
    for episode in range(100):
        stats = fusion.run_episode(num_requests=1000)
        print(f"Episode {episode}: Blocking = {stats['blocking']:.2%}")

    # Save trained Q-table
    fusion.save_q_table('models/q_table_spectrum.json')

    # Deploy trained agent
    engine_props['rl_settings']['is_training'] = False
    engine_props['rl_settings']['q_table_path'] = 'models/q_table_spectrum.json'

    fusion = Fusion(engine_props)
    results = fusion.run_simulation(num_requests=5000)

Further Reading
===============

Foundational Papers
-------------------

**Machine Learning for Optical Networks**:

- Musumeci, F., et al. (2018). "An overview on application of machine learning techniques in optical networks". *IEEE Communications Surveys & Tutorials*, 21(2), 1383-1408.

**Reinforcement Learning for RSA**:

- Chen, X., et al. (2018). "DeepRMSA: A deep reinforcement learning framework for routing, modulation and spectrum assignment in elastic optical networks". *Journal of Lightwave Technology*, 37(16), 4155-4163.

**Graph Neural Networks for Networks**:

- Rusek, K., et al. (2020). "RouteNet: Leveraging graph neural networks for network modeling and optimization in SDN". *IEEE Journal on Selected Areas in Communications*, 38(10), 2260-2270.

**QoT Prediction with ML**:

- Morais, R. M., & Pedro, J. (2018). "Machine learning models for estimating quality of transmission in DWDM networks". *Journal of Optical Communications and Networking*, 10(10), D84-D99.

**Cognitive Optical Networks**:

- Mata, J., et al. (2018). "Artificial intelligence (AI) methods in optical networks: A comprehensive survey". *Optical Switching and Networking*, 28, 43-57.

Books and Surveys
-----------------

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

- Musumeci, F., et al. (2021). "Machine learning for optical networks: A practical perspective". In *Machine Learning for Future Fiber-Optic Communication Systems* (pp. 1-30). Academic Press.

Online Resources
----------------

- StableBaselines3 Documentation: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- PyTorch Geometric (GNN): https://pytorch-geometric.readthedocs.io/

See Also
========

Related FUSION Documentation:

- :doc:`resource_allocation` - Traditional RSA algorithms (context for ML approaches)
- :doc:`flex_grid_networks` - Network fundamentals ML algorithms optimize
- :doc:`../tutorials/getting_started` - Getting started with FUSION
- :doc:`../tutorials/artifical_intelligence` - Tutorial on running ML/RL in FUSION
- :doc:`../api/modules/rl` - RL module API reference
- :doc:`../guides/qot_estimation` - QoT modeling (target for ML prediction)
