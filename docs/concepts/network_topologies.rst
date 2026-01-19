====================
Network Topologies
====================

Introduction
============

The physical and logical structure of an optical network—its **topology**—profoundly impacts network performance, algorithm behavior, and resource requirements. Understanding network topologies is essential for designing algorithms, interpreting simulation results, and making informed decisions about network planning and operation.

This document covers topology fundamentals, metrics, real-world topologies used in FUSION, how topology affects performance, and how to create and visualize custom topologies.

Topology Fundamentals
=====================

What is Network Topology?
--------------------------

A network topology describes how nodes (cities, data centers, network switches) are interconnected by links (fiber-optic cables).

**Formal Definition**: A network topology is a graph G = (V, E) where:
- V is the set of vertices (nodes)
- E is the set of edges (links)

**Node**: A location in the network where switching/routing occurs
- Cities in national backbone networks
- Data centers
- Points of Presence (PoP)
- Optical cross-connects

**Link**: Physical connection between two nodes
- Fiber-optic cable
- May span hundreds or thousands of kilometers
- Has properties: length (distance), capacity (spectrum slots), number of cores

**Path**: A sequence of nodes and links from source to destination
- Example: A → B → C → D
- Number of hops = number of links in path
- Path length = sum of link distances

Types of Network Topologies
----------------------------

**Physical Topology**
    Actual geographic layout of nodes and fiber cables

    - Determined by geographic constraints, right-of-way, infrastructure costs
    - Cannot be easily changed (fiber installation is expensive)

**Logical Topology**
    How connections (lightpaths) are established over physical infrastructure

    - Can be dynamically reconfigured
    - Software-defined networking enables rapid logical reconfiguration
    - Multiple logical topologies can coexist on same physical topology

Basic Topology Patterns
------------------------

**Bus Topology**
    All nodes connected to a single linear cable

    ::

        A --- B --- C --- D --- E

    - Simple, low cost
    - No redundancy (single point of failure)
    - Rarely used in modern networks

**Ring Topology**
    Nodes connected in a closed loop

    ::

        A --- B
        |     |
        D --- C

    - Bidirectional for protection
    - Common in metropolitan area networks
    - Easy failure recovery (two paths between any nodes)

**Star Topology**
    Central hub connected to all other nodes

    ::

            B
            |
        C - A - D
            |
            E

    - Simple management
    - Central hub is single point of failure
    - Used in access networks

**Mesh Topology**
    Multiple interconnections providing redundant paths

    ::

        A --- B --- C
        |  \  |  /  |
        |   \ | /   |
        D --- E --- F

    - High reliability (multiple paths)
    - Expensive (many links required)
    - Common in backbone networks
    - **Most common in FUSION simulations**

**Hybrid Topology**
    Combination of above patterns

    - Real-world networks are typically hybrid
    - Core: mesh, Access: star, Metro: ring

Topology Metrics
================

Network designers and researchers use various metrics to quantify topology characteristics.

Basic Metrics
-------------

**Number of Nodes (N)**
    Total vertices in the network

    - Typical backbone: 10-50 nodes
    - Example: NSFNet has 14 nodes

**Number of Links (L)**
    Total edges in the network

    - For connected network: L ≥ N - 1 (tree is minimum)
    - Example: NSFNet has 21 bidirectional links

**Average Node Degree**
    Average number of links per node

    .. math::

        \bar{d} = \frac{2L}{N}

    - Higher degree = more redundancy, more routing options
    - NSFNet average degree ≈ 3.0

**Network Diameter**
    Maximum shortest path length (in hops) between any pair of nodes

    .. math::

        D = \max_{i,j \in V} d(i, j)

    where d(i, j) is shortest path length from node i to node j

    - Measures "worst-case" network span
    - Lower diameter = shorter paths = less latency
    - NSFNet diameter = 3 hops

**Average Path Length**
    Average shortest path length across all node pairs

    .. math::

        \bar{L} = \frac{1}{N(N-1)} \sum_{i \neq j} d(i, j)

    - Measures typical path length
    - Impacts resource usage and QoT
    - NSFNet average path length ≈ 2.4 hops

Connectivity Metrics
--------------------

**Node Connectivity**
    Minimum number of nodes that must be removed to disconnect the network

    - Measures network robustness
    - Higher connectivity = more fault-tolerant
    - k-connected network: survives k-1 node failures

**Edge Connectivity**
    Minimum number of links that must be removed to disconnect the network

    - Measures link redundancy
    - Important for survivability planning

**Clustering Coefficient**
    Measure of how nodes tend to cluster together

    .. math::

        C_i = \frac{2T_i}{k_i(k_i-1)}

    where T_i is number of triangles (3-node cycles) involving node i, k_i is degree of node i

    - Higher clustering = more local redundancy
    - Affects routing diversity

**Algebraic Connectivity**
    Second-smallest eigenvalue of the Laplacian matrix

    - Measures how well-connected the network is
    - Higher values = better connectivity
    - Zero value = disconnected network

Resilience Metrics
------------------

**Critical Nodes**
    Nodes whose removal most impacts connectivity

    - Identified by betweenness centrality
    - High betweenness = high traffic load, critical for connectivity

**Critical Links**
    Links whose removal most impacts connectivity

    - Bridge links that connect network regions
    - Often correspond to high-utilization links

**Survivability**
    Ability to maintain service after failures

    - Measured by percentage of node pairs that remain connected after k failures
    - Important for network planning

Performance-Related Metrics
----------------------------

**Total Network Distance**
    Sum of all link lengths

    - Correlates with infrastructure cost
    - Longer total distance = more amplifiers, higher cost

**Average Link Distance**
    Mean physical length of links

    - Impacts modulation format selection
    - Longer links require more robust modulations (lower spectral efficiency)

**Load Distribution**
    How evenly traffic is distributed across links

    - Measured by standard deviation of link utilization
    - More balanced load = better resource utilization, lower congestion

**Spectral Radius**
    Largest eigenvalue of adjacency matrix

    - Related to network diameter and connectivity
    - Affects routing algorithm convergence in distributed protocols

Real-World Topologies in FUSION
================================

FUSION includes several well-known topologies used extensively in research.

NSFNet (National Science Foundation Network)
---------------------------------------------

**Description**: Historical backbone network of the National Science Foundation, widely used in research.

**Characteristics**:
- **Nodes**: 14
- **Links**: 21 (bidirectional)
- **Diameter**: 3 hops
- **Average Degree**: 3.0
- **Geographic Coverage**: United States
- **Typical Total Distance**: ~20,000 km

**Topology**:

::

    WA -------- UT -------- CO -------- NE -------- IL
                             |           |          |
                            CA          MO         MI
                             |           |          |
                            TX --------- GA ------- PA
                                         |          |
                                        DC -------- NY
                                                    |
                                                   RI

**Usage**:
- Most common topology in optical networking research
- Moderate size (tractable for experiments)
- Represents national backbone
- Used in ~80% of EON research papers

**FUSION Configuration**:

.. code-block:: python

    engine_props = {
        'topology_file': 'topologies/NSFNet.json',
    }

**Characteristics Impact**:
- Moderate diameter (3) allows testing long-distance effects
- Average degree (3) provides some routing diversity
- Well-studied, easy to compare results with literature

COST239 (European Optical Network)
-----------------------------------

**Description**: European optical network research topology.

**Characteristics**:
- **Nodes**: 11
- **Links**: 26 (bidirectional)
- **Diameter**: 3 hops
- **Average Degree**: 4.7
- **Geographic Coverage**: Europe
- **Typical Total Distance**: ~7,500 km

**Topology**: Connects major European cities including London, Paris, Berlin, Milan, Madrid, etc.

**Usage**:
- Common in European research
- Higher average degree than NSFNet (more connectivity)
- Shorter average link distances (smaller geographic area)

**FUSION Configuration**:

.. code-block:: python

    engine_props = {
        'topology_file': 'topologies/COST239.json',
    }

**Characteristics Impact**:
- Higher connectivity provides more routing options
- Shorter distances allow higher-order modulations
- Smaller diameter reduces path length

Pan-European Network
--------------------

**Description**: Larger European topology covering more countries.

**Characteristics**:
- **Nodes**: 28
- **Links**: 41 (bidirectional)
- **Diameter**: 5 hops
- **Average Degree**: 2.9
- **Geographic Coverage**: Extended Europe
- **Typical Total Distance**: ~15,000 km

**Usage**:
- Tests scalability of algorithms
- More realistic representation of large regional networks
- Includes longer-distance links

**FUSION Configuration**:

.. code-block:: python

    engine_props = {
        'topology_file': 'topologies/Pan-European.json',
    }

**Characteristics Impact**:
- Larger size tests algorithm scalability
- Lower average degree than COST239 (sparser)
- Longer diameter increases path lengths

USNet (United States Network)
------------------------------

**Description**: Detailed U.S. backbone topology.

**Characteristics**:
- **Nodes**: 24
- **Links**: 43 (bidirectional)
- **Diameter**: 4 hops
- **Average Degree**: 3.6
- **Geographic Coverage**: United States
- **Typical Total Distance**: ~35,000 km

**Usage**:
- More detailed than NSFNet
- Includes more cities and regions
- Realistic representation of national network

**FUSION Configuration**:

.. code-block:: python

    engine_props = {
        'topology_file': 'topologies/USNet.json',
    }

Deutsche Telekom (DT) Network
------------------------------

**Description**: German national network topology.

**Characteristics**:
- **Nodes**: 14
- **Links**: 23 (bidirectional)
- **Diameter**: 3 hops
- **Average Degree**: 3.3
- **Geographic Coverage**: Germany
- **Typical Total Distance**: ~3,000 km

**Usage**:
- Smaller geographic area (compact topology)
- Short link distances
- High-capacity requirements (dense population)

**Characteristics Impact**:
- Very short distances enable high-order modulations (64-QAM, etc.)
- Compact topology reduces path length variation
- High traffic density tests spectrum efficiency

Topology Characteristics and Algorithm Impact
==============================================

How Topology Affects Routing
-----------------------------

**Sparse Topologies (low average degree)**:
- Fewer routing options
- Algorithms find similar paths
- First-fit and best-fit perform similarly
- K-shortest paths has less benefit (paths are similar)

**Dense Topologies (high average degree)**:
- Many routing options
- Path diversity important
- K-shortest paths provides significant benefit
- Adaptive algorithms (congestion-aware, fragmentation-aware) more effective

**Example**:

::

    Sparse Topology (Degree 2-3):
        A --- B --- C --- D
         \               /
          \--- E ------/

        Only 2-3 paths between A and D
        Routing choice limited

    Dense Topology (Degree 4-5):
        A --- B --- C --- D
        |\    |  X  |    /|
        | \   | / \ |   / |
        |  F--G-----H--I  |
        |   \   \  /   /  |
        |    J---K---L    |
        |               /
        M--------------/

        Many paths between A and D
        Routing choice critical

How Topology Affects Spectrum Assignment
-----------------------------------------

**Short Average Path Length**:
- Fewer links per connection
- Less spectrum per connection (continuity constraint easier to satisfy)
- Lower blocking probability

**Long Average Path Length**:
- More links per connection
- More spectrum per connection
- Higher blocking probability
- Spectrum fragmentation more severe

**High Diameter**:
- Some connections traverse many hops
- Long-distance connections require robust modulations (more spectrum)
- Increased heterogeneity in spectrum requirements

**Low Diameter**:
- Most connections are short
- Higher-order modulations possible
- More uniform spectrum requirements

How Topology Affects Physical Layer
------------------------------------

**Link Distance Distribution**:

**Short Links (< 500 km)**:
- High-order modulations feasible (16-QAM, 64-QAM)
- High spectral efficiency
- Lower blocking

**Medium Links (500-1500 km)**:
- Moderate modulations (QPSK, 8-QAM)
- Moderate spectral efficiency

**Long Links (> 1500 km)**:
- Low-order modulations required (BPSK, QPSK)
- Low spectral efficiency
- Higher blocking

**Geographic vs. Logical Distance**:

Real-world topologies have correlation between geographic and graph distance:
- Geographically close nodes often have direct links (low hop count)
- Geographically distant nodes require more hops

This affects QoT estimation and modulation selection.

How Topology Affects Congestion and Load Balancing
---------------------------------------------------

**Centralized Topologies** (hub-and-spoke):
- Central nodes/links become congested
- Critical nodes for network connectivity
- Load balancing difficult

**Distributed Topologies** (mesh):
- Load more evenly distributed
- Many alternate paths
- Better resilience to failures and congestion

**Betweenness Centrality**:

Nodes/links with high betweenness centrality carry more traffic:

.. code-block:: python

    import networkx as nx

    # Calculate betweenness centrality
    topology = nx.read_graphml('NSFNet.graphml')
    node_bc = nx.betweenness_centrality(topology)
    edge_bc = nx.edge_betweenness_centrality(topology)

    # High-betweenness nodes are critical
    critical_nodes = sorted(node_bc.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"Most critical nodes: {critical_nodes}")

    # High-betweenness links are bottlenecks
    critical_links = sorted(edge_bc.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"Bottleneck links: {critical_links}")

Geographic vs. Logical Topology
================================

Geographic Topology
-------------------

**Definition**: Physical layout of nodes and links on a map.

**Considerations**:
- Real-world geography (oceans, mountains, political boundaries)
- Right-of-way and infrastructure availability
- Cost (fiber installation, land acquisition)

**Impact**:
- Determines link distances (affects QoT, modulation)
- Influences network design and planning
- Cannot be easily changed

**Visualization**:

Geographic plots show nodes at actual locations:

.. code-block:: python

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # Plot topology on map
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Draw map features
    ax.coastlines()
    ax.stock_img()

    # Plot nodes at geographic coordinates
    for node, data in topology.nodes(data=True):
        lat, lon = data['latitude'], data['longitude']
        ax.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())

    # Plot links
    for u, v, data in topology.edges(data=True):
        lat1, lon1 = topology.nodes[u]['latitude'], topology.nodes[u]['longitude']
        lat2, lon2 = topology.nodes[v]['latitude'], topology.nodes[v]['longitude']
        ax.plot([lon1, lon2], [lat1, lat2], 'b-', linewidth=2, transform=ccrs.Geodetic())

    plt.title("NSFNet Geographic Topology")
    plt.show()

Logical Topology
----------------

**Definition**: Graph structure of connections, independent of geography.

**Considerations**:
- Routing and spectrum allocation
- Connectivity and redundancy
- Algorithmic properties (diameter, centrality)

**Impact**:
- Determines routing options and path lengths
- Can be dynamically reconfigured (virtual topologies)
- Focus of most algorithm research

**Visualization**:

Logical plots show graph structure using layout algorithms:

.. code-block:: python

    import networkx as nx
    import matplotlib.pyplot as plt

    # Plot logical topology using spring layout
    pos = nx.spring_layout(topology, k=0.5, iterations=50)
    nx.draw_networkx(topology, pos, with_labels=True, node_color='lightblue',
                     node_size=500, font_size=10, font_weight='bold')

    plt.title("NSFNet Logical Topology")
    plt.axis('off')
    plt.show()

Why Both Matter
---------------

**Geographic topology**:
- Determines physical constraints (distance, QoT)
- Affects cost and feasibility
- Fixed infrastructure

**Logical topology**:
- Determines algorithmic behavior
- Affects performance metrics (blocking, utilization)
- Can be optimized through algorithms

**Example**: Two topologies with same logical structure but different geographic layouts:

::

    Logical Structure (same):
        A --- B --- C
        |           |
        D --- E --- F

    Geographic Layout 1 (Compact):
        All links ~100 km
        → Can use high-order modulations
        → Low spectrum per connection
        → Low blocking

    Geographic Layout 2 (Spread):
        Some links ~2000 km
        → Must use low-order modulations
        → High spectrum per connection
        → High blocking

Creating Custom Topologies in FUSION
=====================================

Topology File Format
--------------------

FUSION uses JSON format for topology files:

.. code-block:: json

    {
      "nodes": [
        {
          "id": 0,
          "name": "Seattle",
          "latitude": 47.6062,
          "longitude": -122.3321
        },
        {
          "id": 1,
          "name": "Portland",
          "latitude": 45.5152,
          "longitude": -122.6784
        }
      ],
      "links": [
        {
          "source": 0,
          "target": 1,
          "distance": 280,
          "weight": 1.0
        }
      ]
    }

**Node Properties**:
- ``id``: Unique integer identifier (required)
- ``name``: Human-readable name (optional)
- ``latitude``, ``longitude``: Geographic coordinates (optional, for visualization)

**Link Properties**:
- ``source``: Source node ID (required)
- ``target``: Target node ID (required)
- ``distance``: Physical length in km (required for QoT)
- ``weight``: Routing weight (optional, default 1.0)

Creating a Topology Programmatically
-------------------------------------

.. code-block:: python

    import json
    import networkx as nx

    def create_topology(name, nodes, links):
        """Create a topology JSON file."""
        topology = {
            "name": name,
            "nodes": nodes,
            "links": links
        }

        filename = f"topologies/{name}.json"
        with open(filename, 'w') as f:
            json.dump(topology, f, indent=2)

        print(f"Topology saved to {filename}")

    # Example: Create a simple 4-node ring
    nodes = [
        {"id": 0, "name": "Node A", "latitude": 40.0, "longitude": -120.0},
        {"id": 1, "name": "Node B", "latitude": 41.0, "longitude": -120.0},
        {"id": 2, "name": "Node C", "latitude": 41.0, "longitude": -119.0},
        {"id": 3, "name": "Node D", "latitude": 40.0, "longitude": -119.0},
    ]

    links = [
        {"source": 0, "target": 1, "distance": 100},
        {"source": 1, "target": 2, "distance": 100},
        {"source": 2, "target": 3, "distance": 100},
        {"source": 3, "target": 0, "distance": 100},
    ]

    create_topology("SimpleRing", nodes, links)

Importing from Other Formats
-----------------------------

**From GraphML** (common graph format):

.. code-block:: python

    import networkx as nx
    import json

    # Load GraphML
    G = nx.read_graphml("topology.graphml")

    # Convert to FUSION format
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            "id": int(node_id),
            "name": data.get("label", f"Node {node_id}"),
            "latitude": data.get("latitude", 0.0),
            "longitude": data.get("longitude", 0.0),
        })

    links = []
    for u, v, data in G.edges(data=True):
        links.append({
            "source": int(u),
            "target": int(v),
            "distance": data.get("distance", 100.0),
            "weight": data.get("weight", 1.0),
        })

    topology = {"nodes": nodes, "links": links}
    with open("topology.json", 'w') as f:
        json.dump(topology, f, indent=2)

**From Real Geographic Data**:

.. code-block:: python

    from geopy.distance import geodesic

    def calculate_distance(lat1, lon1, lat2, lon2):
        """Calculate geographic distance between two points."""
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers

    # Add calculated distances to links
    for link in links:
        src_node = nodes[link["source"]]
        tgt_node = nodes[link["target"]]

        distance = calculate_distance(
            src_node["latitude"], src_node["longitude"],
            tgt_node["latitude"], tgt_node["longitude"]
        )
        link["distance"] = round(distance, 2)

Generating Synthetic Topologies
--------------------------------

For research, sometimes synthetic topologies are useful:

**Random Topology**:

.. code-block:: python

    import networkx as nx
    import random

    def generate_random_topology(num_nodes, edge_probability=0.3):
        """Generate random Erdős-Rényi topology."""
        G = nx.erdos_renyi_graph(num_nodes, edge_probability)

        # Ensure connected
        if not nx.is_connected(G):
            # Add edges to connect components
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i+1]))
                G.add_edge(node1, node2)

        # Assign properties
        nodes = []
        for node in G.nodes():
            nodes.append({
                "id": node,
                "name": f"Node {node}",
            })

        links = []
        for u, v in G.edges():
            links.append({
                "source": u,
                "target": v,
                "distance": random.randint(100, 1000),  # Random distance
            })

        return {"nodes": nodes, "links": links}

**Small-World Topology**:

.. code-block:: python

    def generate_small_world_topology(num_nodes, k=4, p=0.3):
        """Generate Watts-Strogatz small-world topology."""
        G = nx.watts_strogatz_graph(num_nodes, k, p)

        # Convert to FUSION format
        # ... (similar to above)

**Scale-Free Topology** (Barabási-Albert):

.. code-block:: python

    def generate_scale_free_topology(num_nodes, m=2):
        """Generate Barabási-Albert scale-free topology."""
        G = nx.barabasi_albert_graph(num_nodes, m)

        # Convert to FUSION format
        # ... (similar to above)

Validating Topologies
----------------------

Before using a custom topology, validate it:

.. code-block:: python

    import networkx as nx

    def validate_topology(topology_file):
        """Validate topology file."""
        with open(topology_file) as f:
            topo_data = json.load(f)

        # Build NetworkX graph
        G = nx.Graph()
        for node in topo_data["nodes"]:
            G.add_node(node["id"])

        for link in topo_data["links"]:
            G.add_edge(link["source"], link["target"], distance=link["distance"])

        # Check connectivity
        if not nx.is_connected(G):
            print("ERROR: Topology is not connected!")
            return False

        # Check for self-loops
        if list(nx.selfloop_edges(G)):
            print("WARNING: Topology contains self-loops")

        # Check for duplicate edges
        if G.number_of_edges() != len(topo_data["links"]):
            print("WARNING: Duplicate edges detected")

        # Calculate metrics
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Links: {G.number_of_edges()}")
        print(f"Diameter: {nx.diameter(G)}")
        print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print(f"Average path length: {nx.average_shortest_path_length(G):.2f}")

        return True

    # Validate before using
    if validate_topology("topologies/MyTopology.json"):
        print("Topology is valid!")

Visualization of Topologies
============================

FUSION GUI Visualization
-------------------------

FUSION includes a built-in topology viewer:

.. code-block:: python

    from fusion.gui import TopologyViewer

    # Launch topology viewer
    viewer = TopologyViewer(topology_file='topologies/NSFNet.json')
    viewer.show()

**Features**:
- Interactive node selection
- Link highlighting
- Spectrum utilization overlay
- Path visualization
- Real-time network state

Matplotlib Visualization
-------------------------

For static plots in papers/reports:

.. code-block:: python

    import networkx as nx
    import matplotlib.pyplot as plt

    # Load topology
    G = nx.read_json("topologies/NSFNet.json")

    # Layout algorithms
    pos_spring = nx.spring_layout(G)  # Force-directed
    pos_circular = nx.circular_layout(G)  # Circular
    pos_kamada = nx.kamada_kawai_layout(G)  # Kamada-Kawai

    # Draw topology
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    nx.draw_networkx(G, pos_spring, ax=axes[0], node_color='lightblue',
                     node_size=500, with_labels=True)
    axes[0].set_title("Spring Layout")
    axes[0].axis('off')

    nx.draw_networkx(G, pos_circular, ax=axes[1], node_color='lightgreen',
                     node_size=500, with_labels=True)
    axes[1].set_title("Circular Layout")
    axes[1].axis('off')

    nx.draw_networkx(G, pos_kamada, ax=axes[2], node_color='lightcoral',
                     node_size=500, with_labels=True)
    axes[2].set_title("Kamada-Kawai Layout")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("topology_layouts.png", dpi=300)
    plt.show()

Geographic Visualization
-------------------------

Overlay topology on a map:

.. code-block:: python

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Create figure with map projection
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':')

    # Plot nodes
    for node, data in topology.nodes(data=True):
        lat, lon = data['latitude'], data['longitude']
        ax.plot(lon, lat, 'ro', markersize=10, markeredgecolor='darkred',
                markeredgewidth=2, transform=ccrs.PlateCarree(), zorder=5)
        ax.text(lon, lat+0.5, data.get('name', ''), fontsize=8,
                ha='center', transform=ccrs.PlateCarree())

    # Plot links
    for u, v in topology.edges():
        lat1, lon1 = topology.nodes[u]['latitude'], topology.nodes[u]['longitude']
        lat2, lon2 = topology.nodes[v]['latitude'], topology.nodes[v]['longitude']
        ax.plot([lon1, lon2], [lat1, lat2], 'b-', linewidth=2, alpha=0.7,
                transform=ccrs.Geodetic(), zorder=3)

    # Set extent (adjust for your topology)
    ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())  # USA

    plt.title("NSFNet Topology (Geographic)", fontsize=16)
    plt.savefig("nsfnet_geographic.png", dpi=300, bbox_inches='tight')
    plt.show()

Interactive Visualization
--------------------------

For exploration and presentations:

.. code-block:: python

    import plotly.graph_objects as go

    # Create interactive network plot
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=20,
            color=[],
            colorscale='YlGnBu',
            line_width=2))

    # Add edges
    for u, v in topology.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Add nodes
    for node in topology.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Interactive Network Topology',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    fig.write_html("topology_interactive.html")
    fig.show()

How Topology Affects Performance
=================================

Case Study: Comparing Topologies
---------------------------------

Let's compare algorithm performance on different topologies:

.. code-block:: python

    from fusion import Fusion
    import pandas as pd

    topologies = ['NSFNet', 'COST239', 'Pan-European', 'USNet']
    algorithms = ['first_fit', 'best_fit']

    results = []

    for topo in topologies:
        for alg in algorithms:
            engine_props = {
                'topology_file': f'topologies/{topo}.json',
                'spectrum_algorithm': alg,
                'slots_per_link': 320,
                'traffic_load': 150,
            }

            fusion = Fusion(engine_props)
            stats = fusion.run_simulation(num_requests=5000)

            results.append({
                'topology': topo,
                'algorithm': alg,
                'blocking': stats['blocking_probability'],
                'utilization': stats['spectrum_utilization'],
                'avg_path_length': stats['avg_path_length'],
            })

    # Analyze results
    df = pd.DataFrame(results)
    print(df.pivot_table(index='topology', columns='algorithm', values='blocking'))

**Expected Findings**:

- **NSFNet**: Moderate blocking, balanced performance
- **COST239**: Lower blocking (higher connectivity)
- **Pan-European**: Higher blocking (sparser, larger)
- **USNet**: Varies by traffic pattern (more routing options)

Impact of Topology Size
------------------------

.. code-block:: python

    # Compare performance vs. topology size
    sizes = [10, 15, 20, 25, 30]  # Number of nodes

    for size in sizes:
        # Generate random topology of given size
        topology = generate_random_topology(num_nodes=size, edge_probability=0.3)

        # Run simulation
        stats = run_fusion_simulation(topology)

        print(f"Size {size}: Blocking = {stats['blocking']:.2%}")

**Typical Trend**: Blocking increases with network size (more traffic, longer paths)

Impact of Average Degree
-------------------------

.. code-block:: python

    # Compare performance vs. connectivity
    degrees = [2.5, 3.0, 3.5, 4.0, 4.5]

    for degree in degrees:
        # Generate topology with target average degree
        edge_prob = degree / (num_nodes - 1)
        topology = generate_random_topology(num_nodes=20, edge_probability=edge_prob)

        # Run simulation
        stats = run_fusion_simulation(topology)

        print(f"Degree {degree}: Blocking = {stats['blocking']:.2%}")

**Typical Trend**: Blocking decreases with higher degree (more routing options)

Best Practices
==============

Choosing a Topology for Research
---------------------------------

**For Algorithm Comparison**:
- Use standard topologies (NSFNet, COST239) for comparability
- Report results on multiple topologies to show generalization

**For Scalability Testing**:
- Use range of sizes (small to large)
- Test on both sparse and dense topologies

**For Real-World Relevance**:
- Use realistic topologies (NSFNet, Pan-European)
- Include geographic constraints (link distances)

**For Worst-Case Analysis**:
- Use sparse topologies (low average degree)
- Use high-diameter topologies (long paths)

Topology Design Guidelines
---------------------------

**Connectivity**:
- Ensure network is connected (validate!)
- Aim for average degree ≥ 3 for redundancy
- Consider 2-connectivity for resilience

**Distance**:
- Include mix of short and long links
- Consider realistic geographic constraints
- Match distances to target network type (metro vs. backbone)

**Balance**:
- Avoid extreme hub-and-spoke patterns (single point of failure)
- Distribute connectivity across nodes

**Scalability**:
- For research, test on multiple sizes
- 10-30 nodes typical for simulation studies

Further Reading
===============

Papers on Network Topologies
-----------------------------

**Topology Analysis**:

- Knight, S., et al. (2011). "The internet topology zoo". *IEEE Journal on Selected Areas in Communications*, 29(9), 1765-1775.

**Topology Impact on Optical Networks**:

- Zhu, H., et al. (2003). "A minimal-cost path algorithm for elastic optical networks". *IEEE/ACM Transactions on Networking*, 11(1), 1-14.

**Network Design**:

- Orlowski, S., et al. (2010). "SNDlib 1.0—Survivable network design library". *Networks*, 55(3), 276-286.

Resources
---------

**Topology Datasets**:

- Internet Topology Zoo: http://www.topology-zoo.org/
- SNDlib: http://sndlib.zib.de/
- Optical network topology databases (research repositories)

**Tools**:

- NetworkX: https://networkx.org/
- Graph-tool: https://graph-tool.skewed.de/
- igraph: https://igraph.org/

See Also
========

Related FUSION Documentation:

- :doc:`resource_allocation` - How routing algorithms use topology
- :doc:`flex_grid_networks` - Network fundamentals
- :doc:`optical_networking_basics` - Physical layer constraints
- :doc:`../tutorials/getting_started` - Using topologies in FUSION
- :doc:`../api/core/network` - Network API and topology loading
- :doc:`../guides/visualization` - Advanced visualization techniques
