==================
Custom Topology
==================

Learn how to create custom network topologies for your simulations, from simple
test networks to complex real-world designs.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSION supports custom network topologies through:

* **JSON format**: Define nodes, links, and fiber properties
* **NetworkX integration**: Programmatically create topologies
* **Python API**: Full control over network structure

This guide shows all three approaches.

Method 1: JSON Topology File
==============================

JSON format is the simplest way to define custom topologies.

Basic Triangle Topology
-----------------------

Create ``my_topology.json``:

.. code-block:: json

   {
     "nodes": {
       "A": {"type": "CDC"},
       "B": {"type": "CDC"},
       "C": {"type": "CDC"}
     },
     "links": {
       "1": {
         "source": "A",
         "destination": "B",
         "length": 100,
         "span_length": 100,
         "fiber": {
           "num_cores": 1,
           "attenuation": 4.6e-05,
           "non_linearity": 0.0013,
           "dispersion": 2.04e-26,
           "fiber_type": 0,
           "bending_radius": 0.05,
           "mode_coupling_co": 0.0004,
           "propagation_const": 4000000.0,
           "core_pitch": 4e-05
         }
       },
       "2": {
         "source": "B",
         "destination": "C",
         "length": 150,
         "span_length": 100,
         "fiber": {
           "num_cores": 1,
           "attenuation": 4.6e-05,
           "non_linearity": 0.0013,
           "dispersion": 2.04e-26,
           "fiber_type": 0,
           "bending_radius": 0.05,
           "mode_coupling_co": 0.0004,
           "propagation_const": 4000000.0,
           "core_pitch": 4e-05
         }
       },
       "3": {
         "source": "C",
         "destination": "A",
         "length": 200,
         "span_length": 100,
         "fiber": {
           "num_cores": 1,
           "attenuation": 4.6e-05,
           "non_linearity": 0.0013,
           "dispersion": 2.04e-26,
           "fiber_type": 0,
           "bending_radius": 0.05,
           "mode_coupling_co": 0.0004,
           "propagation_const": 4000000.0,
           "core_pitch": 4e-05
         }
       }
     }
   }

**Use in configuration:**

.. code-block:: ini

   [topology_settings]
   network = my_topology.json
   topology_type = custom

JSON Field Reference
--------------------

**Node Fields:**

* ``type``: Node type - ``"CDC"`` (Colorless-Directionless-Contentionless ROADM)

**Link Fields:**

* ``source``: Source node name
* ``destination``: Destination node name
* ``length``: Link length in km
* ``span_length``: Amplifier spacing in km

**Fiber Fields:**

* ``num_cores``: Number of fiber cores (1 for single-core, 7+ for multicore)
* ``attenuation``: Fiber attenuation (α) in 1/m (typical: 4.6e-05 for 0.2 dB/km)
* ``non_linearity``: Nonlinear coefficient (γ) in 1/(W·m)
* ``dispersion``: Dispersion coefficient (D) in s/m²
* ``fiber_type``: Fiber type identifier (0 = standard SMF)
* ``bending_radius``: Bending radius in m (multicore only)
* ``mode_coupling_co``: Mode coupling coefficient (multicore only)
* ``propagation_const``: Propagation constant (multicore only)
* ``core_pitch``: Core spacing in m (multicore only)

Simplified Format
-----------------

For basic simulations, you can omit detailed fiber properties:

.. code-block:: json

   {
     "nodes": {
       "Node1": {"type": "CDC"},
       "Node2": {"type": "CDC"},
       "Node3": {"type": "CDC"}
     },
     "links": {
       "1": {
         "source": "Node1",
         "destination": "Node2",
         "length": 100
       },
       "2": {
         "source": "Node2",
         "destination": "Node3",
         "length": 150
       }
     }
   }

FUSION will use default fiber parameters.

Method 2: NetworkX (Python API)
================================

For programmatically generated topologies, use NetworkX.

Simple Ring Network
-------------------

.. code-block:: python

   import networkx as nx
   from fusion.configs.config import Config
   from fusion.sim.run_simulation import run_simulation

   # Create a 5-node ring topology
   G = nx.cycle_graph(5)

   # Add link lengths as edge weights
   for u, v in G.edges():
       G[u][v]['length'] = 100  # km

   # Configure simulation
   config = Config(
       topology_settings={
           'network': G,  # Pass NetworkX graph directly
           'topology_type': 'custom'
       },
       general_settings={
           'erlang_start': 300,
           'erlang_stop': 900,
           'erlang_step': 300,
           'num_requests': 500,
           'max_iters': 4,
           'k_paths': 3,
           'route_method': 'k_shortest_path',
           'allocation_method': 'first_fit'
       },
       spectrum_settings={
           'c_band': 320
       }
   )

   # Run simulation
   results = run_simulation(config)

Random Topology Generator
--------------------------

.. code-block:: python

   import networkx as nx
   import random

   def create_random_topology(num_nodes: int, edge_probability: float = 0.3):
       """Create a random connected topology."""
       # Ensure graph is connected
       while True:
           G = nx.erdos_renyi_graph(num_nodes, edge_probability)
           if nx.is_connected(G):
               break

       # Assign realistic link lengths (50-500 km)
       for u, v in G.edges():
           G[u][v]['length'] = random.randint(50, 500)

       return G

   # Generate 10-node random network
   G = create_random_topology(10)

Grid Topology
-------------

.. code-block:: python

   import networkx as nx

   # Create a 3x3 grid network
   G = nx.grid_2d_graph(3, 3)

   # Convert node labels to strings
   mapping = {node: f"Node_{i}" for i, node in enumerate(G.nodes())}
   G = nx.relabel_nodes(G, mapping)

   # Set uniform link lengths
   for u, v in G.edges():
       G[u][v]['length'] = 75  # km

Real-World Topology Example
----------------------------

Recreate a simplified metro network:

.. code-block:: python

   import networkx as nx

   # Metro area network (San Francisco Bay Area example)
   G = nx.Graph()

   # Add nodes (cities)
   nodes = [
       'San Francisco', 'Oakland', 'San Jose',
       'Palo Alto', 'Fremont', 'Berkeley'
   ]
   G.add_nodes_from(nodes)

   # Add links with approximate distances
   links = [
       ('San Francisco', 'Oakland', 20),
       ('San Francisco', 'Palo Alto', 50),
       ('Oakland', 'Berkeley', 15),
       ('Oakland', 'Fremont', 40),
       ('Palo Alto', 'San Jose', 30),
       ('Fremont', 'San Jose', 25),
       ('Berkeley', 'Palo Alto', 55)
   ]

   for src, dst, length in links:
       G.add_edge(src, dst, length=length)

Method 3: Programmatic JSON Generation
=======================================

Generate JSON topology files programmatically:

.. code-block:: python

   import json

   def create_linear_topology(num_nodes: int,
                             link_length: int = 100,
                             output_file: str = "linear_topology.json"):
       """Create a linear (chain) topology."""
       topology = {
           "nodes": {},
           "links": {}
       }

       # Add nodes
       for i in range(num_nodes):
           node_name = f"Node{i}"
           topology["nodes"][node_name] = {"type": "CDC"}

       # Add links
       link_id = 1
       for i in range(num_nodes - 1):
           topology["links"][str(link_id)] = {
               "source": f"Node{i}",
               "destination": f"Node{i+1}",
               "length": link_length,
               "span_length": 100,
               "fiber": {
                   "num_cores": 1,
                   "attenuation": 4.6e-05,
                   "non_linearity": 0.0013,
                   "dispersion": 2.04e-26
               }
           }
           link_id += 1

       # Save to file
       with open(output_file, 'w') as f:
           json.dump(topology, f, indent=2)

       print(f"Created {output_file} with {num_nodes} nodes")

   # Generate 10-node linear network
   create_linear_topology(10)

Topology Validation
===================

Validate your custom topology before running simulations:

.. code-block:: python

   import networkx as nx
   import json

   def validate_topology(topology_file: str):
       """Validate a JSON topology file."""
       with open(topology_file) as f:
           data = json.load(f)

       # Build NetworkX graph
       G = nx.Graph()
       for node in data["nodes"]:
           G.add_node(node)

       for link_id, link in data["links"].items():
           G.add_edge(link["source"], link["destination"],
                     length=link.get("length", 100))

       # Validation checks
       print(f"Nodes: {G.number_of_nodes()}")
       print(f"Links: {G.number_of_edges()}")
       print(f"Connected: {nx.is_connected(G)}")
       print(f"Diameter: {nx.diameter(G) if nx.is_connected(G) else 'N/A'}")
       print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

       # Warn about issues
       if not nx.is_connected(G):
           print("WARNING: Graph is not connected!")
           components = list(nx.connected_components(G))
           print(f"  Found {len(components)} components: {components}")

       # Check for self-loops
       if nx.number_of_selfloops(G) > 0:
           print("WARNING: Graph contains self-loops!")

       return G

   # Validate your topology
   G = validate_topology("my_topology.json")

Common Topology Patterns
=========================

Star Topology
-------------

Central hub connected to all other nodes:

.. code-block:: python

   import networkx as nx

   G = nx.star_graph(5)  # 1 hub + 5 nodes
   for u, v in G.edges():
       G[u][v]['length'] = 100

Full Mesh
---------

Every node connected to every other node:

.. code-block:: python

   import networkx as nx

   G = nx.complete_graph(6)  # 6 fully connected nodes
   for u, v in G.edges():
       G[u][v]['length'] = 150

Ring Topology
-------------

.. code-block:: python

   import networkx as nx

   G = nx.cycle_graph(8)  # 8-node ring
   for u, v in G.edges():
       G[u][v]['length'] = 100

Small-World Network
-------------------

.. code-block:: python

   import networkx as nx

   # Watts-Strogatz small-world
   G = nx.watts_strogatz_graph(20, 4, 0.3)
   for u, v in G.edges():
       G[u][v]['length'] = random.randint(50, 200)

Using Custom Topologies
=======================

Method 1: Configuration File
-----------------------------

.. code-block:: ini

   [topology_settings]
   network = path/to/my_topology.json
   topology_type = custom

Method 2: Python API
--------------------

.. code-block:: python

   from fusion.configs.config import Config
   from fusion.sim.run_simulation import run_simulation

   config = Config.from_file("base_config.ini")
   config.topology_settings.network = "my_topology.json"
   results = run_simulation(config)

Method 3: NetworkX Directly
----------------------------

.. code-block:: python

   import networkx as nx
   from fusion.configs.config import Config
   from fusion.sim.run_simulation import run_simulation

   G = nx.your_custom_graph()
   config = Config.from_file("base_config.ini")
   config.topology_settings.network = G
   results = run_simulation(config)

Troubleshooting
===============

Graph Not Connected
-------------------

If you get "Graph is not connected" errors:

.. code-block:: python

   import networkx as nx

   # Check connectivity
   if not nx.is_connected(G):
       # Find isolated components
       components = list(nx.connected_components(G))
       print(f"Found {len(components)} components")

       # Connect components
       for i in range(len(components) - 1):
           node_a = list(components[i])[0]
           node_b = list(components[i+1])[0]
           G.add_edge(node_a, node_b, length=100)

Invalid Node Names
------------------

Ensure node names are strings:

.. code-block:: python

   # BAD: Integer node names
   G.add_edge(0, 1)

   # GOOD: String node names
   G.add_edge("Node0", "Node1")

Missing Link Lengths
--------------------

All links need a length attribute:

.. code-block:: python

   for u, v in G.edges():
       if 'length' not in G[u][v]:
           G[u][v]['length'] = 100  # Default 100 km

Next Steps
==========

* :doc:`basic_simulation` - Run simulations with your custom topology
* :doc:`batch_simulations` - Test multiple topology variations
* :doc:`../concepts/network_topologies` - Learn about topology design principles
* :doc:`../api/utils` - Network utility functions reference

See Also
========

* `NetworkX Documentation <https://networkx.org/documentation/stable/>`_
* :doc:`../concepts/optical_networking_basics` - Network fundamentals
* :doc:`../user_guide/configuration_reference` - Topology configuration options
