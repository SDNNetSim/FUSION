==================
Extending FUSION
==================

Add custom algorithms and extend FUSION functionality.

.. contents:: Table of Contents
   :local:
   :depth: 2

Adding Custom Routing
=====================

**1. Create your algorithm:**

.. code-block:: python

   from fusion.interfaces.router import RouterInterface
   from fusion.modules.routing.registry import register_router

   @register_router("my_routing")
   class MyRouter(RouterInterface):
       def calculate_paths(self, graph, src, dst, k=3):
           # Your routing logic
           paths = find_my_paths(graph, src, dst, k)
           return paths

**2. Use in configuration:**

.. code-block:: ini

   [general_settings]
   route_method = my_routing
   k_paths = 5

Adding Custom Spectrum Assignment
==================================

.. code-block:: python

   from fusion.interfaces.spectrum import SpectrumInterface
   from fusion.modules.spectrum.registry import register_spectrum

   @register_spectrum("my_spectrum")
   class MySpectrum(SpectrumInterface):
       def assign_spectrum(self, path, num_slots, spectrum_state):
           # Your assignment logic
           return start_slot, end_slot

Adding RL Agents
================

.. code-block:: python

   from fusion.interfaces.agent import AgentInterface
   from fusion.modules.rl.registry import register_agent

   @register_agent("my_agent")
   class MyAgent(AgentInterface):
       def select_action(self, observation):
           # Your RL policy
           return action

Testing Extensions
==================

.. code-block:: python

   def test_my_routing():
       router = MyRouter()
       paths = router.calculate_paths(graph, src=0, dst=5, k=3)
       assert len(paths) == 3

See Also
========

* :doc:`architecture` - System architecture
* :doc:`../api/interfaces` - Interface reference
* :doc:`contributing` - Contribution guidelines
