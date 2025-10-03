================
Coding Standards
================

Code style and quality standards for FUSION.

.. contents:: Table of Contents
   :local:
   :depth: 2

Style Guide
===========

**Follow PEP 8 with these additions:**

* Line length: 88 characters (Black default)
* Use type hints everywhere
* Docstrings: Google style

Type Annotations
================

**All functions must have type hints:**

.. code-block:: python

   def calculate_path_cost(
       graph: nx.Graph,
       path: list[str],
       weight: str = "length"
   ) -> float:
       """Calculate total cost of a path."""
       return sum(graph[u][v][weight] for u, v in zip(path[:-1], path[1:]))

Documentation
=============

**Docstring format:**

.. code-block:: python

   def function_name(param1: str, param2: int) -> bool:
       """Brief description.
       
       Longer description if needed.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ValueError: When invalid input
       """

Naming Conventions
==================

* **Variables**: ``snake_case``
* **Functions**: ``snake_case``
* **Classes**: ``PascalCase``
* **Constants**: ``UPPER_CASE``
* **Private**: ``_leading_underscore``

Code Quality
============

**Enforce with tools:**

.. code-block:: bash

   make format    # ruff auto-fix
   make lint-new  # ruff + mypy check

See Also
========

* :doc:`testing` - Testing standards
* :doc:`contributing` - How to contribute
