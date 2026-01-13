======
FUSION
======

**Flexible Unified System for Intelligent Optical Networking**

FUSION is an open-source simulation framework for Software Defined Elastic Optical Networks (SD-EONs).

.. note::

   This is a placeholder page to preview the Furo theme.

----

Headings
========

This is an H2 heading (equals signs).

Subheading
----------

This is an H3 heading (dashes).

Sub-subheading
~~~~~~~~~~~~~~

This is an H4 heading (tildes).

----

Text Formatting
===============

This is regular paragraph text. It can contain **bold text**, *italic text*,
and ``inline code``. Here is a `hyperlink <https://github.com/SDNNetSim/FUSION>`_.

----

Code Blocks
===========

Python example:

.. code-block:: python

   from fusion import Simulation

   sim = Simulation(topology="nsfnet")
   sim.run(arrivals=1000)

   # Print results
   print(f"Blocking: {sim.blocking_probability:.4f}")

Bash example:

.. code-block:: bash

   pip install fusion-sim
   fusion run --topology nsfnet --arrivals 1000

----

Admonitions
===========

.. note::

   This is a note admonition. Use it for additional information.

.. warning::

   This is a warning admonition. Use it for important cautions.

.. tip::

   This is a tip admonition. Use it for helpful suggestions.

.. important::

   This is an important admonition. Use it for critical information.

.. seealso::

   This is a see-also admonition for cross-references.

----

Lists
=====

Bullet list:

- First item
- Second item
- Third item with nested:

  - Nested item A
  - Nested item B

Numbered list:

1. First step
2. Second step
3. Third step

Definition list:

Term One
   Definition for term one.

Term Two
   Definition for term two with more detail.

----

Tables
======

Simple table:

======== =========== ============
Topology Nodes       Links
======== =========== ============
NSFNet   14          21
COST239  11          26
USNet    24          43
======== =========== ============

----

Blockquote
==========

    "FUSION provides a flexible framework for simulating elastic optical networks
    with integrated AI/ML capabilities."

    -- Project Description

----

Navigation
==========

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting-started/claude-code/index
