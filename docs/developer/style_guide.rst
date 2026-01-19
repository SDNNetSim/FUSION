=======================
Documentation Style Guide
=======================

This guide ensures consistency across FUSION documentation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Terminology Standards
=====================

Use these standardized terms consistently:

Core Technology Terms
---------------------

**Correct Capitalization:**

* **FUSION** - Always ALL CAPS (acronym for Flexible Unified System for Intelligent Optical Networking)
* **SD-EON** - Software Defined Elastic Optical Network
* **EON** - Elastic Optical Network
* **WDM** - Wavelength Division Multiplexing
* **SDN** - Software Defined Networking
* **RSA** - Routing and Spectrum Assignment
* **RMCSA** - Routing, Modulation, Core, and Spectrum Assignment
* **SNR** - Signal-to-Noise Ratio
* **QoT** - Quality of Transmission
* **RL** - Reinforcement Learning (when referring to the field)
* **ML** - Machine Learning (when referring to the field)
* **PyTorch Geometric** - Capitalized (official name)
* **Stable-Baselines3** - Hyphenated, capitalized
* **Gymnasium** - Capitalized (replaces OpenAI Gym)

**Algorithm Names:**

* k-shortest paths (lowercase k, hyphenated)
* First-Fit (capitalized both words)
* Best-Fit (capitalized both words)
* Dijkstra (capitalized)
* A* (A-star)
* DQN (Deep Q-Network)
* PPO (Proximal Policy Optimization)
* A2C (Advantage Actor-Critic)

**Network Components:**

* spectrum slots (lowercase)
* frequency slots (lowercase)
* flex-grid (hyphenated, lowercase)
* fixed-grid (hyphenated, lowercase)
* C-band (capitalized C)
* lightpath (one word, lowercase)
* super-channel (hyphenated, lowercase)

Consistent Phrasing
-------------------

**Use consistently:**

* "connection request" (not "demand" or "call")
* "traffic load" (not "network load" or "arrival rate" alone)
* "blocking probability" (not "blocking rate")
* "network topology" (not just "topology" or "network")
* "simulation run" (not "iteration" or "trial" - unless specifically referring to RL trials)
* "configuration file" (not "config file" in formal docs, though "config" is OK in code examples)

Code and File References
=========================

Python Code
-----------

**Module/Package Names:**

.. code-block:: rst

   # Use code formatting for module references
   :mod:`fusion.core.engine`
   :class:`fusion.core.sdn_controller.SDNController`
   :func:`fusion.modules.routing.k_shortest_paths.calculate_paths`

**Inline Code:**

Use double backticks for inline code:

.. code-block:: rst

   The ``route_method`` parameter determines which algorithm is used.
   Set ``num_spectrum_slots = 320`` in your configuration.

File Paths
----------

**Configuration Files:**

.. code-block:: rst

   Edit ``ini/run_ini/config.ini`` to configure your simulation.
   Example configurations are in ``ini/example_ini/``.

**Data Files:**

.. code-block:: rst

   Results are saved to ``data/output/results.csv``.
   Topology files are located in ``data/topologies/``.

Command-Line Examples
---------------------

Always use bash code blocks with $ prompt:

.. code-block:: rst

   .. code-block:: bash

      $ fusion-sim --network NSFNet --erlang 400
      $ python -m fusion.cli.main --help

Writing Style
=============

Tone and Voice
--------------

* **Active voice**: "FUSION simulates optical networks" (not "Optical networks are simulated by FUSION")
* **Direct**: "Run this command" (not "You can run this command")
* **Present tense**: "The engine processes requests" (not "will process")
* **Concise**: Remove unnecessary words

Audience Levels
---------------

**Getting Started**: Assume minimal background, explain terms
**User Guide**: Assume basic understanding, focus on "how"
**API Reference**: Technical, assume developer knowledge
**Concepts**: Educational, thorough explanations
**Developer Guide**: Assume programming experience

Headings
========

Hierarchy
---------

Use this heading structure consistently:

.. code-block:: rst

   ==================
   Document Title (H1)
   ==================

   Section (H2)
   ============

   Subsection (H3)
   ---------------

   Sub-subsection (H4)
   ^^^^^^^^^^^^^^^^^^^

   Paragraph Title (H5)
   """"""""""""""""""""

Heading Style
-------------

* **H1**: Document title only, one per file
* **H2**: Major sections
* **H3**: Subsections within a major section
* **H4**: Sub-subsections (use sparingly)
* **H5**: Rarely used, only for very detailed breakdowns

**Capitalization**: Use title case for H1 and H2, sentence case for H3-H5

.. code-block:: rst

   ==================
   Installation Guide  # H1 - Title Case
   ==================

   System Requirements  # H2 - Title Case
   ===================

   Python version requirements  # H3 - Sentence case
   ---------------------------

Lists and Admonitions
=====================

Bullet Lists
------------

* Use ``*`` for unordered lists
* Use consistent indentation (3 spaces)
* Keep items parallel in structure

Numbered Lists
--------------

Use numbers for sequential steps:

.. code-block:: rst

   1. Clone the repository
   2. Create virtual environment
   3. Install dependencies

Admonitions
-----------

Use semantic admonitions appropriately:

.. code-block:: rst

   .. note::
      Additional information that helps understanding

   .. tip::
      Helpful suggestions or best practices

   .. warning::
      Important warnings about potential issues

   .. important::
      Critical information users must know

   .. seealso::
      Links to related content

   .. versionadded:: 6.0
      New feature descriptions

   .. deprecated:: 6.0
      Deprecated feature warnings

Code Examples
=============

Format
------

Always include:

1. **Language specification**: ``.. code-block:: python``
2. **Comments**: Explain non-obvious code
3. **Complete examples**: Runnable when possible

Python
------

.. code-block:: rst

   .. code-block:: python

      from fusion.core.engine import SimulationEngine

      # Initialize the simulation engine
      engine = SimulationEngine(config)
      results = engine.run()

Configuration (INI)
-------------------

.. code-block:: rst

   .. code-block:: ini

      [network_settings]
      topology_name = NSFNet
      num_spectrum_slots = 320

Bash/Shell
----------

.. code-block:: rst

   .. code-block:: bash

      # Install FUSION
      pip install -e .

      # Run simulation
      fusion-sim --network NSFNet

Links and References
====================

Internal Links
--------------

Use ``:doc:`` for documentation links:

.. code-block:: rst

   See :doc:`installation` for setup instructions.
   Refer to :doc:`../concepts/optical_networking_basics`.

External Links
--------------

Use descriptive link text, not URLs:

.. code-block:: rst

   ❌ Bad: See https://github.com/SDNNetSim/FUSION
   ✅ Good: See the `FUSION GitHub repository <https://github.com/SDNNetSim/FUSION>`_

Cross-References
----------------

.. code-block:: rst

   See :ref:`manual-installation` for detailed steps.

   # Define anchor:
   .. _manual-installation:

Images and Diagrams
===================

Image Directive
---------------

Always include:

1. **Alt text**: Describe image for accessibility
2. **Alignment**: Center for diagrams
3. **Width**: Percentage for responsive design

.. code-block:: rst

   .. image:: /_static/architecture_diagram.svg
      :alt: FUSION system architecture showing five layers: CLI/Config, Simulation Orchestration, Core Engine, Interfaces, and Modules
      :align: center
      :width: 90%

Figure Directive
----------------

Use for images with captions:

.. code-block:: rst

   .. figure:: /_static/simulation_workflow.svg
      :alt: Flowchart showing simulation workflow from initialization to completion
      :align: center
      :width: 80%

      Figure 1: Complete simulation workflow process

Alt Text Guidelines
-------------------

* **Describe content**: What does the image show?
* **Be concise**: 1-2 sentences maximum
* **Include context**: Why is this image relevant?
* **Skip decorative**: Use empty alt="" for purely decorative images

Tables
======

Grid Tables
-----------

For complex tables:

.. code-block:: rst

   +----------------+-------------+-------------+
   | Parameter      | Default     | Range       |
   +================+=============+=============+
   | erlang_start   | 100         | 1-1000      |
   +----------------+-------------+-------------+

Simple Tables
-------------

For basic tables:

.. code-block:: rst

   =========  ========  ========
   Algorithm  Accuracy  Speed
   =========  ========  ========
   First-Fit  Good      Fast
   Best-Fit   Better    Slower
   =========  ========  ========

Common Mistakes
===============

**Don't:**

* ❌ Use "click here" as link text
* ❌ Reference line numbers in code (code changes!)
* ❌ Use absolute paths (use relative paths)
* ❌ Mix terminology (pick one term and stick with it)
* ❌ Use future tense ("this will do..." → "this does...")
* ❌ Over-use bold/italics (use for emphasis only)

**Do:**

* ✅ Use descriptive link text
* ✅ Reference functions/classes with proper roles
* ✅ Keep sentences under 20 words when possible
* ✅ Use consistent terminology
* ✅ Write in present tense
* ✅ Use formatting purposefully

Version Numbers
===============

Format
------

* Use semantic versioning: ``6.0.0``
* Include version in file headers when relevant
* Use ``versionadded``, ``versionchanged``, ``deprecated`` directives

.. code-block:: rst

   .. versionadded:: 6.0
      Support for multi-core fiber simulation

   .. versionchanged:: 6.0
      Changed default spectrum assignment from Random-Fit to First-Fit

   .. deprecated:: 5.0
      The ``old_function()`` is deprecated. Use :func:`new_function` instead.

Accessibility Checklist
=======================

Before publishing documentation:

* ☐ All images have descriptive alt text
* ☐ Headings follow proper hierarchy (no skipped levels)
* ☐ Links use descriptive text (not "click here")
* ☐ Tables have header rows
* ☐ Code examples have language specified
* ☐ Colors are not the only way to convey information
* ☐ Acronyms are defined on first use
* ☐ Math equations include text descriptions

SEO Best Practices
==================

Page Titles
-----------

* Keep under 60 characters
* Include key terms (FUSION, optical network, simulation)
* Be descriptive and unique

Meta Descriptions
-----------------

Add to ``conf.py`` for main pages:

.. code-block:: python

   html_meta = {
       'description': 'FUSION: Flexible Unified System for Intelligent Optical Networking',
       'keywords': 'optical networks, SDN, simulation, flex-grid, elastic optical networks'
   }

URL Structure
-------------

* Use clear, hierarchical URLs
* Avoid special characters
* Use hyphens, not underscores in file names

Review Process
==============

Before Committing
-----------------

1. **Build locally**: ``make html`` must succeed
2. **Check warnings**: Address Sphinx warnings
3. **Test links**: Run ``make linkcheck``
4. **Review output**: Check formatting in browser
5. **Spell check**: Run codespell or similar

Peer Review
-----------

For major changes:

1. Get feedback from another developer
2. Test on multiple browsers
3. Verify mobile responsiveness
4. Check all cross-references work

See Also
========

* :doc:`contributing` - How to contribute to FUSION
* :doc:`workflow` - Development workflow
* :doc:`testing` - Testing standards
* `Sphinx RST Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
* `Google Developer Documentation Style Guide <https://developers.google.com/style>`_
