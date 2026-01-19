====================
Visualizing Codebase
====================

Tools for visualizing and understanding the FUSION project structure.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSION provides two complementary visualization tools:

**Pydeps** - Module architecture and connectivity
   Shows which modules import which, helps understand project organization

**Pyreverse** - Class design and relationships
   Shows classes, methods, inheritance, and associations within modules

Installation
============

Install visualization tools:

.. code-block:: bash

   pip install pylint pydeps graphviz

Using Pydeps (Module Architecture)
===================================

Pydeps visualizes **how modules connect** to each other.

Quick Start
-----------

.. code-block:: bash

   # High-level architecture (recommended starting point)
   pydeps fusion --max-bacon=2 --exclude '**/tests/**' --cluster \
     -o reports/diagrams/architecture.png

   # View the diagram
   open reports/diagrams/architecture.png

Common Commands
---------------

**Overall architecture:**

.. code-block:: bash

   # Simple view (direct imports only)
   pydeps fusion --max-bacon=1 --exclude '**/tests/**' \
     -o reports/diagrams/simple.png

   # Detailed view with clustering
   pydeps fusion --max-bacon=2 --exclude '**/tests/**' --cluster \
     -o reports/diagrams/clustered.png

**Specific modules:**

.. code-block:: bash

   # RL module dependencies
   pydeps fusion/modules/rl --exclude '**/tests/**' \
     -o reports/diagrams/rl_deps.png

   # Visualization module
   pydeps fusion/visualization --exclude '**/tests/**' \
     -o reports/diagrams/viz_deps.png

   # Core module
   pydeps fusion/core --exclude '**/tests/**' \
     -o reports/diagrams/core_deps.png

**Find circular dependencies:**

.. code-block:: bash

   pydeps fusion --show-cycles --exclude '**/tests/**'

Key Options
-----------

``--max-bacon=N``
   Depth of import relationships (1=direct, 2=direct+indirect)

``--cluster``
   Group related modules together for clarity

``--exclude PATTERN``
   Exclude files/directories (use to remove tests)

``-o FILE``
   Output file path

Using Pyreverse (Class Design)
===============================

Pyreverse visualizes **classes and their relationships** within modules.

Quick Start
-----------

.. code-block:: bash

   # Generate class diagrams for a specific module
   pyreverse -o png -p rl fusion/modules/rl/

   # Move to reports directory
   mv classes_rl.png packages_rl.png reports/diagrams/

   # View the diagrams
   open reports/diagrams/classes_rl.png

Common Commands
---------------

**Visualize specific modules:**

.. code-block:: bash

   # RL module classes
   pyreverse -o png -p rl fusion/modules/rl/
   mv classes_rl.png packages_rl.png reports/diagrams/

   # Visualization module
   pyreverse -o png -p viz fusion/visualization/
   mv classes_viz.png packages_viz.png reports/diagrams/

   # Core classes
   pyreverse -o png -p core fusion/core/
   mv classes_core.png packages_core.png reports/diagrams/

Output Files
------------

Pyreverse generates two diagrams:

``classes_*.png``
   Shows all classes with their methods and attributes

``packages_*.png``
   Shows package structure and relationships

.. warning::
   Generating class diagrams for the entire project produces huge,
   unreadable images. Always visualize specific modules instead.

Automated Scripts
=================

FUSION includes pre-configured scripts for common visualization tasks.

Generate All Diagrams
----------------------

.. code-block:: bash

   # Generate comprehensive project diagrams
   ./scripts/generate_diagrams.sh

Creates in ``reports/diagrams/``:
   * ``classes_fusion.png`` - UML class diagram
   * ``packages_fusion.png`` - UML package diagram
   * ``dependencies.png`` - Overall dependency graph
   * ``architecture_overview.png`` - High-level architecture
   * ``module_interactions.png`` - Module interaction diagram
   * ``deps_*.png`` - Individual module dependency graphs

Dependency Analysis
-------------------

.. code-block:: bash

   # Analyze dependencies and find issues
   ./scripts/analyze_dependencies.sh

Creates in ``reports/analysis/``:
   * ``dependencies.png`` - Visual dependency graph
   * ``circular_dependencies.txt`` - Circular dependency report
   * ``dead_code.txt`` - Dead code analysis
   * ``dependency_report.txt`` - Detailed dependency report

Recommended Workflow
====================

When planning to add features or modify code:

1. **Start with Pydeps** (architecture)

   .. code-block:: bash

      # See overall architecture
      pydeps fusion --max-bacon=2 --exclude '**/tests/**' --cluster \
        -o reports/diagrams/arch.png

      # Identify which module(s) you need to modify

2. **Drill into specific module** (dependencies)

   .. code-block:: bash

      # Example: Working on RL features
      pydeps fusion/modules/rl --exclude '**/tests/**' \
        -o reports/diagrams/rl_module.png

      # See what this module depends on

3. **Use Pyreverse** (class design)

   .. code-block:: bash

      # See classes and their relationships
      pyreverse -o png -p rl fusion/modules/rl/
      mv classes_rl.png packages_rl.png reports/diagrams/

4. **Read the code**

   Now you know where to look and what classes to examine

Tips and Best Practices
========================

Pydeps Tips
-----------

* **Exclude tests** - Always use ``--exclude '**/tests/**'`` for cleaner diagrams
* **Start simple** - Use ``--max-bacon=1`` first, increase if needed
* **Use clustering** - Add ``--cluster`` to group related modules
* **Module-specific** - Focus on one area instead of the whole project
* **SVG format** - If PNG fails, try ``-o diagram.svg`` for browser viewing

Pyreverse Tips
--------------

* **Never visualize entire project** - Class diagrams get too large
* **Focus on modules** - Always specify a specific subdirectory
* **Two outputs** - Remember you get both classes and packages diagrams
* **Move files** - Pyreverse outputs to current directory, move to reports/

Common Issues
=============

"Image too large to open"
   Solution: Reduce ``--max-bacon`` value or focus on specific modules

"pydeps: command not found"
   Solution: ``pip install pydeps``

"No such file or directory"
   Solution: Ensure you're in project root and output directory exists:
   ``mkdir -p reports/diagrams``

"pyreverse: error: no such option: --output-dir"
   Solution: Old pyreverse version doesn't support this flag. Generate in
   current directory and move files manually.

See Also
========

* :doc:`architecture` - Project architecture overview
* :doc:`development_setup` - Development environment setup
* :doc:`extending` - Extending FUSION
