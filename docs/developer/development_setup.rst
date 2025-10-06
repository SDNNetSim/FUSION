==================
Development Setup
==================

Setup your FUSION development environment.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Setup
===========

**Automated installation:**

.. code-block:: bash

   git clone https://github.com/SDNNetSim/FUSION.git
   cd FUSION
   ./install.sh

Or with make:

.. code-block:: bash

   make install      # Full installation
   make install-dev  # Dev tools only

Manual Setup
============

**1. Install core package:**

.. code-block:: bash

   pip install -e .

**2. Install development tools:**

.. code-block:: bash

   pip install -e .[dev]

**3. Install pre-commit hooks:**

.. code-block:: bash

   pre-commit install

Development Tools
=================

Essential Make Commands
-----------------------

.. code-block:: bash

   make format      # Auto-format code
   make lint-new    # Check for issues
   make test-new    # Run tests
   make check-all   # Full quality check

IDE Setup
---------

**VS Code recommended extensions:**

* Python
* Pylance
* Ruff
* MyPy

**PyCharm:**

* Enable ruff integration
* Configure mypy plugin

See Also
========

* :doc:`workflow` - Development workflow
* :doc:`contributing` - Contribution guide  
* :doc:`testing` - Testing standards
