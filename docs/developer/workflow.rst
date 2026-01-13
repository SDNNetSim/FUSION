===================
Development Workflow
===================

Daily development workflow for FUSION contributors.

.. contents:: Table of Contents
   :local:
   :depth: 2

Daily Workflow
==============

**1. Format code:**

.. code-block:: bash

   make format

**2. Check for issues:**

.. code-block:: bash

   make lint-new

**3. Run tests:**

.. code-block:: bash

   make test-new

**4. Before PR:**

.. code-block:: bash

   make check-all

Git Workflow
============

**Feature development:**

.. code-block:: bash

   git checkout -b feature/my-feature
   # Make changes
   make check-all
   git commit -m "Add feature"
   git push origin feature/my-feature

**Bug fixes:**

.. code-block:: bash

   git checkout -b fix/issue-123
   # Fix bug
   make test-new
   git commit -m "Fix issue #123"
   git push origin fix/issue-123

Pre-commit Hooks
================

Automatically run on commit:

* Ruff formatting
* Type checking with mypy
* Import sorting

Testing Workflow
================

**Run specific tests:**

.. code-block:: bash

   pytest tests/test_routing.py
   pytest -k "test_first_fit"

**With coverage:**

.. code-block:: bash

   make test-new

See Also
========

* :doc:`testing` - Testing guidelines
* :doc:`contributing` - How to contribute
