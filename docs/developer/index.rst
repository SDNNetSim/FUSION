.. _developer-docs:

=====================
Developer Guide
=====================

This section provides in-depth documentation for developers working on specific
parts of the FUSION codebase. Each module page explains what the component does,
how it works internally, and how to develop on it.

.. tip::

   **New to FUSION?** Start with the :doc:`/getting-started/installation` section first,
   then return here when you need to work on a specific module.

How to Use This Guide
=====================

Each module page follows a consistent structure:

- **Overview** - What the module does and where it fits
- **Key Concepts** - Domain knowledge you need
- **Architecture** - Internal structure and design
- **Components** - Detailed file/directory descriptions
- **Dependencies** - What connects to what
- **Development Guide** - How to modify and extend
- **Testing** - How to verify your changes

Modules
=======

.. toctree::
   :maxdepth: 2
   :caption: fusion/

   fusion/analysis/index
   fusion/cli/index

Data Directory
==============

Documentation for the ``data/`` directory and its contents.

.. toctree::
   :maxdepth: 2

   data/index
