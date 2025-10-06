============
Installation
============

This guide will walk you through installing FUSION on your system.

.. contents:: Table of Contents
   :local:
   :depth: 2

System Requirements
===================

Before installing FUSION, ensure your system meets these requirements:

* **Python**: Version 3.11.x (required)
* **Memory**: Minimum 4GB RAM (8GB+ recommended for large simulations)
* **Storage**: At least 2GB free space
* **Git**: For cloning the repository

Supported Operating Systems
----------------------------

* macOS (Intel and Apple Silicon)
* Ubuntu 20.04+
* Fedora 37+
* Windows 11

Quick Installation (Recommended)
=================================

The fastest way to install FUSION is using our automated installation script:

.. code-block:: bash

   # Clone the repository
   git clone git@github.com:SDNNetSim/FUSION.git
   cd FUSION

   # Create and activate a Python 3.11 virtual environment
   python3.11 -m venv venv

.. tabs::

   .. tab:: macOS/Linux

      .. code-block:: bash

         source venv/bin/activate
         ./install.sh

   .. tab:: Windows

      .. code-block:: bash

         venv\Scripts\activate
         bash install.sh

The script will automatically:

* ✅ Detect your platform (macOS, Linux, Windows)
* ✅ Handle PyTorch Geometric compilation issues
* ✅ Install all dependencies in the correct order
* ✅ Set up development tools
* ✅ Install and configure pre-commit hooks
* ✅ Verify the installation

.. note::
   On Windows, you may need to run the installation commands manually.
   See the :ref:`manual-installation` section below.

Package Installation
====================

For more control over the installation process:

Core Installation
-----------------

Install the core FUSION package:

.. code-block:: bash

   # Clone and create virtual environment
   git clone git@github.com:SDNNetSim/FUSION.git
   cd FUSION
   python3.11 -m venv venv
   source venv/bin/activate

   # Install core package
   pip install -e .

Optional Components
-------------------

Install additional features as needed:

.. code-block:: bash

   # Development tools (ruff, mypy, pytest, pre-commit)
   pip install -e .[dev]

   # Reinforcement learning (stable-baselines3, gymnasium)
   pip install -e .[rl]

   # Everything except PyTorch Geometric
   pip install -e .[all]

PyTorch Geometric Installation
-------------------------------

PyTorch Geometric requires special installation steps depending on your platform:

.. tabs::

   .. tab:: macOS (Apple Silicon)

      .. code-block:: bash

         MACOSX_DEPLOYMENT_TARGET=11.0 pip install --no-build-isolation \
             torch-scatter torch-sparse torch-cluster torch-spline-conv \
             -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
         pip install torch-geometric==2.6.1

   .. tab:: macOS (Intel)

      .. code-block:: bash

         MACOSX_DEPLOYMENT_TARGET=10.15 pip install --no-build-isolation \
             torch-scatter torch-sparse torch-cluster torch-spline-conv \
             -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
         pip install torch-geometric==2.6.1

   .. tab:: Linux

      .. code-block:: bash

         pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
             -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
         pip install torch-geometric==2.6.1

   .. tab:: Windows

      .. code-block:: bash

         pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
             -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
         pip install torch-geometric==2.6.1

.. _manual-installation:

Manual Installation
===================

If you prefer to install dependencies manually:

Step 1: Install PyTorch
-----------------------

.. code-block:: bash

   pip install torch==2.2.2

Step 2: Install Core Requirements
----------------------------------

.. code-block:: bash

   pip install -r requirements.txt

Step 3: Install Development Tools (Optional)
---------------------------------------------

.. code-block:: bash

   pip install -r requirements-dev.txt

Step 4: Install PyTorch Geometric
----------------------------------

Follow the platform-specific instructions in the :ref:`PyTorch Geometric Installation` section above.

Verifying Installation
======================

After installation, verify everything is working:

.. code-block:: bash

   # Check Python version
   python --version  # Should show Python 3.11.x

   # Try importing FUSION
   python -c "import fusion; print('FUSION installed successfully!')"

   # Run tests (if dev tools installed)
   pytest fusion/ -v

Development Setup
=================

If you plan to contribute to FUSION, set up the development environment:

Install Pre-commit Hooks
-------------------------

.. code-block:: bash

   # Install pre-commit
   pip install pre-commit

   # Install git hooks
   pre-commit install
   pre-commit install --hook-type commit-msg

Run Quality Checks
------------------

.. code-block:: bash

   # Run all checks
   pre-commit run --all-files

   # Run tests
   make test

   # Run linting
   make lint

Troubleshooting
===============

Installation Issues
-------------------

**Issue: Python 3.11 not found**

Install Python 3.11 from the `official website <https://www.python.org/downloads/>`_
or use a package manager:

.. code-block:: bash

   # macOS with Homebrew
   brew install python@3.11

   # Ubuntu/Debian
   sudo apt install python3.11 python3.11-venv

   # Fedora
   sudo dnf install python3.11

**Issue: PyTorch Geometric fails to compile**

Use the automated installer, which handles platform-specific compilation flags:

.. code-block:: bash

   ./install.sh

**Issue: Permission denied when running install.sh**

Make the script executable:

.. code-block:: bash

   chmod +x install.sh
   ./install.sh

**Issue: SSL certificate errors**

Update your pip and try again:

.. code-block:: bash

   pip install --upgrade pip
   pip install --upgrade certifi

Virtual Environment Issues
--------------------------

**Issue: Cannot activate virtual environment**

Ensure you're using the correct command for your shell:

.. code-block:: bash

   # Bash/Zsh
   source venv/bin/activate

   # Windows Command Prompt
   venv\Scripts\activate.bat

   # Windows PowerShell
   venv\Scripts\Activate.ps1

Dependency Conflicts
--------------------

If you encounter dependency conflicts:

.. code-block:: bash

   # Create a fresh virtual environment
   deactivate
   rm -rf venv
   python3.11 -m venv venv
   source venv/bin/activate

   # Reinstall
   ./install.sh

Getting Help
============

If you're still having trouble:

* Check the :doc:`../reference/troubleshooting` guide
* Review `GitHub Issues <https://github.com/SDNNetSim/FUSION/issues>`_
* Open a new issue with:

  * Your operating system and version
  * Python version
  * Full error message
  * Steps to reproduce

Next Steps
==========

Now that FUSION is installed, head over to the :doc:`quickstart` guide
to run your first simulation!

.. seealso::

   * :doc:`quickstart` - Run your first simulation
   * :doc:`configuration` - Learn about configuration options
   * :doc:`../developer/development_setup` - Full development environment setup
