Getting FUSION
==============

This guide walks you through getting a copy of FUSION on your local machine.

Prerequisites
-------------

Before you begin, ensure you have:

- **Git** installed (`download <https://git-scm.com/downloads>`_)
- **GitHub account** (`sign up <https://github.com/signup>`_)
- **SSH key** configured (recommended) or HTTPS access

Option 1: Clone Directly (Read-Only)
------------------------------------

If you just want to use FUSION without contributing:

.. code-block:: bash

   # Using SSH (recommended)
   git clone git@github.com:SDNNetSim/FUSION.git

   # Using HTTPS
   git clone https://github.com/SDNNetSim/FUSION.git

   # Enter the directory
   cd FUSION

Option 2: Fork and Clone (For Contributors)
-------------------------------------------

If you plan to contribute changes:

1. **Fork the repository**

   - Go to https://github.com/SDNNetSim/FUSION
   - Click the "Fork" button in the top right
   - This creates a copy under your GitHub account

2. **Clone your fork**

   .. code-block:: bash

      # Replace YOUR_USERNAME with your GitHub username
      git clone git@github.com:YOUR_USERNAME/FUSION.git
      cd FUSION

3. **Add the upstream remote**

   .. code-block:: bash

      git remote add upstream git@github.com:SDNNetSim/FUSION.git

   This lets you pull updates from the main repository.

4. **Verify your remotes**

   .. code-block:: bash

      git remote -v

   You should see:

   .. code-block:: text

      origin    git@github.com:YOUR_USERNAME/FUSION.git (fetch)
      origin    git@github.com:YOUR_USERNAME/FUSION.git (push)
      upstream  git@github.com:SDNNetSim/FUSION.git (fetch)
      upstream  git@github.com:SDNNetSim/FUSION.git (push)

Staying Updated
---------------

Keep your fork synchronized with the main repository:

.. code-block:: bash

   # Fetch updates from upstream
   git fetch upstream

   # Update main branch (stable - for external contributors)
   git checkout main
   git merge upstream/main
   git push origin main

For internal team members, also sync ``develop``:

.. code-block:: bash

   # Update develop branch (internal team only)
   git checkout develop
   git merge upstream/develop
   git push origin develop

Next Steps
----------

After cloning, follow the :doc:`/getting-started/installation` guide
to set up your development environment.
