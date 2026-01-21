Development Workflow
====================

This guide describes the day-to-day Git workflow for contributing to FUSION.

Branch Strategy
---------------

FUSION uses a three-tier branch strategy:

**Main Branches**

- **main**: Fully stable, production-ready code
- **release/\***: Stabilization branches (e.g., ``release/v6.1``) - more stable but working out kinks
- **develop**: Integration branch for fresh/unstable features (internal team only)

**Working Branches**

- **feature/\***: New features (e.g., ``feature/add-new-algorithm``)
- **fix/\***: Bug fixes (e.g., ``fix/routing-error``)
- **docs/\***: Documentation changes (e.g., ``docs/update-readme``)
- **refactor/\***: Code refactoring (e.g., ``refactor/cli-structure``)

For External Contributors
-------------------------

If you're contributing from outside the core team, work from the stable ``main`` branch:

.. code-block:: bash

   # Update main branch first
   git checkout main
   git pull upstream main

   # Create and switch to a new branch
   git checkout -b feature/my-new-feature

Your pull requests should target ``main``. The team will handle integration.

For Internal Team
-----------------

Core team members work from the ``develop`` branch:

.. code-block:: bash

   # Update develop branch first
   git checkout develop
   git pull upstream develop

   # Create and switch to a new branch
   git checkout -b feature/my-new-feature

**Internal Branch Flow**

.. code-block:: text

   feature/* ─┐
   fix/*     ─┼──> develop ──> release/* ──> main
   docs/*    ─┘

1. Create working branches from ``develop``
2. Merge completed work into ``develop``
3. When ready for release, ``develop`` merges into a ``release/*`` branch
4. After stabilization, ``release/*`` merges into ``main``

Use descriptive branch names that indicate what you're working on.

Making Changes
--------------

1. **Make your changes** to the code

2. **Check what changed**

   .. code-block:: bash

      git status
      git diff

3. **Stage your changes**

   .. code-block:: bash

      # Stage specific files
      git add path/to/file.py

      # Or stage all changes
      git add .

4. **Commit with a descriptive message**

   .. code-block:: bash

      git commit -m "feat(routing): add congestion-aware path selection"

   See :doc:`commit-messages` for commit message guidelines.

5. **Push to your fork**

   .. code-block:: bash

      git push origin feature/my-new-feature

Running Quality Checks
----------------------

Before pushing, run the quality checks:

.. code-block:: bash

   # Format code
   make format

   # Run linting
   make lint

   # Run tests
   make test

   # Run all checks
   make check-all

Pre-commit hooks will also run automatically when you commit.

Keeping Your Branch Updated
---------------------------

If the base branch has been updated while you're working:

.. code-block:: bash

   # Fetch upstream changes
   git fetch upstream

   # Rebase your branch (use main for external, develop for internal)
   git rebase upstream/main   # External contributors
   # OR
   git rebase upstream/develop   # Internal team

   # If there are conflicts, resolve them, then:
   git rebase --continue

   # Force push to update your branch (only for your own branches)
   git push origin feature/my-new-feature --force-with-lease

Common Git Commands
-------------------

.. code-block:: bash

   # Check current status
   git status

   # View commit history
   git log --oneline

   # Undo staged changes
   git restore --staged <file>

   # Discard local changes
   git restore <file>

   # Stash changes temporarily
   git stash
   git stash pop

   # Switch branches
   git checkout branch-name

   # Delete a local branch
   git branch -d branch-name
