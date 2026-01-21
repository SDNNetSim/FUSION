Git and GitHub
==============

FUSION is hosted on GitHub and uses Git for version control. This guide covers
everything you need to know to get the code, contribute changes, and collaborate
with the team.

New to Git?
-----------

If you're new to Git, we recommend these resources:

- `GitHub Git Guides <https://github.com/git-guides>`_ - GitHub's official introduction
- `Pro Git Book <https://git-scm.com/book/en/v2>`_ - Comprehensive free book
- `Learn Git Branching <https://learngitbranching.js.org/>`_ - Interactive tutorial

Repository
----------

FUSION is hosted at: https://github.com/SDNNetSim/FUSION

.. toctree::
   :maxdepth: 1
   :caption: Git and GitHub Guide

   getting-fusion
   workflow
   issues
   pull-requests
   commit-messages
   workflows

Quick Reference
---------------

**For external contributors** (branch from ``main``):

.. code-block:: bash

   # Clone and create a feature branch from main
   git clone git@github.com:SDNNetSim/FUSION.git
   git checkout main
   git checkout -b feature/my-feature

   # Make changes, stage, and commit
   git add .
   git commit -m "feat(module): add new feature"

   # Push and open PR targeting 'main'
   git push origin feature/my-feature

**For internal team** (branch from ``develop``):

.. code-block:: bash

   # Branch from develop
   git checkout develop
   git checkout -b feature/my-feature

   # Push and open PR targeting 'develop'
   git push origin feature/my-feature
