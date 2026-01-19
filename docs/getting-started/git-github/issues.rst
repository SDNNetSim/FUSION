Submitting Issues
=================

GitHub Issues are our primary way to track bugs, feature requests, and discussions.
This guide explains how to submit effective issues.

Before Submitting
-----------------

1. **Search existing issues** at https://github.com/SDNNetSim/FUSION/issues
2. **Check if it's already fixed** in the latest version
3. **Gather information** about your environment and the problem

Creating an Issue
-----------------

1. Go to the `Issues tab <https://github.com/SDNNetSim/FUSION/issues>`_ and click "New issue"
2. GitHub will automatically show you the available templates:

   - **Bug Report** - For reporting problems
   - **Feature Request** - For suggesting new functionality

3. Select a template and fill it out completely

.. tip::

   The issue templates are YAML forms that guide you through providing all the
   necessary information. You can view the template files in the repository at
   ``.github/ISSUE_TEMPLATE/``:

   - ``01_bug_report.yml`` - Bug report template
   - ``02_feature_request.yml`` - Feature request template

Bug Reports
-----------

When reporting a bug, the template will ask for:

**Bug Summary**
   A clear, concise description of the problem.

**Impact Level**
   How severely does this affect your work?

   - **Critical**: Simulation crashes or data corruption
   - **High**: Incorrect results or major functionality broken
   - **Medium**: Workflow disruption or performance issues
   - **Low**: Minor inconvenience or cosmetic issues

**Affected Component**
   Which part of FUSION is affected (CLI, routing, spectrum, etc.)

**Steps to Reproduce**
   Detailed steps to recreate the issue, including:

   - Configuration file used
   - Command executed
   - Step-by-step actions

**Expected vs Actual Behavior**
   What should happen vs what actually happens.

**Error Information**
   Full error messages and stack traces.

**Environment Details**
   - Operating system
   - Python version
   - FUSION version/branch

**Example Bug Report**

.. code-block:: text

   Bug Summary: Simulation crashes when using multi-core fiber with FirstFit

   Impact Level: High

   Steps to Reproduce:
   1. Set num_cores = 7 in config
   2. Run simulation with FirstFit spectrum assignment
   3. Simulation crashes at ~500 arrivals

   Expected: Simulation completes normally
   Actual: Crashes with IndexError

   Error:
   IndexError: index 7 is out of bounds for axis 0 with size 7

   Environment: Ubuntu 22.04, Python 3.11.5, main branch

Feature Requests
----------------

When requesting a feature, the template will ask for:

**Feature Title**
   A concise name for the feature.

**Priority Level**
   How important is this to you?

**Problem Statement**
   What problem does this solve? What can't users do currently?

**Proposed Solution**
   Your suggested implementation approach.

**Acceptance Criteria**
   What does "done" look like? Use checkboxes:

   .. code-block:: text

      - [ ] Feature works with CLI
      - [ ] Configuration validation updated
      - [ ] Unit tests added
      - [ ] Documentation updated

**Example Feature Request**

.. code-block:: text

   Feature Title: Add support for dynamic spectrum defragmentation

   Priority: Medium

   Problem Statement:
   As a researcher studying spectrum efficiency, I want to simulate
   defragmentation strategies so that I can compare their effectiveness.

   Proposed Solution:
   Add a DefragmentationPolicy interface with configurable strategies
   (e.g., repack-left, minimize-moves) that can run periodically or
   on-demand during simulation.

   Acceptance Criteria:
   - [ ] DefragmentationPolicy interface defined
   - [ ] At least two strategies implemented
   - [ ] Configuration options added
   - [ ] Metrics for defragmentation events collected

Issue Labels
------------

Our team uses labels to categorize and prioritize issues:

- **bug**: Something isn't working
- **enhancement**: New feature or improvement
- **documentation**: Documentation improvements
- **needs-triage**: Needs team review
- **needs-discussion**: Requires discussion before implementation
- **good first issue**: Good for newcomers

Tips for Good Issues
--------------------

- **Be specific**: Include exact error messages, not paraphrases
- **Be complete**: Fill out all template sections
- **Be minimal**: If possible, create a minimal reproduction case
- **Be patient**: Maintainers are volunteers; complex issues take time
- **Follow up**: Respond to questions and provide additional info when asked
