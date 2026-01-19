Pull Requests
=============

Pull requests (PRs) are how you contribute code changes to FUSION. This guide
explains the process and requirements.

Before You Start
----------------

1. **Check for existing work**: Search issues and PRs to avoid duplicating effort
2. **Discuss large changes**: Open an issue first to discuss significant changes
3. **Fork and branch**: Create a working branch from an up-to-date base branch

Creating a Pull Request
-----------------------

1. **Push your branch** to your fork:

   .. code-block:: bash

      git push origin feature/my-feature

2. **Open a PR** on GitHub:

   - Go to https://github.com/SDNNetSim/FUSION/pulls
   - Click "New pull request"
   - Select your fork and branch
   - **Set the base branch** (see note below)
   - Click "Create pull request"

3. **Fill out the template** completely

.. note::

   **External contributors**: Target the ``main`` branch. The team will handle integration.

   **Internal team**: Target the ``develop`` branch. Maintainers handle merges
   from ``develop`` to ``release/*`` and from ``release/*`` to ``main``.

PR Template Sections
--------------------

When you open a PR, GitHub automatically loads our template. You can view
the template files in ``.github/PULL_REQUEST_TEMPLATE/``:

- ``pull_request_template.md`` - Standard PR template
- ``feature_pr_template.md`` - Feature-specific template
- ``hotfix_pr_template.md`` - Hotfix template

Our PR template ensures thorough submissions:

**PR Title**
   Use conventional commit format: ``type(scope): description``

   .. code-block:: text

      feat(routing): add congestion-aware path selection
      fix(spectrum): resolve allocation conflict in multi-core
      docs(readme): update installation instructions

**Related Issues**
   Link issues this PR addresses:

   .. code-block:: text

      Fixes #123
      Closes #456
      Relates to #789

**Type of Change**
   Select what kind of change this is:

   - Bug Fix
   - New Feature
   - Breaking Change
   - Refactor
   - Documentation
   - Tests
   - Performance

**Testing**
   Describe how you tested your changes:

   - What tests did you add?
   - What manual testing did you perform?
   - What configuration did you use?

**Impact Analysis**
   Assess the impact on:

   - Performance
   - Memory usage
   - Backward compatibility
   - Dependencies

Requirements
------------

Before your PR can be merged:

**Code Quality**

.. code-block:: bash

   make format      # Auto-format code
   make lint        # Check for issues
   make test        # Run tests
   make check-all   # Full validation

**Tests**
   - Add tests for new functionality
   - Ensure existing tests pass
   - Aim for good coverage

**Documentation**
   - Update docstrings
   - Update relevant docs
   - Add usage examples

**Review**
   - Two approvals required
   - Address all review comments
   - CI checks must pass

Review Process
--------------

1. **Automated checks** run on your PR (linting, tests, etc.)
2. **Team members review** your code
3. **Address feedback** by pushing additional commits
4. **Final approval** from two reviewers
5. **Merge** by a maintainer

Responding to Reviews
---------------------

When reviewers leave comments:

- **Be responsive**: Reply to all comments
- **Ask questions**: If feedback is unclear, ask for clarification
- **Make changes**: Push new commits to address feedback
- **Re-request review**: After addressing comments, re-request review

Keeping Your PR Updated
-----------------------

If main has been updated:

.. code-block:: bash

   git fetch upstream
   git rebase upstream/main
   git push origin feature/my-feature --force-with-lease

Common PR Issues
----------------

**CI Failures**
   - Check the failing job's logs
   - Run checks locally: ``make check-all``
   - Fix issues and push

**Merge Conflicts**
   - Rebase on the latest main
   - Resolve conflicts locally
   - Push the updated branch

**Stale PR**
   - If no activity for a while, maintainers may close it
   - Comment if you're still working on it

After Merge
-----------

Once merged:

1. Delete your feature branch (GitHub offers a button)
2. Pull the updated main to your local repo
3. Celebrate your contribution!
