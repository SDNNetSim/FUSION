CI/CD Workflows
===============

FUSION uses GitHub Actions for continuous integration and deployment. This page
explains what each workflow does and what checks your code must pass.

Overview
--------

When you push code or open a PR, several automated workflows run:

==================================  =============================================
Workflow                            Purpose
==================================  =============================================
**Unit Tests**                      Runs pytest on all unit tests
**Code Quality**                    Linting, formatting, type checking, security
**Commit Message Validation**       Ensures conventional commit format
**Cross-Platform Tests**            Tests on Windows, macOS, and Linux
**Algorithm Results Verification**  Verifies algorithm consistency
**Documentation**                   Builds and deploys Sphinx docs
==================================  =============================================

Workflow files are located in ``.github/workflows/``.

Unit Tests
----------

**File:** ``unittests.yml``

Runs on PRs and pushes to ``develop``, ``main``, and ``release/**`` branches.

**What it checks:**

- All pytest tests pass
- Tests run on Python 3.11

**If it fails:** Fix the failing tests before your PR can be merged.

Code Quality
------------

**File:** ``quality.yml``

Runs on every push and PR.

**What it checks:**

- **Format check**: Code follows ruff formatting rules
- **Linting**: No linting errors (ruff)
- **Type checking**: mypy type annotations are correct
- **Security**: No vulnerabilities detected (bandit)
- **Dead code**: No unused code (vulture)
- **Test coverage**: Pytest with coverage reporting

**If it fails:** Run these commands locally to fix issues:

.. code-block:: bash

   make format      # Auto-fix formatting
   make lint        # Check for linting issues
   make test        # Run tests

Commit Message Validation
-------------------------

**File:** ``commit_message.yml``

Validates that commit messages follow the conventional commits format.

**What it checks:**

- Message follows ``type(scope): description`` format
- Subject line is under 100 characters
- Description is at least 5 characters
- No trailing period

**Valid types:** ``feat``, ``fix``, ``docs``, ``style``, ``refactor``,
``perf``, ``test``, ``chore``, ``build``, ``ci``

**If it fails:** Amend your commit message:

.. code-block:: bash

   git commit --amend -m "feat(module): proper commit message"
   git push --force-with-lease

See :doc:`commit-messages` for detailed guidelines.

Cross-Platform Tests
--------------------

**File:** ``cross_platform.yml``

Ensures FUSION works on all supported operating systems.

**What it checks:**

- Installation succeeds on Windows, macOS, and Linux
- Basic simulation runs successfully on all platforms

**If it fails:** Your changes may have platform-specific issues. Check the
logs to see which platform failed.

Algorithm Results Verification
------------------------------

**File:** ``algorithm_verification.yml``

Compares algorithm results before and after changes to ensure consistency
across many algorithms.

**What it checks:**

- Algorithm outputs match expected baselines
- Results remain consistent after code changes
- No regressions in algorithm behavior

**If it fails:** Your changes may have altered algorithm behavior. This
could be intentional (update baselines) or a bug (fix your code).

Documentation
-------------

**File:** ``docs.yml``

Builds and deploys the Sphinx documentation.

**What it checks:**

- Documentation builds without errors
- No broken links (on PRs)
- No spelling errors (on PRs)

**Deployment:** When changes are merged to ``main``, documentation is
automatically deployed to GitHub Pages.

**If it fails:** Check the Sphinx build output for errors. Common issues:

- Invalid RST/Markdown syntax
- Missing files referenced in toctree
- Broken cross-references

Checking Workflow Status
------------------------

You can see workflow status:

1. **On your PR**: Check marks or X marks appear next to each workflow
2. **Actions tab**: Go to https://github.com/SDNNetSim/FUSION/actions
3. **Commit status**: Each commit shows workflow results

Running Checks Locally
----------------------

Before pushing, run checks locally to catch issues early:

.. code-block:: bash

   # Run all quality checks
   make check-all

   # Or run individual checks
   make format      # Format code
   make lint        # Lint code
   make test        # Run tests

Pre-commit hooks also run automatically when you commit.
