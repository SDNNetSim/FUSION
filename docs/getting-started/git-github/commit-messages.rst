Commit Messages
===============

FUSION follows the `Conventional Commits <https://www.conventionalcommits.org/>`_
specification for clear, consistent commit messages.

Format
------

.. code-block:: text

   <type>(scope): <description>

   [optional body]

   [optional footer]

**Example:**

.. code-block:: text

   feat(routing): add congestion-aware path selection

   Implement a new routing strategy that considers link utilization
   when selecting paths. This reduces blocking probability under
   high load conditions.

   Resolves #123

Commit Types
------------

Use these types based on what you changed:

- **feat**: New features
- **fix**: Bug fixes
- **docs**: Documentation changes
- **style**: Code style (formatting, no logic change)
- **refactor**: Code restructuring (no new features or fixes)
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks
- **build**: Build system changes
- **ci**: CI/CD changes

Scopes
------

Use scopes that match FUSION's modules:

- **cli**: Command-line interface
- **config**: Configuration system
- **core**: Simulation core
- **routing**: Routing algorithms
- **spectrum**: Spectrum assignment
- **snr**: SNR calculations
- **rl**: Reinforcement learning
- **ml**: Machine learning
- **viz**: Visualization
- **test**: Testing framework

Rules
-----

**Do:**

- Start with a lowercase letter after the colon
- Use imperative mood: "add" not "added" or "adds"
- Keep the subject line under 100 characters
- Be descriptive about what and why

**Don't:**

- End with a period
- Use vague messages like "fix bug" or "update code"
- Exceed 100 characters in the subject line

Good Examples
-------------

.. code-block:: text

   feat(cli): add support for custom configuration templates
   fix(spectrum): resolve allocation conflict in multi-core scenarios
   perf(core): reduce simulation startup time by 40%
   refactor(cli): extract argument validation into separate module
   docs: add troubleshooting guide for common installation issues
   test(routing): add unit tests for k-shortest path algorithm

Bad Examples
------------

.. code-block:: text

   Fixed bug                    # Too vague
   Update README.md             # What was updated?
   feat: fix                    # Contradictory
   Added some changes           # Not descriptive
   WIP: working on feature      # Not a complete change

Complex Changes
---------------

For significant changes, add a body:

.. code-block:: text

   feat(config): add schema-based validation system

   Implement comprehensive configuration validation using JSON Schema
   to catch errors early and provide helpful error messages.

   Changes include:
   - Add schema definitions for all config sections
   - Implement validation in ConfigManager class
   - Add detailed error messages with suggestions

   Resolves #123, #145

Referencing Issues
------------------

Link related issues in the footer:

.. code-block:: text

   fix(routing): resolve null pointer exception

   Fixes #234

Keywords that close issues:

- ``Fixes #123``
- ``Closes #123``
- ``Resolves #123``

Why This Matters
----------------

Consistent commit messages:

- Make history easy to read
- Enable automated changelog generation
- Help reviewers understand changes
- Make debugging easier with ``git bisect``
