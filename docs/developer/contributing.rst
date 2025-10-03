============
Contributing
============

Thank you for your interest in contributing to FUSION!

.. contents:: Table of Contents
   :local:
   :depth: 2

How to Contribute
=================

**Ways to contribute:**

1. **Report Issues**: Use GitHub issue tracker
2. **Submit Pull Requests**: Code or documentation improvements
3. **Review PRs**: Provide feedback on others' contributions

Getting Started
===============

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make your changes
4. Run quality checks: ``make check-all``
5. Commit: ``git commit -m "Add feature"``
6. Push and create PR

Coding Standards
================

**Style Guide:**

* Follow PEP 8
* Use type annotations
* Write descriptive docstrings
* Add tests for new features

**Run formatters:**

.. code-block:: bash

   make format  # Auto-format with ruff
   make lint-new  # Check for issues

Pull Request Process
====================

1. Ensure tests pass: ``make test-new``
2. Update documentation if needed
3. Describe changes clearly in PR
4. Link related issues
5. Wait for review (usually 2-3 days)

**PR Checklist:**

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted (``make format``)
- [ ] All checks pass (``make check-all``)

Code Review
===========

**What we look for:**

* Code quality and readability
* Test coverage
* Documentation completeness  
* Breaking changes identified

**Response time:** We aim to review PRs within 3 business days.

See Also
========

* :doc:`development_setup` - Setup your environment
* :doc:`workflow` - Development workflow
* :doc:`coding_standards` - Detailed standards
* :doc:`testing` - Testing guidelines
