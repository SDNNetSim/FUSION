===============
Release Process
===============

Version management and release workflow.

.. contents:: Table of Contents
   :local:
   :depth: 2

Versioning
==========

FUSION uses **Semantic Versioning** (semver):

* **Major**: Breaking changes (v6.0.0 → v7.0.0)
* **Minor**: New features (v6.0.0 → v6.1.0)
* **Patch**: Bug fixes (v6.0.0 → v6.0.1)

Release Checklist
=================

**Before release:**

1. Update version in ``setup.py``
2. Update ``CHANGELOG.md``
3. Run full test suite: ``make test-new``
4. Build documentation: ``cd docs && make html``
5. Create git tag: ``git tag v6.1.0``
6. Push tag: ``git push origin v6.1.0``

**GitHub Release:**

1. Draft new release
2. Add release notes from CHANGELOG
3. Attach built artifacts if needed
4. Publish release

Branching Strategy
==================

* **main**: Stable releases
* **develop**: Development branch
* **feature/***: Feature branches
* **hotfix/***: Urgent fixes

See Also
========

* :doc:`contributing` - Contribution guide
* :doc:`workflow` - Development workflow
