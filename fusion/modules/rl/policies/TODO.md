# Policies Module TODO List

## Status: BETA

This module is currently in **BETA** and is actively being developed.
The API may evolve in future releases.

## Pending Tasks

### High Priority

1. **Add unit tests for offline policies** (v6.X)
   - Test BCPolicy with mock models
   - Test IQLPolicy with mock models
   - Test action masking edge cases
   - Test integration with OfflinePolicyAdapter

2. **Improve model loading robustness** (v6.X)
   - Handle different model formats gracefully
   - Add model validation on load
   - Improve error messages for incompatible models

### Medium Priority

3. **Add CQL policy implementation** (v6.X)
   - Conservative Q-Learning for offline RL
   - Common alternative to IQL in literature
   - Complement existing BC and IQL options

4. **Enhance PointerPolicy** (v6.X)
   - Add configurable attention mechanisms
   - Support for multi-head variants
   - Performance optimization for larger path sets

### Low Priority

5. **Documentation improvements** (v6.X)
   - Add training tutorials for BC and IQL
   - Document model file format requirements
   - Add performance benchmarking guide

6. **Policy comparison framework** (v6.X)
   - Standardized evaluation metrics
   - Automated comparison scripts
   - Visualization of policy decisions

## Notes

This file tracks ongoing development tasks for the policies submodule. When implementing these tasks, ensure compliance with FUSION coding standards and update this file accordingly.
