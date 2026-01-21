# Gymnasium Environments Module TODO List

## Pending Tasks

### High Priority

1. **Extend spectral band support beyond C-band** (v6.X)
   - Currently only C-band is supported for RL environments
   - Add support for L-band spectrum allocation
   - Implement multi-band observation and action spaces
   - Update reward functions to handle cross-band optimization

2. **Complete UnifiedSimEnv migration** (v6.X)
   - Deprecate and remove legacy GeneralSimEnv
   - Ensure feature parity between legacy and unified environments
   - Update all documentation and examples

### Medium Priority

3. **Add comprehensive environment testing** (v6.X)
   - Increase test coverage for edge cases
   - Add integration tests with full simulation stack
   - Test multi-band scenarios once implemented

4. **Improve observation space flexibility** (v6.X)
   - Support configurable observation components
   - Add optional network-wide state information
   - Consider alternative graph representations

### Low Priority

5. **Performance optimization** (v6.X)
   - Profile environment step performance
   - Optimize observation construction
   - Consider vectorized environment support

## Known Limitations

- **C-band only**: RL environments currently only support C-band spectrum allocation. L-band and multi-band scenarios are not yet supported.
- **Legacy environment**: GeneralSimEnv is deprecated but still available for backward compatibility.

## Notes

This file tracks ongoing development tasks for the gymnasium environments submodule. When implementing these tasks, ensure compliance with FUSION coding standards and update this file accordingly.
