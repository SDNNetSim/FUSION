# RL Module TODO List

## Pending Tasks

### High Priority

1. **Support for resuming interrupted training** (v6.1)
   - Implement functionality to pick up where training left off
   - Save and restore training state
   - Handle checkpoint management

2. **Implement testing mode** (v6.1)
   - Currently raises `TrainingError` when `is_training=False`
   - Add support for model evaluation without training

### Medium Priority

3. **Standardize hyperparameter file discovery** (v6.1)
   - Standardize hyperparameter file loading across CLI tools
   - Create unified approach for yaml file discovery
   - Remove hardcoded paths where possible

### Low Priority

4. **Deprecated code cleanup** (v6.1+)
   - Review and remove deprecated pointer_policy.py if no longer needed
   - Clean up any other deprecated components

## Notes

This file tracks ongoing development tasks for the RL module. When implementing these tasks, ensure compliance with FUSION coding standards and update this file accordingly.
