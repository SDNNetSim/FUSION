## 📋 Pull Request Summary

**PR Title**: 
<!-- Use conventional commit format: type(scope): description -->
<!-- Examples: feat(cli): add new CLI command, fix(routing): resolve path calculation bug -->

**Related Issue(s)**: 
<!-- Link related issues: Fixes #123, Closes #456, Relates to #789 -->

**Description**:
<!-- Provide a clear summary of the changes and motivation -->

## 🔧 Type of Change

**Primary Change Type**:
- [ ] 🐛 **Bug Fix** - Non-breaking change that fixes an issue
- [ ] ✨ **New Feature** - Non-breaking change that adds functionality  
- [ ] 💥 **Breaking Change** - Change that would cause existing functionality to break
- [ ] 🔄 **Refactor** - Code change that neither fixes a bug nor adds a feature
- [ ] 📚 **Documentation** - Documentation only changes
- [ ] 🧪 **Tests** - Adding missing tests or correcting existing tests
- [ ] 🏗️ **Build/CI** - Changes to build process or CI configuration
- [ ] 🎨 **Style** - Code style changes (formatting, missing semicolons, etc.)
- [ ] ⚡ **Performance** - Performance improvements
- [ ] 🔒 **Security** - Security vulnerability fixes

**Component(s) Affected**:
- [ ] CLI Interface (`fusion/cli/`)
- [ ] Configuration System (`fusion/configs/`)
- [ ] Simulation Core (`fusion/core/`)
- [ ] ML/RL Modules (`fusion/modules/rl/`, `fusion/modules/ml/`)
- [ ] Routing Algorithms (`fusion/modules/routing/`)
- [ ] Spectrum Assignment (`fusion/modules/spectrum/`)
- [ ] SNR Calculations (`fusion/modules/snr/`)
- [ ] Visualization (`fusion/visualization/`)
- [ ] GUI Interface (`fusion/gui/`)
- [ ] Unity/HPC Integration (`fusion/unity/`)
- [ ] Testing Framework (`tests/`)
- [ ] Documentation
- [ ] GitHub Workflows (`.github/`)
- [ ] Build/Dependencies

## 🧪 Testing

**Test Coverage**:
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated  
- [ ] Manual testing performed
- [ ] Existing tests still pass
- [ ] Performance impact assessed

**Test Details**:
<!-- Describe the tests you ran to verify your changes -->

**Test Configuration Used**:
```ini
# Paste relevant test configuration
[general_settings]
# your test config...
```

**Commands to Reproduce Testing**:
```bash
# Examples:
python -m pytest tests/
python -m fusion.cli.run_sim run_sim --config_path test_config.ini --run_id test
```

**Test Results**:
- **Operating System**: [e.g., Ubuntu 22.04, macOS 13, Windows 11]
- **Python Version**: [e.g., 3.11.5]
- **Test Environment**: [e.g., local, CI/CD, Docker]

## 📊 Impact Analysis

**Performance Impact**:
- [ ] No performance impact
- [ ] Performance improved
- [ ] Minor performance decrease (acceptable)
- [ ] Significant performance impact (needs discussion)

**Memory Usage**:
- [ ] No change in memory usage
- [ ] Memory usage optimized  
- [ ] Minor increase in memory usage
- [ ] Significant memory impact

**Backward Compatibility**:
- [ ] Fully backward compatible
- [ ] Minor breaking changes with migration path
- [ ] Major breaking changes (requires version bump)

**Dependencies**:
- [ ] No new dependencies
- [ ] New dependencies added (list in Additional Notes)
- [ ] Dependencies removed/updated

## 🔄 Migration Guide

<!-- If this PR introduces breaking changes, provide migration instructions -->

**Breaking Changes** (if any):
<!-- List any breaking changes -->

**Migration Steps**:
<!-- Provide step-by-step migration instructions -->

**Before/After Examples**:
```python
# Before (old usage)
# old_code_example()

# After (new usage)  
# new_code_example()
```

## ✅ Code Quality Checklist

**Architecture & Design**:
- [ ] Follows established architecture patterns
- [ ] Code is modular and follows separation of concerns
- [ ] Interfaces are well-defined and documented
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate and informative

**Code Standards**:
- [ ] Code follows project style guidelines
- [ ] Variable and function names are descriptive
- [ ] Code is properly commented
- [ ] Complex logic is documented
- [ ] No dead code or unused imports

**Configuration & CLI**:
- [ ] CLI arguments follow established patterns
- [ ] Configuration validation updated (if needed)
- [ ] Schema updated for new config options
- [ ] Backward compatibility maintained for configs

**Security**:
- [ ] No sensitive information hardcoded
- [ ] Input validation performed where needed
- [ ] No security vulnerabilities introduced
- [ ] Dependencies scanned for vulnerabilities

## 📚 Documentation

**Documentation Updates**:
- [ ] Code comments added/updated
- [ ] API documentation updated  
- [ ] User guide/tutorial updated
- [ ] Configuration reference updated
- [ ] CHANGELOG.md updated
- [ ] README updated (if needed)

**Examples Added**:
- [ ] Usage examples in docstrings
- [ ] Configuration examples
- [ ] CLI usage examples
- [ ] Integration examples

## 🚀 Deployment

**Deployment Considerations**:
- [ ] Safe to deploy to all environments
- [ ] Requires environment-specific configuration
- [ ] Needs database migration (if applicable)
- [ ] Requires manual steps (document below)

**Manual Steps Required**:
<!-- List any manual steps needed during deployment -->

## 🔍 Review Guidelines

**For Reviewers**:
- [ ] PR description is clear and complete
- [ ] Code changes align with described functionality
- [ ] Tests are comprehensive and pass
- [ ] Documentation is adequate
- [ ] No obvious security issues
- [ ] Performance impact is acceptable

**Review Focus Areas**:
<!-- Highlight specific areas that need careful review -->

## 📝 Additional Notes

<!-- Any additional context, concerns, or information for reviewers -->

**Open Questions**:
<!-- List any questions or areas where you'd like specific feedback -->

**Future Work**:
<!-- Note any follow-up work or improvements planned -->

**Related PRs**:
<!-- Link any related or dependent PRs -->

---

## 🏁 Final Checklist

Before submitting this PR, confirm:

- [ ] I have followed the [contributing guidelines](CONTRIBUTING.md)
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

