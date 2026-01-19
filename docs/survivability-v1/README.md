# FUSION v1 Survivability Extensions Documentation

This directory contains the complete specification for implementing survivability and offline RL capabilities in the FUSION Elastic Optical Network simulator.

## Document Organization

The specification is organized into **7 logical phases** to facilitate incremental development and verification:

### Phase 1: Foundation & Setup
Understanding project context, scope boundaries, and development workflow.

- [00-overview.md](phase1-foundation/00-overview.md) - Project Context & Integration Points
- [01-scope-boundaries.md](phase1-foundation/01-scope-boundaries.md) - SHALL NOT, Nice-to-Have, Out of Scope
- [02-module-summary.md](phase1-foundation/02-module-summary.md) - Module-by-Module Summary
- [03-version-control.md](phase1-foundation/03-version-control.md) - Git Workflow & Branching Strategy

### Phase 2: Core Infrastructure
Building the foundational components for failure handling and path management.

- [10-failure-module.md](phase2-infrastructure/10-failure-module.md) - Failure/Disaster Module (F1, F3, F4)
- [11-k-path-cache.md](phase2-infrastructure/11-k-path-cache.md) - K-Path Candidate Generation & Caching
- [12-configuration.md](phase2-infrastructure/12-configuration.md) - Configuration System Integration
- [13-determinism-seeds.md](phase2-infrastructure/13-determinism-seeds.md) - Determinism & Seed Management

### Phase 3: Protection & Recovery
Implementing protection mechanisms and recovery time modeling.

- [20-protection.md](phase3-protection/20-protection.md) - 1+1 Disjoint Protection + Restoration
- [21-recovery-timing.md](phase3-protection/21-recovery-timing.md) - Recovery Time Modeling (Emulated SDN)

### Phase 4: RL Integration
Adding reinforcement learning policy support and dataset generation.

- [30-rl-policies.md](phase4-rl-integration/30-rl-policies.md) - RL Policy Integration (Offline Inference)
- [31-dataset-logging.md](phase4-rl-integration/31-dataset-logging.md) - Offline Dataset Logging

### Phase 5: Metrics & Reporting
Implementing comprehensive metrics collection and reporting.

- [40-metrics-reporting.md](phase5-metrics/40-metrics-reporting.md) - Metrics & Reporting System

### Phase 6: Quality Assurance
Ensuring code quality, test coverage, and performance standards.

- [50-testing.md](phase6-quality/50-testing.md) - Testing Requirements & Standards
- [51-documentation.md](phase6-quality/51-documentation.md) - Documentation Requirements
- [52-performance.md](phase6-quality/52-performance.md) - Performance Budgets & Constraints

### Phase 7: Project Management
Project planning, risk management, and traceability.

- [60-work-breakdown.md](phase7-management/60-work-breakdown.md) - Minimal Work Breakdown (13-17 days)
- [61-risks-mitigations.md](phase7-management/61-risks-mitigations.md) - Risks & Mitigations
- [62-traceability.md](phase7-management/62-traceability.md) - Traceability to Paper Claims
- [63-usage-workflow.md](phase7-management/63-usage-workflow.md) - Example Usage Workflow
- [64-checklist.md](phase7-management/64-checklist.md) - Final Implementation Checklist

## High-Level Goals

Enable stress-testing KSP-FF, 1+1 protection, and an **offline RL policy (BC → IQL)** with **action masking + heuristic fallback** under **F1 (link), F3 (SRLG), F4 (geo radius=2)** failures, measuring:

- **Blocking Probability (BP)** overall and within failure windows
- **Recovery Time** (mean, P95) for protection and restoration
- **Fragmentation** proxy metrics
- **Seed Variance** for statistical significance

## Development Workflow

1. **Start with Phase 1** to understand context and scope
2. **Follow Phases 2-5** for implementation (order matters due to dependencies)
3. **Use Phase 6** throughout development for quality checks
4. **Refer to Phase 7** for project management and tracking

## Estimated Timeline

**Total: 13-17 days** (see [60-work-breakdown.md](phase7-management/60-work-breakdown.md))

## Prerequisites

- FUSION v6.0.0+ installed and configured
- Python 3.9+, PyTorch, NetworkX, Stable-Baselines3
- Familiarity with FUSION's architecture (see [00-overview.md](phase1-foundation/00-overview.md))

## Quick Start

```bash
# 1. Review foundation documents
cd docs/survivability-v1/phase1-foundation
cat 00-overview.md 01-scope-boundaries.md

# 2. Begin implementation with failures module
cd ../../fusion/modules
# Follow phase2-infrastructure/10-failure-module.md

# 3. Run tests as you implement
pytest fusion/modules/failures/tests/ -v --cov

# 4. Refer to quality assurance docs
cd ../../docs/survivability-v1/phase6-quality
```

## Key Architecture Principles

1. **Minimal Invasiveness**: Extend existing modules, don't replace them
2. **Registry Pattern**: Use FUSION's registry system for multi-component modules
3. **Type Safety**: Full type hints on all functions and parameters
4. **Test Coverage**: 80-90% target for all new modules
5. **Determinism**: All experiments fully reproducible with seed control

## Contact & Support

For questions about this specification:
- Review FUSION's [CODING_STANDARDS.md](../../CODING_STANDARDS.md)
- Check [TESTING_STANDARDS.md](../../TESTING_STANDARDS.md)
- Refer to [DEVELOPMENT_WORKFLOW.md](../../DEVELOPMENT_WORKFLOW.md)

---

**Version**: v2 (Contextualized to FUSION Architecture)
**Last Updated**: 2025-10-14
**Status**: Specification Complete, Implementation Pending
