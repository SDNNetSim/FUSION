# Architecture Overview

This document provides a high-level view of the V4 FUSION architecture.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SimulationEngine                                │
│                                                                              │
│  ┌──────────────────┐    ┌───────────────────────────────────────────────┐ │
│  │ SimulationConfig │    │                  NetworkState                  │ │
│  │    (frozen)      │    │  (single instance, owned by engine)           │ │
│  └──────────────────┘    │  - topology graph                             │ │
│                          │  - spectrum allocation                         │ │
│                          │  - lightpath registry                          │ │
│                          └───────────────────────────────────────────────┘ │
│                                          │                                  │
│                                          │ passed by reference              │
│                                          ▼                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          SDNOrchestrator                               │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │  Routing    │  │  Spectrum   │  │  Grooming   │  │    SNR      │  │ │
│  │  │  Pipeline   │  │  Pipeline   │  │  Pipeline   │  │  Pipeline   │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐                                     │ │
│  │  │  Slicing    │  │ Protection  │                                     │ │
│  │  │  Pipeline   │  │  Pipeline   │                                     │ │
│  │  └─────────────┘  └─────────────┘                                     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                          │                                  │
│                                          │ AllocationResult                 │
│                                          ▼                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          StatsCollector                                │ │
│  │  - blocking probability                                                │ │
│  │  - modulation usage                                                    │ │
│  │  - feature counters                                                    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### SimulationConfig

Immutable configuration object created once at simulation start.

- **Location**: `fusion/domain/config.py`
- **Responsibility**: Hold all simulation parameters
- **Lifecycle**: Created once, never modified

### NetworkState

Single source of truth for mutable network state.

- **Location**: `fusion/domain/network_state.py`
- **Responsibility**: Manage topology, spectrum, and lightpaths
- **Lifecycle**: Owned by SimulationEngine, passed by reference

### SDNOrchestrator

Coordinates pipeline stages for request processing.

- **Location**: `fusion/core/orchestrator.py`
- **Responsibility**: Sequence pipelines, handle failures, combine results
- **Lifecycle**: Created once, stateless per-request processing

### Pipelines

Individual processing stages with clear input/output contracts.

| Pipeline | Input | Output | Responsibility |
|----------|-------|--------|----------------|
| Routing | source, dest, bandwidth | `RouteResult` | Find candidate paths |
| Spectrum | path, modulations | `SpectrumResult` | Assign spectrum slots |
| Grooming | request | `GroomingResult` | Reuse existing lightpaths |
| SNR | lightpath | `SNRResult` | Validate signal quality |
| Slicing | request, path | `SlicingResult` | Split large requests |
| Protection | request | Protected allocation | Handle 1+1 protection |

### StatsCollector

Aggregates simulation statistics.

- **Location**: `fusion/stats/collector.py`
- **Responsibility**: Track metrics, compute KPIs
- **Lifecycle**: Created once, updated per-request

## Data Flow

### Request Arrival

```
Request arrives
      │
      ▼
┌─────────────────┐
│ Try Grooming?   │──── yes ──► GroomingPipeline
│ (if enabled)    │                    │
└────────┬────────┘                    ▼
         │                      ┌──────────────┐
         │                      │Fully groomed?│── yes ──► Done
         │                      └──────┬───────┘
         │                             │ no
         │◄────────────────────────────┘
         ▼
┌─────────────────┐
│RoutingPipeline  │──► RouteResult (paths, modulations)
└────────┬────────┘
         │
         ▼ for each path
┌─────────────────┐
│SpectrumPipeline │──► SpectrumResult (slots, core, band)
└────────┬────────┘
         │
         ▼ if spectrum found
┌─────────────────┐
│NetworkState     │──► create_lightpath()
│.create_lightpath│
└────────┬────────┘
         │
         ▼ if SNR enabled
┌─────────────────┐
│ SNRPipeline     │──► SNRResult (pass/fail)
└────────┬────────┘
         │
         ▼ if failed, try next path or slicing
┌─────────────────┐
│AllocationResult │──► success or block_reason
└─────────────────┘
```

### Request Release

```
Request departs
      │
      ▼
For each lightpath_id in request:
      │
      ▼
┌─────────────────┐
│ Remove request  │
│ from lightpath  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ If no requests  │── yes ──► NetworkState.release_lightpath()
│ remain?         │
└─────────────────┘
```

## Key Design Decisions

### 1. Single NetworkState Instance

All components receive `NetworkState` by reference. No caching or copying allowed.

**Rationale**: Prevents stale reads and lost writes from divergent state.

### 2. Immutable Result Objects

All pipeline results are frozen dataclasses.

**Rationale**: Results are shared between components; immutability prevents accidents.

### 3. Pipeline Composition

Pipelines are independent and composable. Orchestrator decides sequencing.

**Rationale**: Easy to add/remove features, test in isolation.

### 4. Explicit Feature Flags

Each feature (grooming, slicing, SNR) has an explicit boolean flag.

**Rationale**: Clear configuration, no magic value checks.

## Related Documents

- [Domain Model](domain_model.md) - Core dataclasses
- [Result Objects](result_objects.md) - Pipeline outputs
- [NetworkState](network_state.md) - State management (Phase 2)
- [Pipelines](pipelines.md) - Pipeline protocols (Phase 2)
- [Orchestration](orchestration.md) - Orchestrator rules (Phase 3)
