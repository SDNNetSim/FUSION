# FUSION Architecture

## Overview

FUSION (Flexible Unified System for Intelligent Optical Networking) is a modular simulation framework designed for Software Defined Elastic Optical Networks (SD-EONs) with extensibility for other networking paradigms. The architecture emphasizes modularity, configurability, and integration with modern AI/ML techniques.

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Interface                         │
│                    (fusion.cli.run_sim)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   Configuration System                       │
│           (INI-based with validation & templates)            │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Simulation Core Engine                    │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Network    │  │   Traffic    │  │   Spectrum      │   │
│  │   Topology   │  │   Generator  │  │   Management    │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐  ┌─────────▼────────┐  ┌──────▼──────────┐
│   Routing    │  │   Survivability   │  │   RL/AI         │
│   Policies   │  │   & Protection    │  │   Integration   │
│  (KSP-FF,    │  │  (Failure Mgmt)   │  │  (BC, IQL,      │
│   RL-based)  │  │                   │  │   Action Mask)  │
└──────────────┘  └──────────────────┘  └─────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              Metrics Collection & Analysis                   │
│    (Blocking, Recovery Time, Fragmentation, etc.)            │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System
**Location:** `fusion/configs/`

- **INI-based Configuration**: Flexible, human-readable configuration files
- **Template System**: Pre-defined templates for common experiment types
- **Validation**: Type checking and parameter validation
- **Documentation**: Comprehensive configuration guide in `fusion/configs/README.md`

**Key Features:**
- Hierarchical configuration sections
- Environment-specific overrides
- CLI argument integration
- Default value management

### 2. Simulation Engine
**Location:** `fusion/core/`

The simulation engine orchestrates the discrete event simulation of network requests:

- **Event Loop**: Discrete event simulation with time-ordered event processing
- **State Management**: Tracks network state, spectrum allocation, active connections
- **Request Processing**: Handles connection requests (arrival/departure)
- **Resource Allocation**: Manages spectrum slots, routing paths, and wavelength assignment

### 3. Network Topology
**Location:** `fusion/modules/topology/`

- **Graph-based Representation**: NetworkX-based topology modeling
- **Standard Topologies**: Support for common network topologies (NSFNet, COST239, etc.)
- **Custom Topologies**: JSON/GraphML import capabilities
- **Link Properties**: Distance, available spectrum, SRLG assignments

### 4. Routing & Spectrum Assignment
**Location:** `fusion/modules/routing/`, `fusion/modules/rsa/`

**Routing Algorithms:**
- K-Shortest Path (KSP)
- Dijkstra's shortest path
- Constrained routing (SRLG-disjoint, geographic-aware)

**Spectrum Assignment:**
- First-Fit (FF)
- Best-Fit
- Random-Fit
- ML-based policies

### 5. Survivability & Protection
**Location:** `fusion/modules/failures/`, `fusion/modules/protection/`

**Failure Types:**
- **F1 (Link Failure)**: Single link disruption
- **F2 (Node Failure)**: Node and adjacent links
- **F3 (SRLG)**: Shared Risk Link Group failures
- **F4 (Geographic)**: Regional disaster scenarios

**Protection Mechanisms:**
- **1+1 Protection**: Disjoint path protection
- **Rerouting**: Dynamic path re-establishment
- **Switchover**: Configurable recovery times

**Key Classes:**
- `FailureManager`: Orchestrates failure injection and recovery
- `FailureScenario`: Defines failure parameters
- `RecoveryMetrics`: Tracks recovery performance

### 6. Reinforcement Learning Integration
**Location:** `fusion/modules/rl/`

**Supported Algorithms:**
- **Behavioral Cloning (BC)**: Offline imitation learning
- **Implicit Q-Learning (IQL)**: Offline RL with conservative Q-estimates
- **Online RL**: Integration with Stable-Baselines3 (PPO, A2C, DQN)

**Features:**
- **Action Masking**: Invalid action filtering
- **State Representation**: Graph-based network state encoding
- **Reward Shaping**: Customizable reward functions
- **Policy Evaluation**: Baseline comparison (KSP-FF)

**Key Components:**
- `RLPolicy`: Abstract base for RL-based routing
- `BCPolicy`, `IQLPolicy`: Offline RL implementations
- `OfflineRLDataset`: JSONL dataset generation for training

### 7. Metrics & Analysis
**Location:** `fusion/modules/metrics/`

**Performance Metrics:**
- **Blocking Probability**: Overall and time-windowed
- **Recovery Time**: Mean, P95, max recovery latency
- **Spectrum Fragmentation**: External fragmentation index
- **Decision Time**: Policy inference latency
- **Utilization**: Link and spectrum usage

**Output Formats:**
- CSV exports
- JSON structured data
- Real-time console logging
- Integration with visualization tools

## Data Flow

### Typical Simulation Flow

1. **Initialization**
   - Load configuration from INI file
   - Parse CLI arguments and override settings
   - Initialize network topology
   - Set up failure scenarios (if enabled)
   - Load RL models (if specified)

2. **Simulation Loop**
   - Generate traffic requests (Poisson arrival process)
   - For each request:
     - Query routing policy (baseline or RL)
     - Attempt spectrum assignment
     - Update network state
     - Log metrics
   - Inject failures at specified times
   - Trigger protection/recovery mechanisms

3. **Termination**
   - Aggregate metrics
   - Export results to CSV/JSON
   - Generate summary reports
   - Save offline RL datasets (if enabled)

## Directory Structure

```
FUSION/
├── fusion/                      # Main package
│   ├── cli/                     # Command-line interface
│   │   └── run_sim.py           # Primary entry point
│   ├── configs/                 # Configuration system
│   │   ├── templates/           # Template INI files
│   │   └── README.md            # Configuration guide
│   ├── core/                    # Simulation engine
│   ├── modules/                 # Feature modules
│   │   ├── failures/            # Failure injection
│   │   ├── protection/          # Protection mechanisms
│   │   ├── routing/             # Routing algorithms
│   │   ├── rsa/                 # Spectrum assignment
│   │   ├── rl/                  # RL integration
│   │   ├── topology/            # Network graphs
│   │   └── metrics/             # Performance tracking
│   ├── models/                  # Trained RL models
│   └── gui/                     # GUI (deprecated)
├── tests/                       # Unit and integration tests
├── docs/                        # Sphinx documentation
├── data/                        # Datasets and results
└── scripts/                     # Utility scripts
```

## Extension Points

### Adding New Routing Policies

1. Inherit from `BaseRoutingPolicy`
2. Implement `select_route(request, network_state)` method
3. Register in configuration system
4. Add unit tests in `tests/modules/routing/`

### Adding New Failure Types

1. Define failure scenario in `FailureScenario` class
2. Implement failure injection logic in `FailureManager`
3. Add recovery hooks for protection mechanisms
4. Update configuration templates

### Adding New RL Algorithms

1. Inherit from `RLPolicy` base class
2. Implement `predict(observation)` method
3. Add model loading/saving logic
4. Integrate with action masking system
5. Add to policy registry

## Technology Stack

- **Language**: Python 3.11+
- **Core Libraries**:
  - NumPy, Pandas: Numerical computation and data handling
  - NetworkX: Graph-based topology representation
  - PyTorch: Deep learning and RL model implementation
  - PyTorch Geometric: Graph neural networks
- **RL Framework**: Stable-Baselines3
- **Testing**: pytest, unittest
- **Documentation**: Sphinx (reStructuredText)
- **Code Quality**: Ruff (linting/formatting), Mypy (type checking)

## Design Principles

1. **Modularity**: Components are loosely coupled and independently testable
2. **Configurability**: Extensive parameterization via INI files and CLI
3. **Extensibility**: Clear interfaces for adding new algorithms and features
4. **Reproducibility**: Seed-based random number generation for deterministic results
5. **Performance**: Efficient discrete event simulation with minimal overhead
6. **Documentation**: Comprehensive guides for users and developers

## Future Enhancements

- **Distributed Simulation**: Multi-process simulation for large-scale experiments
- **Real-time Visualization**: Web-based dashboard for live monitoring
- **Database Integration**: Persistent storage for experiment results
- **API Server**: REST API for remote experiment execution
- **Enhanced RL**: Support for multi-agent RL and hierarchical policies

## References

For detailed documentation on specific components:
- [Configuration Guide](fusion/configs/README.md)
- [Survivability Documentation](docs/survivability-v1/README.md)
- [Testing Standards](TESTING_STANDARDS.md)
- [Coding Standards](CODING_STANDARDS.md)
- [Development Workflow](DEVELOPMENT_WORKFLOW.md)
