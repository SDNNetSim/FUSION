# FUSION Architecture and Migration Plan

This document consolidates the complete, most up-to-date plan for refactoring the FUSION simulator project into a modular, scalable, and maintainable architecture. It merges all prior architectural strategies, file-by-file annotations, phase breakdowns, and directory blueprints into a single authoritative guide.

---

## 🛠️ Target Architecture Overview

### 🔄 Key Design Principles
- **Domain-driven design (DDD):** Orchestrators separate from algorithms  
- **Layered architecture:** `cli/`, `core/`, `sim/`, `modules/`, `interfaces/`, `utils/`, `io/`, `visualization/`  
- **Pluggability:** Routing, spectrum, SNR, and agent modules loaded via registries  
- **Orchestration separation:** Pipelines for batch simulation, training, evaluation  
- **Validation-first configs:** Schema-validated YAML/INI via `configs/schemas/`  
- **Testability:** Unit tests for each decoupled module, isolated with fixtures  

---

## 📁 Final Directory Layout

```
fusion/
├── cli/                      # CLI entrypoints and args
│   ├── run_sim.py
│   ├── run_train.py
│   ├── run_gui.py
│   └── args/	
│       ├── run_sim_args.py
│       ├── run_train_args.py
│       ├── plot_args.py
│       └── common_args.py
│
├── configs/                  # Configs and schemas
│   ├── schemas/
│   ├── templates/
│   ├── config.py
│   ├── cli_to_config.py
│   └── validate.py
│
├── core/                     # Simulation primitives
│   ├── simulation.py
│   ├── environment.py
│   ├── request.py
│   ├── rerouting.py
│   └── metrics.py
│
├── sim/                      # Orchestration workflows
│   ├── batch_runner.py
│   ├── train_pipeline.py
│   ├── evaluate_pipeline.py
│   └── ml_pipeline.py
│
├── modules/                  # Algorithm modules
│   ├── routing/
│   ├── spectrum/
│   ├── snr/
│   ├── rl/
│   │   ├── agents/
│   │   ├── envs/
│   │   ├── feat_extrs/
│   │   ├── runners/
│   │   ├── model_io/
│   │   └── registry.py
│   └── ml/
│       ├── models/
│       ├── train_utils.py
│       └── registry.py
│
├── interfaces/               # ABCs for pluggable modules
│   ├── router.py
│   ├── spectrum.py
│   ├── snr.py
│   └── agent.py
│
├── io/                       # Data generation, structure, export
│   ├── generate.py
│   ├── structure.py
│   └── exporter.py
│
├── utils/                    # Stateless helpers
│   ├── os_helpers.py
│   ├── random_helpers.py
│   └── decorators.py
│
├── visualization/            # Plotting and export
│   ├── plot_stats.py
│   ├── plot_registry.py
│   ├── export_excel.py
│   └── tsv_exporter.py
│
├── gui/
│   ├── main.py
│   ├── widgets/
│   ├── gui_args/
│   ├── gui_helpers/
│   └── runner.py
│
├── unity/                    # HPC job utilities
│   ├── make_manifest.py
│   ├── submit_manifest.py
│   └── fetch_results.py
│
├── tests/                    # Pytest suite
├── scripts/                  # Dev/test scripts
├── examples/                 # Jupyter workflows or demos
├── data/                     # Topologies, modulation formats, etc.
└── README.md
```

---

## 🔄 Migration Phases

### ✅ Phase 1: CLI, Configs, and Visualization
**Objective:** Scaffold structure, move stateless files  
- Move `arg_scripts/` ➔ `cli/args/`  
- Move `plot_scripts/`, `plot_helpers.py`, `rl_excel_stats.py` ➔ `visualization/`  
- Move `os_helpers.py`, `random_helpers.py` ➔ `utils/`  
- Add `args_registry.py` to centralize CLI parsing  
- Copy `parse_args.py` ➔ `main_parser.py`  
- Add test coverage: `test_cli_args.py`, `test_plot_imports.py`  
- **Git branch:** `refactor/scaffold`  

### 🔴 Phase 2: Core Decoupling & Simulation Pipeline
**Objective:** Refactor orchestration logic into reusable modules  
- Refactor `engine.py` ➔ `core/simulation.py`  
- Add `core/environment.py` and `core/metrics.py`  
- Refactor `request_generator.py` ➔ `core/request.py`  
- Move `routing.py`, `spectrum_assignment.py`, `snr_measurements.py` to `modules/`  
- Move SDN logic to `core/rerouting.py`  
- Create `sim/batch_runner.py`  
- Add `run_sim.py` wrapper ➔ call `batch_runner`  
- **Git branch:** `refactor/sim-core`  

### 📈 Phase 3: Reinforcement Learning and ML Modularization
**Objective:** Move and refactor DRL + ML code into pluggable modules  
- Move `agents/`, `algorithms/`, `feat_extrs/` ➔ `modules/rl/`  
- Add `registry.py`, `train_utils.py`, `sb3_loader.py`  
- Move `model_manager.py` ➔ `model_io/`  
- Refactor `workflow_runner.py` ➔ `train_pipeline.py`  
- Split `envs/` per decision type (path, core, spectrum)  
- Add `agent.py` interface in `interfaces/`  
- **Git branch:** `refactor/modules-rl`  

### 🚧 Phase 4: GUI Refactor
**Objective:** Decouple GUI from simulation internals  
- Migrate GUI args/helpers/widgets to `gui/`  
- Refactor GUI runner to call `sim/batch_runner.py`  
- Use shared config validator  
- Add smoke test for config loading and simulation  
- **Git branch:** `refactor/gui`  

### ⚡️ Phase 5: HPC / Unity Integration
**Objective:** Encapsulate manifest-based batch pipelines  
- Move all Unity logic to `unity/`  
- Standardize manifest parsing and result fetching  
- Integrate with `sim/batch_runner.py`  
- **Git branch:** `refactor/unity`  

### 🌐 Phase 6: Final Cleanup, Docs, and Testing
**Objective:** Solidify structure, boost reliability, and document  
- Delete old top-level scripts after migration  
- Add tests:  
  - `test_simulation.py`, `test_batch_runner.py`, `test_train_pipeline.py`  
  - `test_registry.py`, `test_plot_registry.py`  
- Use **Sphinx** or **MkDocs** for documentation  
- Tag release `v1.0`  
- **Git branch:** `refactor/finalize`  

---

## ✅ Checklist Summary

| Phase | Description | Branch | Status |
|-------|-------------|--------|--------|
| 0 | Planning & Scaffolding | main, dev | ✅ Done |
| 1 | CLI + Helpers Migration | refactor/scaffold | ⏳ In Progress |
| 2 | Core Refactor + Pipelines | refactor/sim-core | ⏳ Upcoming |
| 3 | DRL & ML Modularization | refactor/modules-rl | ⏳ Upcoming |
| 4 | GUI Integration | refactor/gui | ⏳ Upcoming |
| 5 | Unity Integration | refactor/unity | ⏳ Upcoming |
| 6 | Cleanup + Testing + Docs | refactor/finalize | ⏳ Upcoming |

---

## 🔧 Best Practices
- Every module must be:  
  - Registered via a central `registry.py`  
  - Interface-compliant  
  - Unit-tested with Pytest and mock inputs  
- Entry points (`run_sim.py`, etc.) should have no logic  
- All config resolution must pass through `ConfigManager`  
- Legacy code should be temporarily wrapped with adapters (`LegacyEngineWrapper`, etc.)  
- Weekly PR merges to `dev`; only stable releases to `main`  

---

## 🔒 Example Base Interfaces

```python
# interfaces/router.py
class AbstractRoutingAlgorithm(ABC):
    @abstractmethod
    def route(self, env, request): ...

# interfaces/spectrum.py
class AbstractSpectrumAssigner(ABC):
    @abstractmethod
    def assign(self, env, path, request): ...

# interfaces/agent.py
class AgentInterface(ABC):
    @abstractmethod
    def act(self, observation): ...
    @abstractmethod
    def train(self, env): ...
    @abstractmethod
    def save(self, path): ...
    @abstractmethod
    def load(self, path): ...
```

---

## 🚀 Final Words
You now have:  
- A future-proof modular structure  
- A phased roadmap for migration  
- Code-safe practices to minimize disruption  
- Clear 0wnership, test strategy, and branching  

You're ready to begin — methodically, phase by phase.  
**Let the migration begin.**
