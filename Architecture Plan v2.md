# FUSION Architecture and Migration Plan v2

This document provides an updated architectural plan based on the current state of the FUSION simulator project migration. It addresses gaps identified during implementation review and provides refined guidance for completing the modular, scalable architecture within the target 1-week completion timeline.

---

## 📊 Current State Assessment

### ✅ **Successfully Implemented**
- **Directory Structure**: Core framework is in place (`cli/`, `core/`, `modules/`, `gui/`, `sim/`, `utils/`, `visualization/`, `unity/`)
- **Module Organization**: Routing, spectrum, SNR, RL, and ML modules are properly structured
- **Basic CLI Framework**: CLI entry points and argument parsing structure exists
- **GUI Architecture**: GUI components are modularized and organized
- **Testing Structure**: Test framework is in place with comprehensive coverage

### ❌ **Missing Components to Add**
- **Interfaces Directory**: Abstract base classes for pluggable architecture
- **Config Management System**: Schema validation and configuration handling  
- **I/O Module**: Data generation, import/export, and pipeline management
- **Key Orchestration Files**: `batch_runner.py`, `evaluate_pipeline.py`
- **Developer Tools**: Examples, scripts, and development utilities

---

## 🎯 **Target Architecture**

### **Enhanced Directory Layout**
```
fusion/
├── cli/                      # ✅ CLI entrypoints and args
├── configs/                  # ❌ Config management system
│   ├── schemas/              # Schema validation files
│   ├── templates/            # Default config templates  
│   ├── __init__.py
│   ├── config.py
│   ├── cli_to_config.py
│   └── validate.py
├── core/                     # ✅ Simulation primitives
├── interfaces/               # ❌ Abstract base classes
│   ├── __init__.py
│   ├── router.py             # AbstractRoutingAlgorithm
│   ├── spectrum.py           # AbstractSpectrumAssigner  
│   ├── snr.py                # AbstractSNRMeasurer
│   └── agent.py              # AgentInterface
├── io/                       # ❌ Data management
│   ├── __init__.py
│   ├── generate.py           # Data generation
│   ├── structure.py          # Data structuring
│   └── exporter.py           # Export utilities
├── sim/                      # 🚧 Missing key orchestrators
│   ├── batch_runner.py       # ❌ Main batch execution
│   ├── evaluate_pipeline.py  # ❌ Evaluation workflows
│   ├── ml_pipeline.py        # ✅ Exists
│   ├── train_pipeline.py     # ✅ Exists
│   └── ...
├── modules/                  # ✅ Algorithm modules
├── utils/                    # ✅ Stateless helpers
├── visualization/            # ✅ Plotting and export
├── gui/                      # ✅ GUI components
├── unity/                    # ✅ HPC utilities
├── examples/                 # ❌ Demo workflows
├── scripts/                  # ❌ Development utilities
└── tests/                    # ✅ Pytest suite
```

---

## 🔄 **Module-by-Module Migration Phases** 

### **Phase 2: Core Decoupling & Simulation Pipeline**
**Objective:** Refactor orchestration logic into reusable modules  
- Refactor `engine.py` ➔ `core/simulation.py`  
- Add `core/environment.py` and `core/metrics.py`  
- Refactor `request_generator.py` ➔ `core/request.py`  
- Move `routing.py`, `spectrum_assignment.py`, `snr_measurements.py` to `modules/`  
- Move SDN logic to `core/rerouting.py`  
- Create `sim/batch_runner.py`  
- Add `run_sim.py` wrapper ➔ call `batch_runner`  
**Branch:** `refactor/sim-core`  

### **Phase 3: Reinforcement Learning and ML Modularization**
**Objective:** Move and refactor DRL + ML code into pluggable modules  
- Move `agents/`, `algorithms/`, `feat_extrs/` ➔ `modules/rl/`  
- Add `registry.py`, `train_utils.py`, `sb3_loader.py`  
- Move `model_manager.py` ➔ `model_io/`  
- Refactor `workflow_runner.py` ➔ `train_pipeline.py`  
- Split `envs/` per decision type (path, core, spectrum)  
- Add `agent.py` interface in `interfaces/`  
**Branch:** `refactor/modules-rl`  

### **Phase 4: GUI Refactor**
**Objective:** Decouple GUI from simulation internals  
- Migrate GUI args/helpers/widgets to `gui/`  
- Refactor GUI runner to call `sim/batch_runner.py`  
- Use shared config validator  
- Add smoke test for config loading and simulation  
**Branch:** `refactor/gui`  

### **Phase 5: HPC / Unity Integration**
**Objective:** Encapsulate manifest-based batch pipelines  
- Move all Unity logic to `unity/`  
- Standardize manifest parsing and result fetching  
- Integrate with `sim/batch_runner.py`  
**Branch:** `refactor/unity`  

### **Phase 6: Final Cleanup, Docs, and Testing**
**Objective:** Solidify structure, boost reliability, and document  
- Delete old top-level scripts after migration  
- Add tests:  
  - `test_simulation.py`, `test_batch_runner.py`, `test_train_pipeline.py`  
  - `test_registry.py`, `test_plot_registry.py`  
- Use **Sphinx** or **MkDocs** for documentation  
- Tag release `v1.0`  
**Branch:** `refactor/finalize`  

---

## 🚨 **Critical Implementation Notes**

### **Interface Design Principles**
```python
# Example: Enhanced interface with validation and metadata
class AbstractRoutingAlgorithm(ABC):
    """Base class for all routing algorithms."""
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str: ...
    
    @property  
    @abstractmethod
    def supported_topologies(self) -> List[str]: ...
    
    @abstractmethod
    def validate_environment(self, env) -> bool: ...
    
    @abstractmethod
    def route(self, env, request) -> Optional[Path]: ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]: ...
```

### **Configuration Management**
```python
# configs/config.py - Centralized configuration
class ConfigManager:
    def __init__(self, config_path: str):
        self.schema_validator = SchemaValidator()
        self.config = self.load_and_validate(config_path)
    
    def load_and_validate(self, path: str) -> Dict:
        # Load, validate against schema, return config
        pass
        
    def get_module_config(self, module_name: str) -> Dict:
        # Return validated config for specific module
        pass
```

### **Data Pipeline Architecture**  
```python
# io/exporter.py - Unified data management
class SimulationDataPipeline:
    def __init__(self, config: Dict):
        self.importers = ImporterRegistry()
        self.exporters = ExporterRegistry()
        
    def import_topology(self, source: str) -> NetworkTopology:
        # Unified topology import
        pass
        
    def export_results(self, results: SimResults, format: str) -> None:
        # Multi-format result export
        pass
```

---

## ✅ **Additional Components from Original Plan**

### **Missing Interfaces Directory**
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

### **Missing Config Management System**
```
configs/
├── schemas/
├── templates/
├── config.py
├── cli_to_config.py
└── validate.py
```

### **Missing I/O Module**
```
io/
├── generate.py
├── structure.py
└── exporter.py
```

### **Missing Orchestration Files**
- `sim/batch_runner.py` - Main batch execution
- `sim/evaluate_pipeline.py` - Evaluation workflows

### **Missing Developer Tools**
```
examples/                 # Jupyter workflows or demos
scripts/                  # Dev/test scripts
```

---

## 🚀 **One-Week Completion Strategy**
Phases 2-6 from the original plan should be completed sequentially, module by module, with the additional components integrated as needed for each phase. The modular approach ensures systematic progression while maintaining functionality throughout the refactor.

---

## 🔧 **Best Practices**
- Every module must be:  
  - Registered via a central `registry.py`  
  - Interface-compliant  
  - Unit-tested with Pytest and mock inputs  
- Entry points (`run_sim.py`, etc.) should have no logic  
- All config resolution must pass through `ConfigManager`  
- Legacy code should be temporarily wrapped with adapters (`LegacyEngineWrapper`, etc.)  
- Weekly PR merges to `dev`; only stable releases to `main`  

---

## 🚀 **Final Words**
You now have:  
- A future-proof modular structure  
- A phased roadmap for migration  
- Code-safe practices to minimize disruption  
- Clear ownership, test strategy, and branching  

You're ready to begin — methodically, phase by phase.  
**Let the migration begin.**