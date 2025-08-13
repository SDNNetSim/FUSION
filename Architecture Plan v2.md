# FUSION Architecture and Migration Plan v2

This document provides an updated architectural plan based on the current state of the FUSION simulator project migration. It addresses gaps identified during implementation review and provides refined guidance for completing the modular, scalable architecture.

---

## 📊 Current State Assessment

### ✅ **Successfully Implemented**
- **Directory Structure**: Core framework is in place (`cli/`, `core/`, `modules/`, `gui/`, `sim/`, `utils/`, `visualization/`, `unity/`)
- **Module Organization**: Routing, spectrum, SNR, RL, and ML modules are properly structured
- **Basic CLI Framework**: CLI entry points and argument parsing structure exists
- **GUI Architecture**: GUI components are modularized and organized
- **Testing Structure**: Test framework is in place with comprehensive coverage

### 🚧 **In Progress / Cleanup Needed** 
- **File Duplication**: Many files have " 2.py" duplicates indicating incomplete migration
- **Module Completion**: Some modules need internal restructuring and cleanup

### ❌ **Critical Missing Components**
- **Interfaces Directory**: Abstract base classes for pluggable architecture
- **Config Management System**: Schema validation and configuration handling  
- **I/O Module**: Data generation, import/export, and pipeline management
- **Key Orchestration Files**: `batch_runner.py`, `evaluate_pipeline.py`
- **Developer Tools**: Examples, scripts, and development utilities

---

## 🎯 **Updated Target Architecture**

### **Enhanced Directory Layout**
```
fusion/
├── cli/                      # ✅ CLI entrypoints and args
├── configs/                  # ❌ CRITICAL: Config management system
│   ├── schemas/              # Schema validation files
│   ├── templates/            # Default config templates  
│   ├── __init__.py
│   ├── config_manager.py     # Central config management
│   ├── schema_validator.py   # YAML/INI validation
│   └── defaults.py           # Default configuration values
├── core/                     # ✅ Simulation primitives
├── interfaces/               # ❌ CRITICAL: Abstract base classes
│   ├── __init__.py
│   ├── router.py             # AbstractRoutingAlgorithm
│   ├── spectrum.py           # AbstractSpectrumAssigner  
│   ├── snr.py                # AbstractSNRMeasurer
│   ├── agent.py              # AgentInterface
│   └── pipeline.py           # AbstractPipeline
├── io/                       # ❌ MISSING: Data management
│   ├── __init__.py
│   ├── data_generator.py     # Simulation data generation
│   ├── importers.py          # Data import utilities
│   ├── exporters.py          # Export to various formats
│   └── pipeline_io.py        # Pipeline data management
├── sim/                      # 🚧 Missing key orchestrators
│   ├── batch_runner.py       # ❌ MISSING: Main batch execution
│   ├── evaluate_pipeline.py  # ❌ MISSING: Evaluation workflows
│   ├── ml_pipeline.py        # ✅ Exists
│   ├── train_pipeline.py     # ✅ Exists
│   └── ...
├── modules/                  # ✅ Algorithm modules (needs cleanup)
├── utils/                    # ✅ Stateless helpers
├── visualization/            # ✅ Plotting and export
├── gui/                      # ✅ GUI components
├── unity/                    # ✅ HPC utilities
├── examples/                 # ❌ MISSING: Demo workflows
├── scripts/                  # ❌ MISSING: Development utilities
└── tests/                    # ✅ Pytest suite
```

---

## 🔄 **Updated Migration Phases** 

### **Phase 1A: Critical Infrastructure (IMMEDIATE PRIORITY)**
**Objective:** Implement missing foundational components for pluggable architecture

**Tasks:**
1. **Create `interfaces/` module** with abstract base classes:
   ```python
   # interfaces/router.py
   class AbstractRoutingAlgorithm(ABC):
       @abstractmethod
       def route(self, env, request) -> Path: ...
       
   # interfaces/spectrum.py  
   class AbstractSpectrumAssigner(ABC):
       @abstractmethod
       def assign(self, env, path, request) -> SpectrumSlots: ...
   ```

2. **Implement `configs/` system**:
   - Schema validation using Pydantic or similar
   - Configuration manager for unified config handling
   - Template system for different simulation scenarios

3. **Add `io/` module** for data management:
   - Unified data import/export interfaces
   - Pipeline data management utilities
   - Support for multiple output formats (JSON, Excel, CSV)

**Branch:** `refactor/critical-infrastructure`  
**Success Criteria:** All modules can be imported and instantiated via interfaces

### **Phase 1B: File Cleanup and Deduplication (HIGH PRIORITY)**  
**Objective:** Remove duplicate files and consolidate migration artifacts

**Tasks:**
1. **Systematic cleanup of " 2.py" files**:
   - Compare original vs duplicate files
   - Merge improvements from duplicates into originals
   - Remove all " 2.py" files after verification

2. **Module consolidation**:
   - Verify all imports work correctly after cleanup
   - Update any remaining references to old file locations
   - Run full test suite to verify no breakage

**Branch:** `refactor/cleanup-duplicates`  
**Success Criteria:** No duplicate files remain, all tests pass

### **Phase 2: Enhanced Orchestration (UPDATED)**
**Objective:** Complete simulation orchestration and pipeline management

**Tasks:**
1. **Implement missing `sim/` orchestrators**:
   - `batch_runner.py` - Main simulation batch execution
   - `evaluate_pipeline.py` - Model and algorithm evaluation workflows
   - Enhanced pipeline coordination

2. **Interface Integration**:
   - Update existing modules to implement new interfaces  
   - Add registry systems that use interface contracts
   - Validate pluggable architecture works end-to-end

**Branch:** `refactor/orchestration`  
**Success Criteria:** Can run full simulation workflows using interface-based architecture

### **Phase 3: Module Modernization (UPDATED)**
**Objective:** Modernize existing modules to use new infrastructure

**Tasks:**
1. **Update all algorithm modules** to implement interfaces
2. **Integrate config management** throughout all modules
3. **Add comprehensive logging** and error handling
4. **Enhance registry systems** with validation and discovery

**Branch:** `refactor/module-modernization`  

### **Phase 4: Developer Experience & Documentation**
**Objective:** Improve developer experience and project maintainability  

**Tasks:**
1. **Create `examples/` directory**:
   - Jupyter notebook tutorials
   - Common simulation workflows  
   - Integration examples

2. **Add `scripts/` directory**:
   - Development utilities
   - Migration helpers
   - Testing scripts

3. **Enhanced documentation**:
   - API documentation with Sphinx
   - Architecture decision records (ADRs)
   - Migration guide for users

**Branch:** `feature/developer-experience`

### **Phase 5: Testing & Quality Assurance (ENHANCED)**
**Objective:** Comprehensive testing and quality assurance

**Tasks:**
1. **Interface testing**: Ensure all implementations satisfy contracts
2. **Integration testing**: End-to-end workflow validation  
3. **Performance testing**: Benchmark key operations
4. **Configuration testing**: Validate schema and templates
5. **Documentation testing**: Verify all examples work

**Branch:** `feature/comprehensive-testing`

### **Phase 6: Production Readiness**
**Objective:** Prepare for production deployment and long-term maintenance

**Tasks:**
1. **Packaging and distribution** setup
2. **CI/CD pipeline** enhancements  
3. **Performance optimization**
4. **Security review**
5. **Release preparation** and versioning

**Branch:** `release/v2.0`

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
# configs/config_manager.py - Centralized configuration
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
# io/pipeline_io.py - Unified data management
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

## 📈 **Migration Success Metrics**

### **Phase 1A Success Criteria:**
- [ ] All interfaces defined and documented
- [ ] Configuration system validates sample configs
- [ ] Basic I/O operations working
- [ ] No import errors in any module

### **Phase 1B Success Criteria:**
- [ ] Zero duplicate " 2.py" files remain
- [ ] Full test suite passes 
- [ ] All imports resolved correctly
- [ ] Documentation reflects current structure

### **Overall Project Success:**
- [ ] **Modularity**: Can swap algorithm implementations via interfaces
- [ ] **Scalability**: Can handle large-scale simulations efficiently  
- [ ] **Maintainability**: Clear separation of concerns, comprehensive tests
- [ ] **Usability**: Simple CLI and GUI interfaces for all workflows
- [ ] **Extensibility**: Easy to add new algorithms and features

---

## 🛠️ **Implementation Priorities**

### **Immediate (Next 1-2 weeks):**
1. Create `interfaces/` directory with core ABCs
2. Clean up duplicate " 2.py" files  
3. Implement basic `configs/` system

### **Short-term (Next month):**
1. Complete `io/` module implementation
2. Add missing `sim/` orchestrators
3. Update existing modules to use interfaces

### **Medium-term (2-3 months):**
1. Comprehensive testing and documentation
2. Developer experience improvements
3. Performance optimization

---

## 🎯 **Next Actions**

1. **Start with Phase 1A** - Create interfaces and config system
2. **Immediately address duplicate files** - This creates technical debt
3. **Focus on core functionality** before adding new features
4. **Maintain test coverage** throughout all changes
5. **Document architecture decisions** as you go

This updated plan reflects the current state of your project and provides a clear path to achieving your goals of a highly organized, functional, and scalable simulator architecture.