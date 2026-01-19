# Simulation Module

## Purpose
The sim module provides simulation orchestration, execution, and pipeline management for the FUSION optical network simulation framework, separating simulation workflow concerns from core simulation logic.

## Key Components

### Core Orchestration
- `batch_runner.py`: Modern batch simulation orchestrator with parallel execution support
- `network_simulator.py`: Legacy multi-process network simulator (deprecated)
- `run_simulation.py`: Compatibility entry points for legacy simulation execution

### Pipeline Management
- `evaluate_pipeline.py`: Evaluation workflow for analyzing simulation results and model performance
- `train_pipeline.py`: RL training pipeline integration with legacy workflow
- `ml_pipeline.py`: Machine learning training pipeline (placeholder)

### Input/Output
- `input_setup.py`: Input data preparation including topology and bandwidth configuration
- `utils.py`: Network analysis, path calculations, spectrum management utilities

## Usage Examples

### Batch Simulation
```python
from fusion.sim.batch_runner import BatchRunner

# Initialize batch runner
config = {
    'erlang': '100,200,300',  # Traffic loads to simulate
    'network': 'NSFNET',
    'max_iters': 1000
}

runner = BatchRunner(config)

# Run simulations sequentially
results = runner.run(parallel=False)

# Run simulations in parallel
results = runner.run(parallel=True, num_processes=4)
```

### Evaluation Pipeline
```python
from fusion.sim.evaluate_pipeline import EvaluationPipeline

config = {
    'model_evaluation': {
        'model_path': '/path/to/model.pkl',
        'test_configs': [test_config1, test_config2]
    },
    'generate_report': True,
    'output_dir': './results'
}

pipeline = EvaluationPipeline(config)
results = pipeline.run_full_evaluation(config)
```

### Input Setup
```python
from fusion.sim.input_setup import create_input

engine_props = {
    'network': 'NSFNET',
    'mod_assumption': 'standard',
    'cores_per_link': 7,
    'thread_num': 's1'
}

# Create all necessary input data
updated_props = create_input(
    base_fp='data',
    engine_props=engine_props
)
```

## Dependencies

### Internal Dependencies
- `fusion.core.simulation`: Core simulation engine
- `fusion.io.structure`: Network topology creation
- `fusion.io.generate`: Bandwidth and physical topology generation
- `fusion.utils.logging_config`: Logging configuration
- `fusion.utils.os`: Operating system utilities

### External Dependencies
- `multiprocessing`: Parallel execution support
- `numpy`: Numerical computations and array operations
- `networkx`: Network topology analysis
- `pathlib`: Path manipulation
- `yaml`: YAML configuration file parsing
