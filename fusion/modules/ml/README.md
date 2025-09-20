# Machine Learning Module

⚠️ **IMPORTANT NOTICE** ⚠️

**This module is currently not actively maintained or supported.** The ML functionality has not been used in recent development cycles and may require updates to work with the current version of FUSION. Use at your own risk and expect that you may need to make modifications for your specific use case.

---

## Overview

The Machine Learning module provides capabilities for:
- Training models for network optimization decisions
- Feature engineering from network states
- Model evaluation and comparison
- Visualization of results

## Module Structure

```
modules/ml/
├── README.md               # This file
├── ML_STANDARDS.md         # Module-specific coding standards
├── __init__.py            # Public API exports
├── registry.py            # Model registry (empty)
├── visualization.py       # Plotting and visualization functions
├── preprocessing.py       # Data preparation utilities
├── feature_engineering.py # Feature extraction and creation
├── model_io.py           # Model saving/loading
└── evaluation.py         # Model evaluation metrics
```

## Current State

### What Works
- Basic structure and organization following FUSION coding standards
- Function interfaces are well-defined with proper type hints
- Documentation is comprehensive

### Known Limitations
- No active test coverage
- May have compatibility issues with current simulation engine
- Dependencies on older versions of ML libraries
- Limited model types currently implemented

## Usage (Experimental)

If you choose to use this module, here's a basic example:

```python
from fusion.modules.ml import (
    extract_ml_features,
    process_training_data,
    save_model,
    load_model
)

# Extract features from network request
features = extract_ml_features(request_dict, engine_props, sdn_props)

# Process training data
processed_data = process_training_data(sim_dict, raw_data, erlang)

# Train your model (not provided by this module)
# model = train_your_model(processed_data)

# Save the trained model
save_model(sim_dict, model, "algorithm_name", "1000")
```

## Future Development

This module serves as a foundation for ML integration but requires:
1. Updated compatibility testing with current FUSION version
2. Implementation of actual ML algorithms in registry.py
3. Integration tests with the simulation engine
4. Performance benchmarking
5. Example notebooks demonstrating usage

## Contributing

If you're interested in reviving this module:
1. Check ML_STANDARDS.md for coding guidelines
2. Ensure compatibility with current FUSION architecture
3. Add comprehensive tests
4. Update this README when functionality is restored

## Support

This module is provided as-is without active support. For questions about FUSION's current capabilities, please refer to the main documentation or active modules.