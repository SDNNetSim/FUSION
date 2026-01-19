# SNR Module

Signal-to-Noise Ratio (SNR) measurement algorithms for optical network quality assessment in FUSION.

## Overview

This module provides comprehensive SNR measurement capabilities for optical networks, including support for both single-core and multi-core fiber systems. It implements various noise models and interference calculations to accurately assess signal quality across different network topologies.

## Features

- **Multiple Noise Models**: ASE (Amplified Spontaneous Emission), SCI (Self-Channel Interference), XCI (Cross-Channel Interference), and cross-talk modeling
- **Multi-core Fiber Support**: Cross-talk calculations for multi-core fiber systems
- **Modular Architecture**: Registry-based algorithm selection and management
- **Comprehensive Metrics**: Performance tracking and algorithm comparison capabilities
- **Type Safety**: Full mypy compliance with modern Python type hints

## Module Structure

```
snr/
├── __init__.py          # Module exports and public API
├── registry.py          # Algorithm registry and management
├── snr.py              # Standard SNR measurement implementation
├── utils.py            # Utility functions for file loading and response computation
└── README.md           # This documentation
```

## Components

### SNRRegistry

Central registry for managing SNR measurement algorithm implementations.

**Key Features:**
- Dynamic algorithm registration and discovery
- Type-safe algorithm instantiation
- Multi-core algorithm filtering
- Algorithm metadata and information retrieval

### StandardSNRMeasurer

Primary SNR measurement algorithm implementing comprehensive noise modeling.

**Supported Calculations:**
- **Linear Noise**: ASE noise from optical amplifiers
- **Nonlinear Noise**: SCI and XCI modeling with dispersion effects
- **Cross-talk Noise**: Multi-core fiber interference calculations
- **Modulation Thresholds**: SNR requirements for different modulation formats

**Noise Models:**
- ASE calculation with EDFA noise figure consideration
- SCI using power spectral density and dispersion parameters
- XCI based on interfering channel analysis
- Cross-talk modeling for adjacent core interference

### Utility Functions

- **File Loading**: Modulation format and GSNR data retrieval
- **Slot Indexing**: Band-aware spectrum slot calculations
- **Response Validation**: SNR threshold and modulation validation

## Usage

### Basic SNR Calculation

```python
from fusion.modules.snr import create_snr_algorithm

# Create SNR measurement instance
snr_measurer = create_snr_algorithm(
    "standard_snr",
    engine_props,
    sdn_props,
    spectrum_props,
    route_props
)

# Calculate SNR for a path
snr_db = snr_measurer.calculate_snr(path, spectrum_info)
```

### Algorithm Discovery

```python
from fusion.modules.snr import list_snr_algorithms, get_snr_algorithm_info

# List available algorithms
algorithms = list_snr_algorithms()

# Get algorithm details
info = get_snr_algorithm_info("standard_snr")
print(f"Supports multi-core: {info['supports_multicore']}")
```

### Multi-core Support

```python
from fusion.modules.snr import get_multicore_snr_algorithms

# Get algorithms supporting multi-core fiber
multicore_algos = get_multicore_snr_algorithms()

# Calculate with cross-talk consideration
spectrum_info = {
    "start_slot": 10,
    "end_slot": 15,
    "core_num": 1,
    "band": "c"
}
snr_db = snr_measurer.calculate_snr(path, spectrum_info)
```

## Algorithm Parameters

### Engine Properties
- `bw_per_slot`: Bandwidth per spectrum slot (Hz)
- `input_power`: Signal input power (W)
- `fiber_attenuation`: Fiber attenuation coefficient (dB/km)
- `fiber_dispersion`: Chromatic dispersion (ps/nm/km)
- `nonlinear_coefficient`: Nonlinear coefficient (1/W/km)
- `edfa_noise_figure`: EDFA noise figure (dB)
- `cores_per_link`: Number of cores in multi-core fiber
- `xt_coefficient`: Cross-talk coefficient (dB)

### Spectrum Information
- `start_slot`: Starting spectrum slot index
- `end_slot`: Ending spectrum slot index
- `core_number`: Core identifier for multi-core systems
- `band`: Spectrum band ('l', 'c', 's')

## Modulation Support

Supported modulation formats with SNR thresholds:
- **BPSK**: 6.0 dB
- **QPSK**: 9.0 dB
- **8QAM**: 12.0 dB
- **16QAM**: 15.0 dB
- **32QAM**: 18.0 dB
- **64QAM**: 21.0 dB

Thresholds include automatic reach penalty calculation (0.1 dB per 100 km).

## Performance Metrics

The module tracks comprehensive performance metrics:
- Total calculations performed
- Average SNR computed
- Algorithm execution statistics
- Noise model contributions

## Dependencies

- **Python 3.11+**: Modern type hint support
- **NumPy**: Optional for enhanced array operations
- **fusion.core**: Properties and configuration management
- **fusion.interfaces**: Abstract base classes

## Type Safety

This module is fully compliant with mypy static type checking:
- Complete type annotations for all functions and methods
- Proper handling of optional dependencies (NumPy)
- Safe attribute access patterns
- Modern Python type hint syntax

## Testing

SNR calculations can be validated using:
- Known reference scenarios with expected SNR values
- Cross-validation between different noise models
- Performance benchmarking against analytical solutions
- Multi-core interference validation

## Contributing

When extending this module:
1. Implement the `AbstractSNRMeasurer` interface
2. Register new algorithms in the SNRRegistry
3. Follow the established docstring format (`:param:`, `:return:`)
4. Ensure full type annotation coverage
5. Add comprehensive unit tests
6. Update this documentation

## See Also

- `fusion.interfaces.snr`: Abstract base classes
- `fusion.core.properties`: Configuration management
- `fusion.modules.routing`: Path calculation integration
- `fusion.modules.spectrum`: Spectrum assignment coordination
