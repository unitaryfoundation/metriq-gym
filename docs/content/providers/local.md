# Local Simulators

Run benchmarks locally using Qiskit Aer simulators. This is useful for development, testing, and running benchmarks without cloud access.

## Prerequisites

No additional setup required - Qiskit Aer is included with Metriq-Gym.

## Discovering Devices

To see available local simulators:

```python
from qbraid.runtime import load_provider

provider = load_provider("local")
for device in provider.get_devices():
    print(f"{device.id}")
```

Common simulators include:

- `aer_simulator` - Ideal state vector simulation
- `aer_simulator_statevector` - State vector simulator (explicit)
- `aer_simulator_density_matrix` - Density matrix simulator

### IBM Noise Model Simulators

Use any IBM device name (e.g., `ibm_sherbrooke`, `ibm_fez`) to run locally with that device's noise model.

!!! note
    Noise model simulators require valid IBM credentials to fetch the noise model, but execution is local.

## Usage

### Ideal Simulation

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider local --device aer_simulator
```

### Noisy Simulation

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider local --device ibm_sherbrooke
```

### Poll Results

```bash
mgym job poll <JOB_ID>
```

Local jobs complete immediately, so polling returns results instantly.

## Configuration

### Cache Directory

Control where simulator results are cached:

```bash
# In .env
MGYM_LOCAL_SIMULATOR_CACHE_DIR="/path/to/cache"
```

Default: Platform-specific cache directory (e.g., `~/Library/Caches/metriq-gym` on macOS)

## Use Cases

### Development and Testing

Test benchmark configurations before running on hardware:

```bash
# Quick validation
mgym job dispatch new_config.json -p local -d aer_simulator
mgym job poll latest
```

### Benchmarking Simulators

Compare ideal vs. noisy simulation:

```bash
# Ideal
mgym job dispatch wit.example.json -p local -d aer_simulator

# With noise
mgym job dispatch wit.example.json -p local -d ibm_sherbrooke
```

### Large-Scale Simulations

For benchmarks requiring many qubits:

```bash
# Use density matrix for mixed states
mgym job dispatch config.json -p local -d aer_simulator_density_matrix
```

## Performance Considerations

### Memory Usage

State vector simulation requires `2^n * 16` bytes of memory for `n` qubits:

| Qubits | Memory |
|--------|--------|
| 20 | ~16 MB |
| 25 | ~512 MB |
| 30 | ~16 GB |
| 35 | ~512 GB |

### GPU Acceleration

Aer supports GPU acceleration via CUDA. If available, simulations automatically use the GPU.

### Parallelization

For benchmarks with multiple circuits, Aer parallelizes across available CPU cores.

## Noise Model Details

When using `ibm_<device>` noise models:

1. Metriq-Gym fetches the current noise model from IBM Quantum
2. The model includes:
   - Single-qubit gate errors
   - Two-qubit gate errors
   - Readout errors
   - T1/T2 decoherence
3. Circuits are transpiled to the device's basis gates
4. Simulation runs locally with the noise model applied

### Noise Model Caching

Noise models are cached to reduce API calls. Clear the cache by deleting:

```bash
rm -rf $MGYM_LOCAL_SIMULATOR_CACHE_DIR
```

## Troubleshooting

### Out of Memory

Reduce qubit count or use a more efficient simulation method:

```bash
# For specific amplitude calculations
mgym job dispatch config.json -p local -d aer_simulator_statevector
```

### Slow Simulation

For deep circuits:
- Reduce shot count for testing
- Use ideal simulation without noise models
- Consider cloud hardware for production runs
