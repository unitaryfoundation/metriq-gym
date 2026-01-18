# Benchmarks Overview

Metriq-Gym provides a comprehensive suite of quantum benchmarks to characterize and compare quantum hardware performance.

## Available Benchmarks

| Benchmark | Category | Description |
|-----------|----------|-------------|
| [BSEQ](bseq.md) | Connectivity | Binary Sequence - measures device connectivity |
| [CLOPS](clops.md) | Throughput | Circuit Layer Operations Per Second |
| [EPLG](eplg.md) | Error Rate | Error Per Layered Gate |
| [Mirror Circuits](mirror-circuits.md) | Error Rate | Randomized mirror circuit benchmarking |
| [QML Kernel](qml-kernel.md) | Application | Quantum Machine Learning kernel estimation |
| [Quantum Volume](quantum-volume.md) | Holistic | Overall device capability metric |
| [WIT](wit.md) | Application | Wormhole-Inspired Teleportation |
| [LR-QAOA](lr-qaoa.md) | Application | Linear Ramp QAOA for MaxCut |
| [QED-C Benchmarks](qedc.md) | Algorithm | Bernstein-Vazirani, Phase Estimation, Hidden Shift, QFT |

## Benchmark Categories

### Connectivity Benchmarks

Test device qubit connectivity and topology:
- **BSEQ**: Measures the largest connected subgraph of working qubits

### Throughput Benchmarks

Measure execution speed and efficiency:
- **CLOPS**: Measures how many circuit layer operations can be executed per second

### Error Rate Benchmarks

Characterize noise and error rates:
- **EPLG**: Measures error per layered gate using randomized benchmarking
- **Mirror Circuits**: Uses self-inverting circuits to measure total circuit fidelity

### Holistic Benchmarks

Overall device capability metrics:
- **Quantum Volume**: Standard metric combining qubit count and error rates

### Application Benchmarks

Test performance on specific quantum algorithms:
- **WIT**: Wormhole-inspired teleportation fidelity
- **QML Kernel**: Quantum kernel estimation for machine learning
- **LR-QAOA**: Linear ramp QAOA for combinatorial optimization

### Algorithm Benchmarks

Standard quantum algorithm implementations:
- **QED-C Benchmarks**: Suite including Bernstein-Vazirani, Phase Estimation, Hidden Shift, and QFT

## Configuration Format

All benchmarks use JSON configuration files:

```json
{
  "benchmark_name": "Quantum Volume",
  "num_qubits": 5,
  "shots": 1000,
  "trials": 100
}
```

### Common Parameters

| Parameter | Description |
|-----------|-------------|
| `benchmark_name` | Required benchmark identifier |
| `shots` | Number of measurement repetitions |
| `num_qubits` | Number of qubits to use |

### Example Configurations

Pre-made configurations are available in `metriq_gym/schemas/examples/`:

```bash
ls metriq_gym/schemas/examples/
```

## Running Benchmarks

### Single Benchmark

```bash
mgym job dispatch metriq_gym/schemas/examples/quantum_volume.example.json \
    --provider local --device aer_simulator
```

### Benchmark Suite

Create a suite file to run multiple benchmarks:

```json
{
  "name": "performance_suite",
  "benchmarks": [
    {"name": "qv", "config": {"benchmark_name": "Quantum Volume", "num_qubits": 5}},
    {"name": "clops", "config": {"benchmark_name": "CLOPS", "shots": 100}}
  ]
}
```

```bash
mgym suite dispatch suite.json --provider local --device aer_simulator
```

## Results Format

Benchmark results include:

```json
{
  "app_version": "0.3.1",
  "provider": "ibm",
  "device": "ibm_fez",
  "job_type": "Quantum Volume",
  "timestamp": "2025-01-15T12:00:00Z",
  "results": {
    "values": {
      "quantum_volume": 32,
      "heavy_output_probability": 0.68
    },
    "uncertainties": {
      "heavy_output_probability": 0.02
    }
  }
}
```

## Uploading Results

Share results with the community:

```bash
mgym job upload <JOB_ID>
```

Results are uploaded to [unitaryfoundation/metriq-data](https://github.com/unitaryfoundation/metriq-data) and displayed on [metriq.info](https://metriq.info).

## Adding Custom Benchmarks

See [Adding New Benchmarks](../development/adding-benchmarks.md) to contribute new benchmarks.
