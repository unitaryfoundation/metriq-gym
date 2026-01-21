# Benchmarks

Metriq-Gym provides a comprehensive suite of quantum benchmarks to characterize and compare quantum hardware performance.

## Running Benchmarks

### Single Benchmark

```bash
mgym job dispatch metriq_gym/schemas/examples/quantum_volume.example.json \
    --provider local --device aer_simulator
```

### Poll for Results

```bash
mgym job poll <JOB_ID>
```

## Configuration

All benchmarks use JSON configuration files:

- **Schemas**: `metriq_gym/schemas/*.schema.json` - Define parameters, types, and allowed values
- **Examples**: `metriq_gym/schemas/examples/*.example.json` - Ready-to-run configurations

## Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| [Mirror Circuits](#metriq_gym.benchmarks.mirror_circuits) | Tests state fidelity via forward/reverse Clifford layers |
| [EPLG](#metriq_gym.benchmarks.eplg) | Error per layered gate across qubit chains |
| [BSEQ](#metriq_gym.benchmarks.bseq) | Bell state effective qubits via CHSH violation |
| [WIT](#metriq_gym.benchmarks.wit) | Wormhole-inspired teleportation protocol |
| [LR-QAOA](#metriq_gym.benchmarks.lr_qaoa) | Linear-ramp QAOA for Max-Cut optimization |
| [QML Kernel](#metriq_gym.benchmarks.qml_kernel) | Quantum machine learning kernel accuracy |
| [QED-C Benchmarks](#metriq_gym.benchmarks.qedc_benchmarks) | Application-oriented benchmarks (BV, QFT, etc.) |

---

::: metriq_gym.benchmarks.mirror_circuits
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.eplg
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.bseq
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.wit
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.lr_qaoa
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.qml_kernel
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.qedc_benchmarks
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

## Adding Custom Benchmarks

See [Adding New Benchmarks](../development/adding-benchmarks.md) to contribute new benchmarks.
