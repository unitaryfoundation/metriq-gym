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

---

::: metriq_gym.benchmarks.quantum_volume
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

::: metriq_gym.benchmarks.clops
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

::: metriq_gym.benchmarks.mirror_circuits
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

::: metriq_gym.benchmarks.eplg
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

::: metriq_gym.benchmarks.bseq
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

::: metriq_gym.benchmarks.wit
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

::: metriq_gym.benchmarks.lr_qaoa
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

::: metriq_gym.benchmarks.qml_kernel
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

::: metriq_gym.benchmarks.qedc_benchmarks
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false

---

## Adding Custom Benchmarks

See [Adding New Benchmarks](../development/adding-benchmarks.md) to contribute new benchmarks.
