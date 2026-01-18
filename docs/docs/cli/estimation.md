# Resource Estimation

Estimate circuit resource requirements before dispatching jobs. This is especially useful for understanding costs on paid hardware like Quantinuum.

## estimate

Approximate the circuit footprint, gate counts, and (for Quantinuum) HQCs before running a job.

```bash
mgym job estimate <config_file> --provider <provider> [--device <device>]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `config_file` | Path to benchmark configuration JSON file |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--provider` | `-p` | Provider name (required) |
| `--device` | `-d` | Device identifier (required for some benchmarks) |

### Examples

```bash
# Basic estimation
mgym job estimate metriq_gym/schemas/examples/wit.example.json \
    --provider quantinuum

# With device (required for topology-dependent benchmarks)
mgym job estimate metriq_gym/schemas/examples/bseq.example.json \
    --provider ibm --device ibm_fez
```

---

## Quantinuum HQC Estimation

For Quantinuum providers, the estimator calculates H-series Quantum Credits (HQCs) using the published formula:

```
HQC = 5 + C * (N1 + 10*N2 + 5*Nm) / 5000
```

Where:
- `C` = number of shots
- `N1` = single-qubit gate count
- `N2` = two-qubit gate count
- `Nm` = measurement count

### Example Output

```
Resource Estimation for WIT benchmark
=====================================

Circuits: 1
Total shots: 8192

Per-circuit statistics:
+----------+--------+----------+----------+-------+
| Circuit  | Qubits | 1Q Gates | 2Q Gates | Meas  |
+==========+========+==========+==========+=======+
| wit_0    | 7      | 42       | 28       | 7     |
+----------+--------+----------+----------+-------+

Aggregated totals:
- Total 1Q gates: 42
- Total 2Q gates: 28
- Total measurements: 7

Quantinuum HQC estimate: 52.4
```

---

## Device-Dependent Benchmarks

Some benchmarks require device topology information for accurate estimation:

| Benchmark | Requires Device |
|-----------|-----------------|
| BSEQ | Yes |
| CLOPS | Yes |
| Mirror Circuits | Yes |
| LR-QAOA | Yes |
| EPLG | No |
| Quantum Volume | No |
| WIT | No |
| QML Kernel | No |

For these benchmarks, supply `--device` so the estimator can inspect connectivity:

```bash
mgym job estimate metriq_gym/schemas/examples/mirror_circuits.example.json \
    --provider ibm --device ibm_sherbrooke
```

---

## Use Cases

### Cost Planning

Before running expensive hardware jobs:

```bash
# Estimate Quantinuum costs
mgym job estimate my_benchmark.json -p quantinuum

# Output shows HQC estimate for budgeting
```

### Benchmark Comparison

Compare resource requirements across benchmarks:

```bash
# Compare EPLG vs Mirror Circuits
mgym job estimate metriq_gym/schemas/examples/eplg.example.json -p ibm -d ibm_fez
mgym job estimate metriq_gym/schemas/examples/mirror_circuits.example.json -p ibm -d ibm_fez
```

### Configuration Tuning

Adjust shots or qubit counts to fit within budget:

```bash
# Original: high shot count
mgym job estimate high_shots.json -p quantinuum

# Modified: reduced shots
mgym job estimate low_shots.json -p quantinuum
```
