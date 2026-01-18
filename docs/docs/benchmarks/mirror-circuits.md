# Mirror Circuits

Mirror Circuits benchmark measures circuit fidelity using self-inverting random Clifford circuits.

## Overview

Mirror circuits are constructed to return to the initial state by mirroring the circuit: if the first half applies operations U, the second half applies U†. This allows measurement of total circuit fidelity without needing a reference simulation.

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"Mirror Circuits"` |
| `width` | integer | No | all qubits | Number of qubits |
| `num_layers` | integer | No | 3 | Number of Clifford layers (1-500) |
| `two_qubit_gate_prob` | float | No | 0.5 | Probability of two-qubit gates (0-1) |
| `two_qubit_gate_name` | string | No | `"CNOT"` | Gate type (`CNOT` or `CZ`) |
| `shots` | integer | No | 1000 | Measurement shots |
| `num_circuits` | integer | No | 10 | Number of random circuits (1-1000) |
| `seed` | integer | No | - | Random seed for reproducibility |

### Example Configuration

```json
{
  "benchmark_name": "Mirror Circuits",
  "width": 10,
  "num_layers": 5,
  "two_qubit_gate_prob": 0.5,
  "two_qubit_gate_name": "CNOT",
  "shots": 1000,
  "num_circuits": 10,
  "seed": 42
}
```

## Usage

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/mirror_circuits.example.json \
    --provider ibm --device ibm_sherbrooke

# Poll results
mgym job poll <JOB_ID>
```

## Results

| Metric | Description |
|--------|-------------|
| `success_probability` | Average probability of returning to |0⟩ |
| `polarization` | Normalized success probability |
| `std_error` | Statistical uncertainty |

### Example Output

```
MirrorCircuitsResult(success_probability=0.85, polarization=0.70)
```

## How It Works

1. **Forward Half**: Apply random Clifford layers respecting device connectivity
2. **Inversion**: Compute and apply the inverse of the forward circuit
3. **Measurement**: Measure all qubits in the computational basis
4. **Analysis**: The circuit should return to |0...0⟩ on a perfect device

## Circuit Structure

```
|0⟩ ─[Layer 1]─[Layer 2]─...─[Layer n]─[Layer n†]─...─[Layer 2†]─[Layer 1†]─ M → |0⟩
```

## Parameters Guide

### `num_layers`

Controls circuit depth:
- Low (1-3): Quick connectivity test
- Medium (5-10): Typical benchmark
- High (20+): Stress test for error accumulation

### `two_qubit_gate_prob`

Controls entanglement density:
- 0.3: Sparse two-qubit gates
- 0.5: Balanced (default)
- 0.8: Dense two-qubit gates

## Interpretation

| Success Probability | Polarization | Interpretation |
|--------------------|--------------|----------------|
| > 0.9 | > 0.8 | Excellent fidelity |
| 0.7 - 0.9 | 0.4 - 0.8 | Good fidelity |
| 0.5 - 0.7 | 0.0 - 0.4 | Moderate errors |
| < 0.5 | < 0.0 | High error rate |

## Device Requirements

- Requires device topology for connectivity-aware circuit generation
- Works on any gate-based quantum computer

## References

- [arXiv:2008.11294](https://arxiv.org/abs/2008.11294) - Mirror Circuit Benchmarking
- [Qiskit Device Benchmarking](https://github.com/qiskit-community/qiskit-device-benchmarking)
