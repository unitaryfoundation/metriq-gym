# EPLG (Error Per Layered Gate)

EPLG measures the error rate per layer of two-qubit gates using randomized benchmarking techniques.

## Overview

Error Per Layered Gate (EPLG) characterizes the layer fidelity of a quantum device by running randomized benchmarking on linear qubit chains. It provides a scalable measure of how errors accumulate with circuit depth.

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"EPLG"` |
| `num_qubits_in_chain` | integer | Yes | - | Number of qubits in the chain (min: 4) |
| `lengths` | array[int] | No | `[2, 4, 8, 16]` | Circuit depths for benchmarking |
| `num_samples` | integer | No | 3 | Random circuits per depth |
| `shots` | integer | No | 100 | Shots per circuit |
| `seed` | integer | No | 12345 | Random seed for reproducibility |
| `two_qubit_gate` | string | No | `"cz"` | Two-qubit gate type (`cz`, `cx`, `ecr`) |
| `one_qubit_basis_gates` | array[str] | No | `["rz", "rx", "x"]` | Single-qubit basis gates |
| `decompose_clifford_ops` | boolean | No | false | Decompose Clifford operations |

### Example Configuration

```json
{
  "benchmark_name": "EPLG",
  "num_qubits_in_chain": 10,
  "lengths": [2, 4, 8, 16],
  "num_samples": 3,
  "shots": 100,
  "two_qubit_gate": "cz"
}
```

## Usage

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/eplg.example.json \
    --provider ibm --device ibm_sherbrooke

# Poll results
mgym job poll <JOB_ID>
```

## Results

| Metric | Description |
|--------|-------------|
| `eplg` | Error Per Layered Gate value |
| `layer_fidelity` | Fidelity of a single layer |
| `fit_quality` | Goodness of exponential fit |

### Example Output

```
EPLGResult(eplg=0.015, layer_fidelity=0.985)
```

This indicates a 1.5% error per layer, with 98.5% layer fidelity.

## How It Works

1. **Chain Selection**: A linear chain of qubits is selected on the device
2. **Layer Construction**: Each layer applies random Clifford gates followed by two-qubit gates
3. **Inversion**: An inverse layer is appended to return to the initial state
4. **Depth Sweep**: Circuits are run at multiple depths
5. **Fitting**: Survival probability is fit to an exponential decay

## Gate Selection

Choose gates native to your device:

| Device Type | Recommended `two_qubit_gate` |
|-------------|------------------------------|
| IBM Eagle/Heron | `ecr` or `cz` |
| Google | `cz` |
| Rigetti | `cz` |
| IonQ/Quantinuum | `cx` |

## Interpretation

| EPLG | Layer Fidelity | Interpretation |
|------|---------------|----------------|
| < 0.01 | > 99% | Excellent gate quality |
| 0.01 - 0.03 | 97-99% | Good for most applications |
| 0.03 - 0.10 | 90-97% | Error correction recommended |
| > 0.10 | < 90% | High error rate |

## Provider Notes

For some providers that don't support Qiskit Clifford primitives, enable decomposition:

```json
{
  "decompose_clifford_ops": true
}
```

## References

- [Layer Fidelity Benchmarking](https://arxiv.org/abs/2311.05933)
- [Qiskit Experiments: Layer Fidelity](https://qiskit-community.github.io/qiskit-experiments/)
