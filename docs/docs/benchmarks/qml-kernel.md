# QML Kernel

The QML Kernel benchmark measures a device's ability to estimate quantum kernel functions for machine learning applications.

## Overview

Quantum kernels are a key primitive in quantum machine learning (QML). This benchmark evaluates a device's accuracy in computing kernel matrix elements, which are used in support vector machines and other kernel-based algorithms.

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"QML Kernel"` |
| `num_qubits` | integer | Yes | - | Number of qubits (min: 2) |
| `shots` | integer | No | 1000 | Measurement shots |

### Example Configuration

```json
{
  "benchmark_name": "QML Kernel",
  "num_qubits": 10,
  "shots": 1000
}
```

## Usage

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/qml_kernel.example.json \
    --provider local --device aer_simulator

# Poll results
mgym job poll <JOB_ID>
```

## Results

| Metric | Description |
|--------|-------------|
| `kernel_accuracy` | Accuracy of kernel estimation |
| `fidelity` | State preparation fidelity |

### Example Output

```
QMLKernelResult(kernel_accuracy=0.95, fidelity=0.98)
```

## How It Works

1. **Feature Map**: Encode classical data into quantum states using a feature map circuit
2. **Kernel Computation**: Measure overlap between encoded states
3. **Comparison**: Compare hardware results to ideal kernel values
4. **Scoring**: Calculate accuracy metrics

## Kernel Formula

The quantum kernel is computed as:

```
K(x, x') = |⟨φ(x)|φ(x')⟩|²
```

Where φ(x) is the quantum feature map encoding classical data x.

## Feature Maps

The benchmark uses a ZZ feature map by default:

```
|0⟩ ─ H ─ U(x) ─ ZZ(x) ─ ... ─ M
```

## Interpretation

| Kernel Accuracy | Interpretation |
|----------------|----------------|
| > 0.95 | Excellent for QML applications |
| 0.85 - 0.95 | Good accuracy |
| 0.70 - 0.85 | Moderate - may impact ML performance |
| < 0.70 | Poor - consider error mitigation |

## Use Cases

This benchmark is relevant for:
- Quantum Support Vector Machines (QSVM)
- Quantum kernel regression
- Quantum feature space analysis
- Comparing devices for QML tasks

## Scaling Considerations

| Qubits | Feature Dimension | Typical Circuits |
|--------|-------------------|-----------------|
| 4 | 16 | ~10 |
| 8 | 256 | ~50 |
| 12 | 4096 | ~200 |

!!! warning
    Kernel matrix computation scales quadratically with dataset size. Use modest qubit counts for initial testing.

## References

- [arXiv:1804.11326](https://arxiv.org/abs/1804.11326) - Quantum Machine Learning
- [arXiv:2101.11020](https://arxiv.org/abs/2101.11020) - Power of Quantum Kernels
