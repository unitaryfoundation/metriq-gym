# LR-QAOA (Linear Ramp QAOA)

The LR-QAOA benchmark evaluates a device's ability to solve combinatorial optimization problems using the Quantum Approximate Optimization Algorithm with linear parameter schedules.

## Overview

Linear Ramp QAOA (LR-QAOA) applies QAOA to weighted MaxCut problems with linearly increasing parameters. This removes the need for classical optimization, making it a fair hardware benchmark that tests quantum operation quality rather than hybrid optimizer performance.

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"Linear Ramp QAOA"` |
| `graph_type` | string | Yes | - | Graph topology: `"1D"`, `"NL"`, or `"FC"` |
| `num_qubits` | integer | Yes | - | Number of qubits (min: 2) |
| `qaoa_layers` | array[int] | No | `[3,5,7,10,15,20]` | QAOA circuit depths |
| `delta_beta` | float | No | 0.3 | Beta parameter slope (0-2) |
| `delta_gamma` | float | No | 0.3 | Gamma parameter slope (0-2) |
| `shots` | integer | No | 1000 | Shots per circuit |
| `trials` | integer | No | 3 | Circuits per configuration |
| `num_random_trials` | integer | No | 25 | Random baseline circuits |
| `confidence_level` | float | No | 0.995 | Statistical confidence |
| `seed` | integer | No | 123 | Random seed |

### Example Configurations

**1D Chain Graph**:
```json
{
  "benchmark_name": "Linear Ramp QAOA",
  "graph_type": "1D",
  "num_qubits": 10,
  "qaoa_layers": [3, 5, 7],
  "shots": 1000
}
```

**Native Layout (Device Topology)**:
```json
{
  "benchmark_name": "Linear Ramp QAOA",
  "graph_type": "NL",
  "num_qubits": 20,
  "qaoa_layers": [3, 5, 10],
  "shots": 1000
}
```

**Fully Connected**:
```json
{
  "benchmark_name": "Linear Ramp QAOA",
  "graph_type": "FC",
  "num_qubits": 6,
  "qaoa_layers": [3, 5],
  "shots": 1000
}
```

## Usage

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/lr_qaoa_1d_chain.example.json \
    --provider ibm --device ibm_sherbrooke

# Poll results
mgym job poll <JOB_ID>
```

## Results

| Metric | Description |
|--------|-------------|
| `approximation_ratio` | Solution quality vs optimal |
| `success_probability` | Probability of finding good solutions |
| `layers_passed` | Maximum layers where QAOA outperforms random |

### Example Output

```
LRQAOAResult(approximation_ratio=0.85, layers_passed=7)
```

## Graph Types

### 1D Chain (`"1D"`)
- Linear chain of qubits
- Each qubit connected to neighbors
- Easiest to implement on most devices

### Native Layout (`"NL"`)
- Uses device's actual connectivity
- Most realistic benchmark
- Requires device topology information

### Fully Connected (`"FC"`)
- Every qubit connected to every other
- Requires SWAP routing on real devices
- Tests both gates and routing

## Linear Ramp Parameters

Parameters increase linearly with layer index p:

```
beta_p = delta_beta * (p + 0.5)
gamma_p = delta_gamma * (p + 0.5)
```

## Success Criteria

LR-QAOA "passes" at layer p if:
- The approximation ratio exceeds random guessing
- With statistical confidence ≥ `confidence_level`

## Interpretation

| Approximation Ratio | Interpretation |
|--------------------|----------------|
| > 0.9 | Near-optimal solutions |
| 0.8 - 0.9 | Good solution quality |
| 0.7 - 0.8 | Acceptable for some applications |
| < 0.7 | Poor - noise dominates |

| Layers Passed | Interpretation |
|--------------|----------------|
| > 15 | Excellent coherence |
| 10 - 15 | Good coherence |
| 5 - 10 | Moderate coherence |
| < 5 | Short coherence times |

## Device Requirements

- `"NL"` and `"1D"` work best on fixed-topology devices
- `"FC"` requires SWAP networks (increased depth)
- Use `--device` flag for topology-dependent configurations

## References

- [arXiv:1411.4028](https://arxiv.org/abs/1411.4028) - A Quantum Approximate Optimization Algorithm
- [arXiv:2004.01372](https://arxiv.org/abs/2004.01372) - Fixed-angle QAOA
