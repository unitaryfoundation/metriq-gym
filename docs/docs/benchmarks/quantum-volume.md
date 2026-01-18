# Quantum Volume

Quantum Volume is a holistic benchmark that measures the largest square circuit a device can reliably execute.

## Overview

Quantum Volume (QV) is IBM's standard metric for overall quantum computer capability. A device achieves QV = 2^n if it can successfully run random n-qubit circuits with depth n. The metric captures:
- Qubit count
- Connectivity
- Gate fidelity
- Measurement fidelity
- Crosstalk

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"Quantum Volume"` |
| `num_qubits` | integer | Yes | - | Number of qubits to test (min: 1) |
| `shots` | integer | No | 1000 | Shots per circuit |
| `trials` | integer | No | 100 | Number of random circuits |
| `confidence_level` | float | No | 0.95 | Statistical confidence (0-1) |

### Example Configuration

```json
{
  "benchmark_name": "Quantum Volume",
  "num_qubits": 5,
  "shots": 1000,
  "trials": 100,
  "confidence_level": 0.95
}
```

## Usage

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/quantum_volume.example.json \
    --provider ibm --device ibm_sherbrooke

# Poll results
mgym job poll <JOB_ID>
```

## Results

| Metric | Description |
|--------|-------------|
| `quantum_volume` | Achieved QV (2^n if successful) |
| `heavy_output_probability` | Fraction of heavy outputs |
| `success` | Whether QV threshold was met |
| `confidence_interval` | Statistical bounds |

### Example Output

```
QuantumVolumeResult(quantum_volume=32, heavy_output_probability=0.68, success=True)
```

## Success Criteria

QV is achieved at level 2^n if:
1. Heavy output probability > 2/3 (66.7%)
2. Statistical confidence ≥ specified level (default 95%)

## How It Works

1. **Circuit Generation**: Create random SU(4) layers on n qubits
2. **Simulation**: Classically compute ideal output distribution
3. **Heavy Outputs**: Identify bitstrings with above-median probability
4. **Execution**: Run circuits on hardware
5. **Analysis**: Count fraction of heavy outputs measured

## Circuit Structure

```
     ┌───────┐┌───────┐     ┌───────┐
q0 ──┤       ├┤       ├ ... ┤       ├── M
     │ SU(4) ││ SU(4) │     │ SU(4) │
q1 ──┤       ├┤       ├ ... ┤       ├── M
     └───────┘└───────┘     └───────┘
```

Each layer has n depth and applies random SU(4) gates.

## Interpretation

| Quantum Volume | Interpretation |
|---------------|----------------|
| 2 | Minimal 1-qubit operation |
| 8-16 | Early quantum advantage regime |
| 32-64 | Current mid-range devices |
| 128-256 | High-end devices (2024-2025) |
| 512+ | Target for error-corrected systems |

## Testing Strategy

Start with lower qubit counts and increase:

```bash
# Test QV = 8 (3 qubits)
mgym job dispatch qv_3.json -p ibm -d ibm_sherbrooke

# Test QV = 16 (4 qubits)
mgym job dispatch qv_4.json -p ibm -d ibm_sherbrooke

# Test QV = 32 (5 qubits)
mgym job dispatch qv_5.json -p ibm -d ibm_sherbrooke
```

## Statistical Requirements

For reliable QV claims:
- **trials**: Minimum 100 (200 recommended for publications)
- **shots**: Minimum 1000 per circuit
- **confidence_level**: 0.95 or higher

## References

- [arXiv:1811.12926](https://arxiv.org/abs/1811.12926) - Validating quantum computers using randomized model circuits
- [IBM Quantum: What is Quantum Volume?](https://www.ibm.com/quantum/blog/quantum-volume)
