# WIT (Wormhole-Inspired Teleportation)

The WIT benchmark tests quantum teleportation fidelity inspired by traversable wormhole physics.

## Overview

Wormhole-Inspired Teleportation (WIT) implements a quantum teleportation protocol based on the SYK model, which has connections to traversable wormhole physics in AdS/CFT correspondence. The benchmark measures how well a device can perform this specific teleportation task.

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"WIT"` |
| `num_qubits` | integer | No | 6 | Number of qubits (6 or 7 only) |
| `shots` | integer | No | 1000 | Measurement shots |

### Example Configuration

```json
{
  "benchmark_name": "WIT",
  "num_qubits": 7,
  "shots": 8192
}
```

## Usage

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider local --device aer_simulator

# Poll results
mgym job poll <JOB_ID>
```

## Results

| Metric | Description |
|--------|-------------|
| `expectation_value` | Teleportation fidelity (ideal = 1.0) |

### Example Output

```
WITResult(expectation_value=0.9970703125)
```

## How It Works

The WIT protocol:

1. **Preparation**: Create an entangled state between "left" and "right" subsystems
2. **Encoding**: Encode information on the left side
3. **Interaction**: Apply a coupling operation between subsystems
4. **Teleportation**: Information appears on the right side
5. **Measurement**: Verify successful teleportation via expectation value

## Circuit Structure

The WIT circuit uses either 6 or 7 qubits:

- **6 qubits**: Minimal implementation
- **7 qubits**: Extended version with ancilla

## Qubit Selection

| `num_qubits` | Structure |
|--------------|-----------|
| 6 | 3 left + 3 right qubits |
| 7 | 3 left + 3 right + 1 ancilla |

## Interpretation

| Expectation Value | Interpretation |
|------------------|----------------|
| > 0.95 | Excellent teleportation fidelity |
| 0.85 - 0.95 | Good fidelity |
| 0.70 - 0.85 | Moderate fidelity |
| < 0.70 | Poor fidelity - high error rates |

## Ideal vs. Real Performance

| Device Type | Typical Result |
|-------------|---------------|
| Ideal simulator | 1.0 |
| Noisy simulator | 0.95-0.99 |
| Current hardware | 0.70-0.95 |

## Physics Background

The benchmark is based on:
- The Sachdev-Ye-Kitaev (SYK) model
- Traversable wormhole protocols in AdS/CFT
- Quantum teleportation enhanced by scrambling dynamics

The protocol demonstrates that information can be "teleported" through an Einstein-Rosen bridge analog in the quantum circuit.

## References

- [arXiv:2205.14081](https://arxiv.org/abs/2205.14081) - Wormhole-Inspired Teleportation
- [Nature 612, 51-55 (2022)](https://www.nature.com/articles/s41586-022-05424-3) - Traversable wormhole dynamics on a quantum processor
