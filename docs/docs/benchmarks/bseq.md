# BSEQ (Binary Sequence)

The BSEQ benchmark measures device connectivity by identifying the largest connected subgraph of functioning qubits.

## Overview

BSEQ characterizes real device connectivity by testing two-qubit gates across the device topology. It identifies which qubit pairs can successfully execute entangling operations, revealing the effective connectivity of the device.

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"BSEQ"` |
| `shots` | integer | No | 1000 | Measurement repetitions |
| `max_colors` | integer | No | all | Maximum graph coloring iterations |

### Example Configuration

```json
{
  "benchmark_name": "BSEQ",
  "shots": 1000
}
```

## Usage

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/bseq.example.json \
    --provider ibm --device ibm_sherbrooke

# Poll results
mgym job poll <JOB_ID>
```

## Results

| Metric | Description |
|--------|-------------|
| `largest_connected_size` | Number of qubits in the largest working subgraph |
| `fraction_connected` | Ratio of connected qubits to total qubits |

### Example Output

```
BSEQResult(largest_connected_size=100, fraction_connected=0.7874)
```

This indicates that 100 qubits form the largest connected component, representing 78.74% of the device.

## How It Works

1. **Graph Coloring**: The device topology is colored to identify parallel executable edges
2. **Bell State Preparation**: For each edge, a Bell state is prepared
3. **Measurement**: Correlation measurements determine if the edge is functional
4. **Connected Components**: Functional edges are analyzed to find the largest connected subgraph

## Device Requirements

- Requires device topology information
- Works best on devices with fixed connectivity (superconducting, neutral atom)
- Fully-connected devices (trapped ion) will show 100% connectivity

## Interpretation

| Result | Interpretation |
|--------|---------------|
| `fraction_connected` > 0.9 | Excellent device health |
| `fraction_connected` 0.7-0.9 | Good, some defective qubits/couplers |
| `fraction_connected` < 0.7 | Significant connectivity issues |

## References

- [Qiskit Device Benchmarking](https://github.com/qiskit-community/qiskit-device-benchmarking)
