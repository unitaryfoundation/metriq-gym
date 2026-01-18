# CLOPS (Circuit Layer Operations Per Second)

CLOPS measures the throughput of a quantum system - how many circuit layer operations can be executed per second.

## Overview

CLOPS is IBM's standard benchmark for measuring quantum computer throughput. It accounts for:
- Circuit compilation time
- Job submission latency
- Queue wait time
- Execution time
- Result retrieval

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"CLOPS"` |
| `width` | integer | No | 1000 | Circuit width (number of qubits) |
| `num_layers` | integer | No | 1000 | Number of layers per circuit |
| `num_circuits` | integer | No | 100 | Number of circuits to run |
| `shots` | integer | No | 1000 | Shots per circuit |

### Example Configuration

```json
{
  "benchmark_name": "CLOPS",
  "width": 100,
  "num_layers": 100,
  "num_circuits": 100,
  "shots": 100
}
```

## Usage

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/clops.example.json \
    --provider ibm --device ibm_sherbrooke

# Poll results
mgym job poll <JOB_ID>
```

## Results

| Metric | Description |
|--------|-------------|
| `clops_score` | Circuit Layer Operations Per Second |
| `total_time` | Total execution time in seconds |
| `circuits_per_second` | Raw circuit throughput |

### Example Output

```
CLOPSResult(clops_score=25000, total_time=4.0)
```

## How It Works

1. **Circuit Generation**: Creates `num_circuits` circuits with `num_layers` layers each
2. **Parameterization**: Uses parameterized circuits updated between executions
3. **Timing**: Measures total wall-clock time from first submission to last result
4. **Calculation**: `CLOPS = (circuits × layers × shots) / time`

## CLOPS Formula

```
CLOPS = M × K × S × D / elapsed_time
```

Where:
- `M` = number of circuit templates
- `K` = number of parameter updates
- `S` = number of shots
- `D` = number of QV layers

## Device Requirements

- Works on any device
- Meaningful comparison requires similar circuit structure
- Best for comparing system-level performance

## Interpretation

| CLOPS | Interpretation |
|-------|---------------|
| > 100,000 | Very high throughput (latest IBM systems) |
| 10,000 - 100,000 | High throughput |
| 1,000 - 10,000 | Moderate throughput |
| < 1,000 | Lower throughput (cloud overhead, simulator) |

!!! note
    CLOPS is heavily influenced by classical infrastructure, not just quantum hardware quality.

## References

- [IBM Quantum: What is CLOPS?](https://www.ibm.com/quantum/blog/quantum-volume-and-clops)
- [arXiv:2110.14108](https://arxiv.org/abs/2110.14108) - Volumetric Benchmarking Framework
