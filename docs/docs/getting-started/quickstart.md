# Getting Started

This guide walks you through running your first benchmark, checking results, and optionally uploading them to the Metriq community database.

## Installation

Install Metriq-Gym from PyPI:

```bash
pip install metriq-gym
```

## Running Your First Benchmark

### 1. Download an Example Configuration

Benchmarks are configured via JSON files. Download an example for the WIT (Wormhole-inspired teleportation) benchmark:

```bash
curl -O https://raw.githubusercontent.com/unitaryfoundation/metriq-gym/main/metriq_gym/schemas/examples/wit.example.json
```

!!! note
    Example configurations for all benchmarks are in `metriq_gym/schemas/examples/`.

### 2. Dispatch the Benchmark

Run the benchmark on the local Aer simulator:

```bash
mgym job dispatch wit.example.json -p local -d aer_simulator
```

Output:
```
Starting dispatch on local:aer_simulator...
Dispatching WIT...
Job dispatched with metriq-gym Job ID: 7cb5b2df-e62d-423f-ac22-4bf6739d2ea4
```

### 3. Poll for Results

Check status and retrieve results:

```bash
mgym job poll latest
```

Output:
```
{'app_version': '0.6.0',
 'job_type': 'WIT',
 'platform': {'device': 'aer_simulator', 'provider': 'local'},
 'results': {'expectation_value': {'uncertainty': 0.0007, 'value': 0.996}},
 'timestamp': '2025-01-15T09:54:47.904520'}

Results:
  expectation_value: 0.996 ± 0.0007
```

For WIT, the ideal expectation value is `1.0`. A result of `~0.996` indicates correct circuit execution.

!!! tip
    Use `mgym job poll` without arguments to interactively select from recent jobs.

## Viewing Job Details

List all tracked jobs:

```bash
mgym job view
```

View a specific job:

```bash
mgym job view <JOB_ID>
```

## Running on Real Hardware

To run benchmarks on cloud quantum hardware:

### 1. Set Up Credentials

Create a `.env` file with your provider credentials:

```bash
# IBM Quantum
QISKIT_IBM_TOKEN="your-ibm-token"

# IonQ
IONQ_API_KEY="your-ionq-key"

# See Provider Configuration for all options
```

See [Provider Configuration](../providers/overview.md) for detailed setup instructions.

### 2. Dispatch to Hardware

```bash
mgym job dispatch wit.example.json --provider ibm --device ibm_fez
```

!!! note
    Hardware jobs may be queued. Poll periodically to check status.

## Uploading Results

Contribute your results to the community database on [metriq.info](https://metriq.info).

### 1. Set Up GitHub Token

Create a Personal Access Token at [github.com/settings/tokens](https://github.com/settings/tokens) with `repo` scope, then:

```bash
export GITHUB_TOKEN="your-token-here"
```

### 2. Upload

```bash
mgym job upload <JOB_ID>
```

This creates a pull request to [unitaryfoundation/metriq-data](https://github.com/unitaryfoundation/metriq-data). Once merged, your results appear on [metriq.info](https://metriq.info).

See [GitHub Integration](../uploading/github.md) for advanced options.

## Configuration Files

Benchmark configurations are JSON files specifying the benchmark type and parameters:

```json
{
  "benchmark_name": "WIT",
  "num_qubits": 7,
  "shots": 8192
}
```

Customize by:

- Changing `benchmark_name` for different benchmarks
- Adjusting `num_qubits` or `shots`
- Adding benchmark-specific parameters

See [Benchmarks](../benchmarks/overview.md) for available benchmarks and their parameters.

## Next Steps

- [CLI Reference](../cli/overview.md) - Complete command documentation
- [Provider Configuration](../providers/overview.md) - Setup guides for all providers
- [Benchmarks](../benchmarks/overview.md) - Available benchmarks
- [Python API](tutorial.md) - Using Metriq-Gym programmatically
