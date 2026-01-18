# Quickstart

Metriq-Gym provides a command-line interface for running quantum benchmark jobs on simulators and hardware. This guide focuses on the essentials so you can submit your first job quickly.

## Installation

Install the package from PyPI:

```bash
pip install metriq-gym
```

## Running Your First Benchmark

### 1. Download an Example Configuration

Download an example configuration file for the WIT (Wormhole-inspired teleportation) benchmark:

```bash
curl -O https://raw.githubusercontent.com/unitaryfoundation/metriq-gym/main/metriq_gym/schemas/examples/wit.example.json
```

!!! note
    Example configurations for all benchmarks are available in the `metriq_gym/schemas/examples/` directory.

### 2. Dispatch the Benchmark

Run the benchmark on the local Aer simulator:

```bash
mgym job dispatch wit.example.json -p local -d aer_simulator
```

This command dispatches the job and returns a job ID that you'll use to check results.

### 3. Poll for Results

Check the status and retrieve results:

```bash
mgym job poll latest
```

If the job completed, metrics such as expectation values are reported in your terminal.

!!! tip
    Use `mgym job poll` without arguments to choose from recent jobs interactively.

## Configuration Files

Each benchmark is configured via JSON documents. The `metriq_gym/schemas/examples/` directory contains ready-to-run templates for all supported benchmarks. Customize a copy to:

- Switch benchmarks (change `benchmark_name`)
- Adjust qubit counts or shots
- Supply provider-specific options

Example WIT configuration:

```json
{
  "benchmark_name": "WIT",
  "num_qubits": 7,
  "shots": 8192
}
```

## Running on Real Hardware

To run benchmarks on cloud quantum hardware:

1. Set up provider credentials in a `.env` file (see [Provider Configuration](../providers/overview.md))
2. Specify the provider and device:

```bash
mgym job dispatch wit.example.json --provider ibm --device ibm_fez
```

## Next Steps

- [End-to-End Tutorial](tutorial.md) - Complete walkthrough including uploading results
- [CLI Reference](../cli/overview.md) - All available commands
- [Provider Configuration](../providers/overview.md) - Setup guides for each provider
- [Benchmarks](../benchmarks/overview.md) - Available benchmarks and their configurations
