<div align="center">
  <img src="assets/logo.svg" alt="metriq-gym logo" width="450" />
</div>

# Metriq-Gym

**Metriq-Gym** is an open-source framework for running standardized quantum benchmarks across different quantum hardware providers. It provides a unified CLI and Python interface to dispatch, monitor, and analyze benchmark jobs on simulators and real quantum devices.

## What is Metriq?

Metriq-Gym is part of the [Metriq](https://metriq.info) ecosystem, a community-driven platform for tracking quantum computing progress. Results collected with Metriq-Gym can be uploaded to the [Metriq platform](https://metriq.info) to contribute to the global benchmarking database.

## Key Features

- **Multi-provider support**: Run benchmarks on IBM Quantum, IonQ, AWS Braket, Azure Quantum, Quantinuum, OriginQ, and local simulators
- **Standardized benchmarks**: 12+ benchmark implementations including Quantum Volume, CLOPS, Mirror Circuits, and more
- **Simple CLI**: Dispatch jobs with a single command, poll for results, and upload to the community
- **Extensible**: Add custom benchmarks and providers with a clean plugin architecture

## Installation

Install Metriq-Gym from PyPI:

```bash
pip install metriq-gym
```

## Quick Example

Run a benchmark on the local Aer simulator:

```bash
# Download an example configuration
curl -O https://raw.githubusercontent.com/unitaryfoundation/metriq-gym/main/metriq_gym/schemas/examples/wit.example.json

# Dispatch the benchmark
mgym job dispatch wit.example.json -p local -d aer_simulator

# Poll for results
mgym job poll latest
```

## Documentation

- [Quickstart](getting-started/quickstart.md) - Get running in minutes
- [End-to-End Tutorial](getting-started/tutorial.md) - Complete walkthrough including result upload
- [CLI Reference](cli/overview.md) - All available commands
- [Provider Configuration](providers/overview.md) - Setup guides for each provider
- [Benchmarks](benchmarks/overview.md) - Available benchmarks and their configurations
- [Developer Guide](development/developer-guide.md) - Contributing to Metriq-Gym

## Links

- [GitHub Repository](https://github.com/unitaryfoundation/metriq-gym)
- [Metriq Platform](https://metriq.info)
- [Issue Tracker](https://github.com/unitaryfoundation/metriq-gym/issues)
