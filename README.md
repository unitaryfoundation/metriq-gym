<div align="center">
  <img 
    src="https://raw.githubusercontent.com/unitaryfoundation/metriq-gym/refs/heads/main/docs/content/assets/logo.svg" 
    alt="metriq-gym logo" 
    width="450" 
  />
</div>

# metriq-gym

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![PyPI version](https://img.shields.io/pypi/v/metriq-gym.svg?color=blue)](https://pypi.org/project/metriq-gym/)
[![Downloads](https://static.pepy.tech/badge/metriq-gym)](https://pepy.tech/project/metriq-gym)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://unitaryfoundation.github.io/metriq-gym/)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=orange&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.foundation)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/unitaryfoundation/metriq-gym/blob/main/CODE_OF_CONDUCT.md)


metriq-gym is a Python framework for implementing and running standard quantum benchmarks on different quantum devices by different providers.

- _Open_ – Open-source since its inception and fully developed in public.
- _Transparent_ – All benchmark parameters are defined in a schema file and the benchmark code is reviewable by the community.
- _Cross-platform_ – Supports running benchmarks on multiple quantum hardware providers (_integration powered by [qBraid-SDK](https://github.com/qBraid/qBraid)_)
- _User-friendly_ – Provides a simple command-line interface for dispatching, monitoring, and polling benchmark jobs (you can go on with your life while your job waits in the queue).

## Quick Start

Four easy steps to get started with `metriq-gym`!

1. Install `metriq-gym` directly in your Python environment using pip:

   ```sh
   pip install metriq-gym
   ```

2. Download a benchmark configuration file from the `schemas/examples/` directory (this example uses the WIT — Wormhole-inspired teleportation — benchmark)

    ```sh
    curl -O https://raw.githubusercontent.com/unitaryfoundation/metriq-gym/refs/heads/main/metriq_gym/schemas/examples/wit.example.json
    ```

3. Dispatch it to a quantum device or simulator.

    ```sh
    mgym job dispatch wit.example.json -p local -d aer_simulator
    ```
4. Poll the job to get the results.

    ```sh
    mgym job poll latest
    ```

You will see the results of the benchmark printed in your terminal. E.g.
```sh
{'app_version': '0.6.0',
 'job_type': 'WIT',
 'platform': {'device': 'aer_simulator',
              'device_metadata': {'num_qubits': 31,
                                  'simulator': True,
                                  'version': '0.17.2'},
              'provider': 'local'},
 'results': {'expectation_value': {'uncertainty': 0.0006673812593654682,
                                   'value': 0.996337890625},
             'score': {'uncertainty': 0.0006673812593654682,
                       'value': 0.996337890625}},
 'runtime_seconds': 0.009346791077405214,
 'suite_id': None,
 'timestamp': '2026-01-16T15:42:18.173736'}

Results:
  expectation_value: 0.996337890625 ± 0.0006673812593654682
  score: 0.996337890625 ± 0.0006673812593654682
```

Explore more examples in the ready-made JSON schemas under ``metriq_gym/schemas/examples/``.

## Documentation

- [Quickstart Guide](https://unitaryfoundation.github.io/metriq-gym/getting-started/quickstart/) - Get up and running quickly
- [CLI Reference](https://unitaryfoundation.github.io/metriq-gym/cli/overview/) - Command-line workflows and credential setup
- [Provider Configuration](https://unitaryfoundation.github.io/metriq-gym/providers/overview/) - Setup guides for IBM, IonQ, AWS, Azure, Quantinuum, OriginQ
- [Benchmarks](https://unitaryfoundation.github.io/metriq-gym/benchmarks/overview/) - Available benchmarks and configuration
- [Developer Guide](https://unitaryfoundation.github.io/metriq-gym/development/developer-guide/) - Contributing to metriq-gym

## Community

- Join the discussion on [Discord](http://discord.unitary.foundation) (`#metriq` channel)
- Ask questions or share ideas via [GitHub Discussions](https://github.com/unitaryfoundation/metriq-gym/discussions)

## Contributing

Start with [CONTRIBUTING.md](https://github.com/unitaryfoundation/metriq-gym/blob/main/CONTRIBUTING.md) for the workflow checklist, and review the [Developer
Guide](https://unitaryfoundation.github.io/metriq-gym/development/developer-guide/). Issues and pull requests are welcome!

## License

metriq-gym is available under the [Apache License 2.0](https://github.com/unitaryfoundation/metriq-gym/blob/main/LICENSE).
