# Metriq Platform

This page explains how your uploaded results appear on the Metriq platform.

!!! tip
    For an overview of Metriq-Gym, see the [home page](../index.md).

## Metriq.info

Visit [metriq.info](https://metriq.info) to explore:

- Historical benchmark results across devices
- Performance trends over time
- Device comparisons
- Community contributions

!!! note
    The Metriq platform is currently available at [beta.metriq.info](https://beta.metriq.info). The URL will transition to [metriq.info](https://metriq.info) at launch.

## How Results Appear

### Result Discovery

The metriq-data repository is periodically scanned for new results. Merged PRs are automatically processed.

### Result Display

Results are displayed with:

- Device and provider information
- Benchmark type and parameters
- Metric values with uncertainties
- Timestamp and software version

### Historical Tracking

The platform tracks results over time, allowing you to see:

- How device performance changes
- Calibration effects
- Long-term trends

## Data Format

Results in metriq-data follow a standardized format:

```json
{
  "app_version": "0.3.1",
  "timestamp": "2025-01-15T12:00:00.000000",
  "platform": {
    "provider": "ibm",
    "device": "ibm_sherbrooke"
  },
  "job_type": "Quantum Volume",
  "results": {
    "values": {
      "quantum_volume": 32,
      "heavy_output_probability": 0.68
    },
    "uncertainties": {
      "heavy_output_probability": 0.02
    }
  },
  "params": {
    "benchmark_name": "Quantum Volume",
    "num_qubits": 5,
    "shots": 1000,
    "trials": 100
  }
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `app_version` | Metriq-Gym version used |
| `timestamp` | When the benchmark was run |
| `platform.provider` | Hardware provider |
| `platform.device` | Specific device |
| `job_type` | Benchmark type |
| `results.values` | Metric values |
| `results.uncertainties` | Statistical uncertainties |
| `params` | Benchmark configuration |

## Contributing Quality Data

To ensure your results are valuable:

### Do

- Run benchmarks with sufficient shots for statistical significance
- Use recommended trial counts for your benchmark
- Include all relevant parameters
- Verify results before uploading

### Don't

- Upload test runs or debugging data
- Artificially cherry-pick results
- Modify result files manually
- Upload duplicate results

## metriq-data Repository

The [unitaryfoundation/metriq-data](https://github.com/unitaryfoundation/metriq-data) repository:

- Is open source and community-driven
- Accepts contributions via pull requests
- Uses automated validation
- Is backed up regularly

### Directory Structure

```
metriq-data/
├── metriq-gym/
│   ├── v0.3/
│   │   ├── ibm/
│   │   │   ├── ibm_sherbrooke/
│   │   │   │   └── 2025-01-15T12:00:00_BSEQ_abc123.json
│   │   │   └── ibm_fez/
│   │   │       └── ...
│   │   ├── ionq/
│   │   └── local/
│   └── v0.4/
└── ...
```

## Accessing Data

### Via GitHub

Browse or clone the repository:

```bash
git clone https://github.com/unitaryfoundation/metriq-data.git
```

### Via Metriq API

The Metriq platform provides an API for programmatic access:

```bash
# Get your API key at https://metriq.info/Token
export METRIQ_CLIENT_API_KEY="your-key"
```

## Community and Support

- **Report issues**: [metriq-gym issues](https://github.com/unitaryfoundation/metriq-gym/issues)
- **Data questions**: [metriq-data discussions](https://github.com/unitaryfoundation/metriq-data/discussions)
- **Platform feedback**: Contact the Unitary Foundation team

## Related Links

- [Metriq Platform](https://metriq.info)
- [metriq-data Repository](https://github.com/unitaryfoundation/metriq-data)
- [Unitary Foundation](https://unitary.foundation)
