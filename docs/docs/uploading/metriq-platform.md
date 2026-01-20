# Result Format and Display

This page documents how Metriq-Gym results are formatted and displayed on the Metriq platform.

For an overview of the Metriq ecosystem, see [The Metriq Ecosystem](../ecosystem.md).

## Data Format

Results uploaded to [metriq-data](https://github.com/unitaryfoundation/metriq-data) follow a standardized JSON format:

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

### Field Reference

| Field | Description |
|-------|-------------|
| `app_version` | Metriq-Gym version used |
| `timestamp` | When the benchmark was run (ISO 8601) |
| `platform.provider` | Hardware provider identifier |
| `platform.device` | Specific device name |
| `job_type` | Benchmark type |
| `results.values` | Computed metric values |
| `results.uncertainties` | Statistical uncertainties (where applicable) |
| `params` | Full benchmark configuration |

## Directory Structure

Results are organized in metriq-data by version, provider, and device:

```
metriq-data/
└── metriq-gym/
    ├── v0.3/
    │   ├── ibm/
    │   │   ├── ibm_sherbrooke/
    │   │   │   └── 2025-01-15T12:00:00_BSEQ_abc123.json
    │   │   └── ibm_fez/
    │   ├── ionq/
    │   └── local/
    └── v0.4/
```

Filenames follow the pattern: `<timestamp>_<benchmark>_<hash>.json`

## How Results Appear on metriq.info

### Result Discovery

The metriq-data repository is periodically scanned for new results. Merged PRs are automatically processed and indexed.

### Display Information

Each result on [metriq.info](https://metriq.info) shows:

- Device and provider information
- Benchmark type and parameters
- Metric values with uncertainties
- Timestamp and software version
- Links to raw data

### Historical Tracking

The platform tracks results over time, enabling:

- Performance trend analysis
- Calibration effect visibility
- Cross-device comparisons

## Accessing Raw Data

### Clone the Repository

```bash
git clone https://github.com/unitaryfoundation/metriq-data.git
```

### Metriq API

The platform provides an API for programmatic access:

```bash
# Get your API key at https://metriq.info/Token
export METRIQ_CLIENT_API_KEY="your-key"
```

See the [Metriq API documentation](https://metriq.info) for details.
