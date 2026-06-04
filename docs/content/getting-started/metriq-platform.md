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
- Metric values and, when available, uncertainties
- Timestamp and software version

### Historical Tracking

The platform tracks results over time, allowing you to see:

- How device performance changes
- Calibration effects
- Long-term trends

## Data Format

Each uploaded file in metriq-data is a JSON array of result records. A single-job upload
usually contains one record; suite uploads contain several.

```json
[
  {
    "app_version": "0.6.0",
    "timestamp": "2026-01-16T15:42:18.173736",
    "suite_id": null,
    "job_type": "BSEQ",
    "results": {
      "largest_connected_size": 31,
      "fraction_connected": 1.0,
      "score": {
        "value": 31.0,
        "uncertainty": null
      }
    },
    "platform": {
      "provider": "dulwich",
      "device": "dulwich_31q",
      "device_metadata": {
        "num_qubits": 31,
        "simulator": true,
        "version": "0.17.2",
        "calibration": {
          "avg_t1_s": 0.000121,
          "avg_t2_s": 0.000096,
          "avg_readout_error": 0.021,
          "avg_1q_gate_error": 0.0004,
          "avg_2q_gate_error": 0.008,
          "last_update_date": "2026-01-16T15:30:00+00:00"
        }
      }
    },
    "params": {
      "benchmark_name": "BSEQ",
      "shots": 1000
    }
  }
]
```

In this example, `largest_connected_size` and `fraction_connected` are plain numeric
metrics, while `score` is stored as a `{value, uncertainty}` object. Benchmarks that
report uncertainty on a metric use that same object shape for the metric itself.

### Key Fields

| Field | Description |
|-------|-------------|
| `app_version` | `metriq-gym` version that generated the record |
| `timestamp` | ISO 8601 dispatch timestamp recorded by `metriq-gym` |
| `suite_id` | Nullable suite identifier; `null` for single-job uploads |
| `job_type` | Benchmark name, such as `BSEQ` or `WIT` |
| `results` | Benchmark outputs. Metrics may be plain numbers or `{value, uncertainty}` objects; `results.score` is the summary score when the benchmark defines one |
| `platform.provider` | Provider identifier used for the run |
| `platform.device` | Device or backend identifier used for the run |
| `platform.device_metadata` | Optional normalized metadata such as `num_qubits`, `simulator`, backend `version`, and vendor calibration summaries when available |
| `params` | Validated benchmark configuration used to run the job |

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

## Community and Support

- **Report issues**: [metriq-gym issues](https://github.com/unitaryfoundation/metriq-gym/issues)
- **Data questions**: [metriq-data discussions](https://github.com/unitaryfoundation/metriq-data/discussions)
- **Platform feedback**: Contact the Unitary Foundation team

## Related Links

- [Metriq Platform](https://metriq.info)
- [metriq-data Repository](https://github.com/unitaryfoundation/metriq-data)
- [Unitary Foundation](https://unitary.foundation)

