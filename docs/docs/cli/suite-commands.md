# Suite Commands

Commands for working with benchmark suites - collections of multiple benchmarks run together.

## Suite Configuration

Suites are defined in JSON files that specify multiple benchmarks:

```json
{
  "name": "my_test_suite",
  "description": "A collection of benchmarks for testing",
  "benchmarks": [
    {
      "name": "BSEQ",
      "config": {
        "benchmark_name": "BSEQ",
        "shots": 10
      }
    },
    {
      "name": "wit_7_qubits",
      "config": {
        "benchmark_name": "WIT",
        "num_qubits": 7,
        "shots": 1000
      }
    },
    {
      "name": "quantum_volume_5",
      "config": {
        "benchmark_name": "Quantum Volume",
        "num_qubits": 5,
        "shots": 1000
      }
    }
  ]
}
```

### Suite Fields

| Field | Description |
|-------|-------------|
| `name` | Suite identifier |
| `description` | Optional description |
| `benchmarks` | Array of benchmark configurations |

### Benchmark Entry Fields

| Field | Description |
|-------|-------------|
| `name` | Unique name within the suite |
| `config` | Benchmark configuration (same as standalone JSON files) |

---

## dispatch

Dispatch all benchmarks in a suite.

```bash
mgym suite dispatch <suite.json> --provider <provider> --device <device>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `suite.json` | Path to suite configuration file |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--provider` | `-p` | Provider name (required) |
| `--device` | `-d` | Device identifier (required) |

### Example

```bash
mgym suite dispatch my_suite.json -p ibm -d ibm_fez
```

### Output

```
Starting suite dispatch on ibm:ibm_fez...
Dispatching BSEQ...
Dispatching wit_7_qubits...
Dispatching quantum_volume_5...
Suite dispatched with ID: a1b2c3d4-5678-90ab-cdef-1234567890ab
```

---

## poll

Poll suite status and retrieve results for all benchmarks.

```bash
mgym suite poll [suite_id]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `suite_id` | Metriq-Gym suite ID (optional; interactive if omitted) |

### Example

```bash
mgym suite poll a1b2c3d4-5678-90ab-cdef-1234567890ab
```

### Output

Displays status for each benchmark in the suite:

```
Suite: my_test_suite
+------------------+----------+--------------------------+
| Benchmark        | Status   | Result                   |
+==================+==========+==========================+
| BSEQ             | complete | fraction_connected=0.78  |
| wit_7_qubits     | complete | expectation_value=0.95   |
| quantum_volume_5 | queued   | -                        |
+------------------+----------+--------------------------+
```

---

## upload

Upload all suite results to GitHub.

```bash
mgym suite upload <suite_id>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `suite_id` | Metriq-Gym suite ID (required) |

### Options

Same as `mgym job upload` - see [Job Commands](job-commands.md#upload).

### Example

```bash
mgym suite upload a1b2c3d4-5678-90ab-cdef-1234567890ab
```

### Output

Creates a single pull request containing all benchmark results from the suite.

---

## Example Suite Configurations

### Performance Suite

```json
{
  "name": "performance_suite",
  "description": "Standard performance benchmarks",
  "benchmarks": [
    {
      "name": "clops",
      "config": {
        "benchmark_name": "CLOPS",
        "shots": 100
      }
    },
    {
      "name": "qv_5",
      "config": {
        "benchmark_name": "Quantum Volume",
        "num_qubits": 5,
        "shots": 1000
      }
    }
  ]
}
```

### Error Characterization Suite

```json
{
  "name": "error_suite",
  "description": "Error characterization benchmarks",
  "benchmarks": [
    {
      "name": "mirror_10",
      "config": {
        "benchmark_name": "Mirror Circuits",
        "num_qubits": 10,
        "depth": 5,
        "shots": 1000
      }
    },
    {
      "name": "eplg",
      "config": {
        "benchmark_name": "EPLG",
        "num_qubits": 5,
        "shots": 1000
      }
    }
  ]
}
```
