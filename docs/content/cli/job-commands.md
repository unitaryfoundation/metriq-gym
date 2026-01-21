# Job Commands

Commands for dispatching, monitoring, and managing individual benchmark jobs.

## estimate

Estimate circuit resource requirements before dispatching jobs. This is especially useful for understanding costs on paid hardware like Quantinuum.

```bash
mgym job estimate <config_file> --provider <provider> [--device <device>]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `config_file` | Path to benchmark configuration JSON file |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--provider` | `-p` | Provider name (required) |
| `--device` | `-d` | Device identifier (required for some benchmarks) |

### Examples

```bash
# Basic estimation
mgym job estimate metriq_gym/schemas/examples/wit.example.json \
    --provider quantinuum

# With device (required for topology-dependent benchmarks)
mgym job estimate metriq_gym/schemas/examples/bseq.example.json \
    --provider ibm --device ibm_fez
```

### Quantinuum HQC Estimation

For Quantinuum providers, the estimator calculates H-series Quantum Credits (HQCs) using the published formula:

```
HQC = 5 + C * (N1 + 10*N2 + 5*Nm) / 5000
```

Where:

- `C` = number of shots
- `N1` = single-qubit gate count
- `N2` = two-qubit gate count
- `Nm` = measurement count

### Device-Dependent Benchmarks

Some benchmarks require device topology information for accurate estimation:

| Benchmark | Requires Device |
|-----------|-----------------|
| BSEQ | Yes |
| CLOPS | Yes |
| Mirror Circuits | Yes |
| LR-QAOA | Yes |
| EPLG | No |
| Quantum Volume | No |
| WIT | No |
| QML Kernel | No |

For these benchmarks, supply `--device` so the estimator can inspect connectivity:

```bash
mgym job estimate metriq_gym/schemas/examples/mirror_circuits.example.json \
    --provider ibm --device ibm_sherbrooke
```

---

## dispatch

Dispatch a benchmark job to a quantum device or simulator.

```bash
mgym job dispatch <config_file> --provider <provider> --device <device>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `config_file` | Path to benchmark configuration JSON file |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--provider` | `-p` | Provider name (required) |
| `--device` | `-d` | Device identifier (required) |

### Examples

```bash
# Run on local simulator
mgym job dispatch wit.example.json -p local -d aer_simulator

# Run on IBM hardware
mgym job dispatch wit.example.json -p ibm -d ibm_fez

# Run with noise model simulation
mgym job dispatch qml_kernel.example.json -p local -d ibm_sherbrooke
```

### Output

```
Starting dispatch on ibm:ibm_sherbrooke...
Dispatching BSEQ benchmark from bseq.example.json on ibm_sherbrooke...
Job dispatched with ID: 93a06a18-41d8-475a-a030-339fbf3accb9
```

---

## poll

Poll job status and retrieve results when complete.

```bash
mgym job poll [job_id]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `job_id` | Metriq-Gym job ID (optional; interactive if omitted) |

### Options

| Option | Description |
|--------|-------------|
| `--json` | Export results to JSON file |
| `latest` | Poll the most recently dispatched job |

### Examples

```bash
# Poll specific job
mgym job poll 93a06a18-41d8-475a-a030-339fbf3accb9

# Poll latest job
mgym job poll latest

# Interactive selection
mgym job poll

# Export to JSON
mgym job poll 93a06a18-41d8-475a-a030-339fbf3accb9 --json
```

### Output

When complete:
```
Polling job...
BSEQResult(largest_connected_size=100, fraction_connected=0.7874)
```

When queued:
```
Job is queued at position 5. Please try again later.
```

---

## view

View job details and metadata.

```bash
mgym job view [job_id]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `job_id` | Metriq-Gym job ID (optional; lists all if omitted) |

### Examples

```bash
# View all jobs
mgym job view

# View specific job
mgym job view 93a06a18-41d8-475a-a030-339fbf3accb9
```

### Output (all jobs)

```
+--------------------------------------+------------+----------------+------------+----------+
| Metriq-gym Job Id                    | Provider   | Device         | Type       | # Qubits |
+======================================+============+================+============+==========+
| 93a06a18-41d8-475a-a030-339fbf3accb9 | ibm        | ibm_sherbrooke | BSEQ       | 127      |
| 7cb5b2df-e62d-423f-ac22-4bf6739d2ea4 | local      | aer_simulator  | WIT        | 7        |
+--------------------------------------+------------+----------------+------------+----------+
```

### Output (specific job)

```
+-----------------+----------------------------------------------------------+
| suite_id        |                                                          |
| id              | 93a06a18-41d8-475a-a030-339fbf3accb9                     |
| job_type        | BSEQ                                                     |
| params          | {'benchmark_name': 'BSEQ', 'shots': 10}                  |
| provider_name   | ibm                                                      |
| device_name     | ibm_sherbrooke                                           |
| dispatch_time   | 2025-01-15T10:30:45.123456                               |
| app_version     | 0.3.1                                                    |
+-----------------+----------------------------------------------------------+
```

---

## upload

Upload job results to GitHub via pull request.

```bash
mgym job upload <job_id>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `job_id` | Metriq-Gym job ID (required) |

### Options

| Option | Environment Variable | Default |
|--------|---------------------|---------|
| `--repo` | `MGYM_UPLOAD_REPO` | `unitaryfoundation/metriq-data` |
| `--dir` | `MGYM_UPLOAD_DIR` | `metriq-gym/v<version>/<provider>/<device>` |

### Prerequisites

Set a GitHub token with repo permissions:

```bash
export GITHUB_TOKEN="your-token-here"
```

### Examples

```bash
# Upload with defaults
mgym job upload 93a06a18-41d8-475a-a030-339fbf3accb9

# Upload to custom repository
mgym job upload 93a06a18-41d8-475a-a030-339fbf3accb9 --repo myorg/my-data
```

### Output

The command creates a pull request and returns the PR URL. Files are named using the pattern `<timestamp>_<benchmark>_<hash>.json`.

---

## delete

Delete a job from the local database.

```bash
mgym job delete <job_id>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `job_id` | Metriq-Gym job ID (required) |

!!! note
    This only removes the job from the local tracking database. It does not cancel jobs running on quantum hardware.
