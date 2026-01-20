# End-to-End Tutorial

This tutorial provides a comprehensive walkthrough of using Metriq-Gym, from dispatching benchmarks to uploading results to the [Metriq platform](https://metriq.info).

## Prerequisites

Before starting, ensure you have:

- Metriq-Gym installed (`pip install metriq-gym`)
- A `.env` file with provider credentials (see [Provider Configuration](../providers/overview.md))

## Setup

First, load environment variables in your Python session:

```python
from types import SimpleNamespace
from dotenv import load_dotenv
from metriq_gym.run import dispatch_job, poll_job, view_job, upload_job
from metriq_gym.job_manager import JobManager

# Load environment variables from .env file
load_dotenv()
```

## Job Handling

Metriq-Gym provides a simple API to dispatch jobs to quantum devices and poll/view their status.

### Dispatching a Job

Use the Python interface to dispatch a job to a quantum device or local simulator:

```python
job_manager = JobManager()
benchmark_path = "metriq_gym/schemas/examples/wit.example.json"
provider_name = "local"
device_name = "aer_simulator"

dispatch_config = SimpleNamespace(
    config=benchmark_path,
    provider=provider_name,
    device=device_name,
)

dispatch_job(dispatch_config, job_manager)
```

Output:
```
Starting dispatch on local:aer_simulator...
Dispatching WIT...
Job dispatched with metriq-gym Job ID: 7cb5b2df-e62d-423f-ac22-4bf6739d2ea4
```

Dispatching creates a local `.jsonl` file that tracks all job metadata.

### Viewing Job Details

Check job metadata before polling:

```python
jobs = job_manager.get_jobs()
job_id_to_poll = jobs[-1].id

job_config = SimpleNamespace(job_id=job_id_to_poll)
view_job(job_config, job_manager)
```

This displays a table with job details including provider, device, parameters, and dispatch time.

### Polling for Results

Poll the job to retrieve results:

```python
poll_job(job_config, job_manager)
```

Output:
```python
{
    'app_version': '0.3.1',
    'device': 'aer_simulator',
    'job_type': 'WIT',
    'provider': 'local',
    'results': {'expectation_value': 0.9970703125},
    'suite_id': None,
    'timestamp': '2025-01-15T09:54:47.904520'
}
```

For the WIT benchmark, the ideal expectation value is `1.0`. The simulated result of `~0.997` is very close, indicating correct circuit execution.

## Uploading Results

After polling your job and verifying the results, you can upload them to the [Metriq platform](https://metriq.info) via GitHub.

### Prerequisites

Set up a GitHub token with repository permissions:

```bash
export GITHUB_TOKEN="your-token-here"
```

!!! tip "Creating a GitHub Token"
    Create a Personal Access Token at [github.com/settings/tokens](https://github.com/settings/tokens).
    For fine-grained tokens, grant **Contents** (Read and write) and **Pull requests** (Read and write) permissions.

### Upload via Python

```python
from types import SimpleNamespace
from metriq_gym.run import upload_job

upload_config = SimpleNamespace(
    job_id=job_id_to_poll,
    repo="unitaryfoundation/metriq-data",  # default repository
    dir=None,  # uses default: metriq-gym/v<version>/<provider>/<device>
)

upload_job(upload_config, job_manager)
```

### Upload via CLI

```bash
mgym job upload <METRIQ_GYM_JOB_ID>
```

This creates a pull request to `unitaryfoundation/metriq-data` containing your benchmark results. Once merged, results appear on [metriq.info](https://metriq.info).

### Upload Configuration

You can customize the upload destination:

| Option | Environment Variable | Default |
|--------|---------------------|---------|
| Target repository | `MGYM_UPLOAD_REPO` | `unitaryfoundation/metriq-data` |
| Base branch | `MGYM_UPLOAD_BASE_BRANCH` | `main` |
| Upload directory | `MGYM_UPLOAD_DIR` | `metriq-gym/v<major.minor>/<provider>/<device>` |

## CLI Equivalents

All operations above can also be performed via the CLI:

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider local --device aer_simulator

# View
mgym job view <JOB_ID>

# Poll
mgym job poll <JOB_ID>

# Upload
mgym job upload <JOB_ID>
```

Use `mgym job poll` or `mgym job view` without arguments to interactively select a job.

## Running on Real Hardware

To run benchmarks on IBM Quantum hardware:

```python
dispatch_config = SimpleNamespace(
    config="metriq_gym/schemas/examples/wit.example.json",
    provider="ibm",
    device="ibm_fez",
)

dispatch_job(dispatch_config, job_manager)
```

!!! note
    Hardware jobs may be queued. Poll periodically to check status and retrieve results when complete.

## Next Steps

- [CLI Reference](../cli/overview.md) - Detailed command documentation
- [Provider Configuration](../providers/overview.md) - Setup guides for all providers
- [Benchmarks](../benchmarks/overview.md) - Available benchmarks
- [Metriq Platform](metriq-platform.md) - How results appear on metriq.info
