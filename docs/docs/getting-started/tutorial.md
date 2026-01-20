# Python API

While the CLI is the primary interface for Metriq-Gym, you can also use the Python API directly for programmatic workflows, integration with notebooks, or custom automation.

!!! tip
    For most users, the [CLI](../cli/overview.md) is simpler and recommended. Use the Python API when you need programmatic control.

## Setup

```python
from types import SimpleNamespace
from dotenv import load_dotenv
from metriq_gym.run import dispatch_job, poll_job, view_job, upload_job
from metriq_gym.job_manager import JobManager

# Load environment variables from .env file
load_dotenv()

# Initialize job manager
job_manager = JobManager()
```

## Dispatching a Job

```python
dispatch_config = SimpleNamespace(
    config="metriq_gym/schemas/examples/wit.example.json",
    provider="local",
    device="aer_simulator",
)

dispatch_job(dispatch_config, job_manager)
```

Output:
```
Starting dispatch on local:aer_simulator...
Dispatching WIT...
Job dispatched with metriq-gym Job ID: 7cb5b2df-e62d-423f-ac22-4bf6739d2ea4
```

## Viewing Jobs

```python
# Get all tracked jobs
jobs = job_manager.get_jobs()

# Get the most recent job
latest_job = jobs[-1]
print(f"Job ID: {latest_job.id}")
print(f"Type: {latest_job.job_type}")
print(f"Device: {latest_job.device_name}")

# View job details
job_config = SimpleNamespace(job_id=latest_job.id)
view_job(job_config, job_manager)
```

## Polling for Results

```python
job_config = SimpleNamespace(job_id=latest_job.id)
result = poll_job(job_config, job_manager)
```

Output:
```python
{
    'app_version': '0.6.0',
    'job_type': 'WIT',
    'platform': {'device': 'aer_simulator', 'provider': 'local'},
    'results': {'expectation_value': {'value': 0.997, 'uncertainty': 0.0007}},
    'timestamp': '2025-01-15T09:54:47.904520'
}
```

## Uploading Results

```python
upload_config = SimpleNamespace(
    job_id=latest_job.id,
    repo="unitaryfoundation/metriq-data",  # default
    dir=None,  # uses default path
)

upload_job(upload_config, job_manager)
```

This creates a pull request to the metriq-data repository.

## Running on Hardware

```python
# IBM Quantum
dispatch_config = SimpleNamespace(
    config="metriq_gym/schemas/examples/wit.example.json",
    provider="ibm",
    device="ibm_fez",
)

dispatch_job(dispatch_config, job_manager)
```

!!! note
    Hardware jobs may be queued. Poll periodically until results are available.

## Working with Providers

```python
from metriq_gym.run import load_provider

# Load a provider
provider = load_provider("ibm")

# List available devices
devices = provider.get_devices()
for device in devices:
    print(f"{device.id}: {device.status}")
```

## Example: Batch Processing

```python
import time

benchmarks = [
    "metriq_gym/schemas/examples/wit.example.json",
    "metriq_gym/schemas/examples/bseq.example.json",
]

# Dispatch all benchmarks
job_ids = []
for config_path in benchmarks:
    dispatch_config = SimpleNamespace(
        config=config_path,
        provider="local",
        device="aer_simulator",
    )
    dispatch_job(dispatch_config, job_manager)
    job_ids.append(job_manager.get_jobs()[-1].id)

# Poll all jobs
for job_id in job_ids:
    job_config = SimpleNamespace(job_id=job_id)
    result = poll_job(job_config, job_manager)
    print(f"Job {job_id}: {result['job_type']}")
```

## CLI Equivalents

| Python | CLI |
|--------|-----|
| `dispatch_job(config, manager)` | `mgym job dispatch <config> -p <provider> -d <device>` |
| `poll_job(config, manager)` | `mgym job poll <job_id>` |
| `view_job(config, manager)` | `mgym job view <job_id>` |
| `upload_job(config, manager)` | `mgym job upload <job_id>` |

## Next Steps

- [API Reference](../development/api-reference.md) - Complete API documentation
- [CLI Reference](../cli/overview.md) - Command-line interface
- [Adding Benchmarks](../development/adding-benchmarks.md) - Create custom benchmarks
