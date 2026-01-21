# API Reference

This page documents the main Python modules in Metriq-Gym.

## Core Modules

### metriq_gym.run

Main runtime module with dispatch, poll, and upload functions.

::: metriq_gym.run
    options:
      show_source: false
      members:
        - dispatch_job
        - poll_job
        - upload_job
        - view_job
        - load_provider
        - setup_device

### metriq_gym.cli

Command-line interface utilities.

::: metriq_gym.cli
    options:
      show_source: false

### metriq_gym.job_manager

Job tracking and persistence.

::: metriq_gym.job_manager
    options:
      show_source: false
      members:
        - JobManager
        - MetriqGymJob

### metriq_gym.schema_validator

JSON Schema validation utilities.

::: metriq_gym.schema_validator
    options:
      show_source: false

### metriq_gym.suite_parser

Benchmark suite parsing.

::: metriq_gym.suite_parser
    options:
      show_source: false

## Key Classes

### JobManager

Manages job lifecycle and persistence.

```python
from metriq_gym.job_manager import JobManager

manager = JobManager()

# Get all jobs
jobs = manager.get_jobs()

# Get specific job
job = manager.get_job("job-id")

# Add a new job
manager.add_job(job)

# Update job status
manager.update_job(job)
```

### MetriqGymJob

Represents a benchmark job.

```python
from metriq_gym.job_manager import MetriqGymJob

job = MetriqGymJob(
    id="unique-id",
    job_type=JobType.QUANTUM_VOLUME,
    params={"num_qubits": 5, "shots": 1000},
    provider_name="ibm",
    device_name="ibm_sherbrooke",
    provider_job_ids=["remote-job-id"],
    dispatch_time="2025-01-15T12:00:00",
    app_version="0.3.1",
)
```

## Benchmark Base Classes

### Benchmark

Base class for all benchmarks.

```python
from metriq_gym.benchmarks.benchmark import Benchmark

class MyBenchmark(Benchmark):
    def dispatch_handler(self, device):
        # Create and submit circuits
        pass

    def poll_handler(self, job_data, result_data):
        # Process results
        pass
```

### BenchmarkResult

Base class for benchmark results.

```python
from dataclasses import dataclass
from metriq_gym.benchmarks.benchmark import BenchmarkResult

@dataclass
class MyResult(BenchmarkResult):
    metric_value: float
```

### BenchmarkScore

Metric with uncertainty.

```python
from metriq_gym.benchmarks.benchmark import BenchmarkScore

score = BenchmarkScore(value=0.95, uncertainty=0.02)
```

## Exporter Classes

### GitHubPRExporter

Exports results to GitHub via pull request.

```python
from metriq_gym.exporters.github_pr_exporter import GitHubPRExporter

exporter = GitHubPRExporter(job, result)
pr_url = exporter.export(
    repo="unitaryfoundation/metriq-data",
    base_branch="main",
    directory="results/",
)
```

### JSONExporter

Exports results to local JSON file.

```python
from metriq_gym.exporters.json_exporter import JSONExporter

exporter = JSONExporter(job, result)
exporter.export(output_path="result.json")
```

## Constants

### JobType

Enum of supported benchmark types.

```python
from metriq_gym.constants import JobType

JobType.QUANTUM_VOLUME  # "Quantum Volume"
JobType.CLOPS           # "CLOPS"
JobType.BSEQ            # "BSEQ"
JobType.WIT             # "WIT"
# ... etc
```

### SCHEMA_MAPPING

Maps JobType to schema files.

```python
from metriq_gym.constants import SCHEMA_MAPPING

schema_file = SCHEMA_MAPPING[JobType.QUANTUM_VOLUME]
# "quantum_volume.schema.json"
```

## Usage Examples

### Dispatch and Poll

```python
from types import SimpleNamespace
from dotenv import load_dotenv
from metriq_gym.run import dispatch_job, poll_job
from metriq_gym.job_manager import JobManager

load_dotenv()
job_manager = JobManager()

# Dispatch
dispatch_config = SimpleNamespace(
    config="metriq_gym/schemas/examples/wit.example.json",
    provider="local",
    device="aer_simulator",
)
dispatch_job(dispatch_config, job_manager)

# Poll
jobs = job_manager.get_jobs()
poll_config = SimpleNamespace(job_id=jobs[-1].id)
poll_job(poll_config, job_manager)
```

### Custom Provider Setup

```python
from metriq_gym.run import load_provider, setup_device

# Load provider
provider = load_provider("ibm")

# Get device
device = setup_device("ibm", "ibm_sherbrooke")

# List available devices
devices = provider.get_devices()
for d in devices:
    print(f"{d.id}: {d.status}")
```

### Working with Results

```python
from metriq_gym.job_manager import JobManager

manager = JobManager()
job = manager.get_job("job-id")

# Access result data
if job.result_data:
    print(job.result_data)
```
