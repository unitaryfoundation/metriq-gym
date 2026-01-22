# Adding New Benchmarks

This guide explains how to integrate a new benchmark into Metriq-Gym.

## Overview

Adding a benchmark involves:

1. Creating Python classes (Benchmark, Data, Result)
2. Defining a JSON Schema for configuration
3. Providing an example configuration
4. Registering the benchmark in the system

## Step 1: Create the Benchmark Class

Create a new file in `metriq_gym/benchmarks/`:

```python
# metriq_gym/benchmarks/my_benchmark.py

from dataclasses import dataclass
from qbraid.runtime import QuantumDevice, GateModelResultData

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)


@dataclass
class MyBenchmarkResult(BenchmarkResult):
    """Stores the results from running My Benchmark."""

    # Simple numeric metric
    success_rate: float

    # Metric with uncertainty
    fidelity: BenchmarkScore


@dataclass
class MyBenchmarkData(BenchmarkData):
    """Stores intermediate data for My Benchmark."""

    # Data needed between dispatch and poll
    expected_outputs: list[str]
    num_circuits: int


class MyBenchmark(Benchmark):
    """Benchmark implementation for My Benchmark."""

    def dispatch_handler(self, device: QuantumDevice) -> MyBenchmarkData:
        """Create and submit benchmark circuits.

        Args:
            device: The quantum device to run on

        Returns:
            Data needed for result analysis
        """
        # Access configuration via self.config
        num_qubits = self.config.get("num_qubits", 5)
        shots = self.config.get("shots", 1000)

        # Create circuits
        circuits = self._create_circuits(num_qubits)

        # Submit to device (handled by base class)
        # Just return the circuits and metadata

        return MyBenchmarkData(
            expected_outputs=["00000"] * len(circuits),
            num_circuits=len(circuits),
        )

    def poll_handler(
        self,
        job_data: BenchmarkData,
        result_data: list[GateModelResultData]
    ) -> MyBenchmarkResult:
        """Process results and compute metrics.

        Args:
            job_data: Data from dispatch phase
            result_data: Raw measurement results

        Returns:
            Computed benchmark results
        """
        data = job_data  # Cast to MyBenchmarkData

        # Analyze results
        successes = 0
        for i, result in enumerate(result_data):
            counts = result.measurements
            if data.expected_outputs[i] in counts:
                successes += counts[data.expected_outputs[i]]

        total_shots = sum(sum(r.measurements.values()) for r in result_data)
        success_rate = successes / total_shots

        # Calculate fidelity with uncertainty
        fidelity_val = self._calculate_fidelity(result_data)
        fidelity_unc = self._calculate_uncertainty(result_data)

        return MyBenchmarkResult(
            success_rate=success_rate,
            fidelity=BenchmarkScore(value=fidelity_val, uncertainty=fidelity_unc),
        )

    def _create_circuits(self, num_qubits: int):
        """Create benchmark circuits."""
        # Implementation details...
        pass

    def _calculate_fidelity(self, results):
        """Calculate fidelity metric."""
        pass

    def _calculate_uncertainty(self, results):
        """Calculate statistical uncertainty."""
        pass
```

## Step 2: Define the JSON Schema

Create `metriq_gym/schemas/my_benchmark.schema.json`:

```json
{
  "$id": "metriq-gym/my_benchmark.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "My Benchmark",
  "description": "Schema for My Benchmark configuration.",
  "type": "object",
  "properties": {
    "benchmark_name": {
      "type": "string",
      "const": "My Benchmark",
      "description": "Must be 'My Benchmark'"
    },
    "num_qubits": {
      "type": "integer",
      "description": "Number of qubits to use",
      "minimum": 1,
      "examples": [5]
    },
    "shots": {
      "type": "integer",
      "description": "Measurement shots per circuit",
      "default": 1000,
      "minimum": 1,
      "examples": [1000]
    },
    "custom_param": {
      "type": "number",
      "description": "A custom parameter for this benchmark",
      "default": 0.5,
      "minimum": 0.0,
      "maximum": 1.0
    }
  },
  "required": ["benchmark_name", "num_qubits"]
}
```

## Step 3: Create Example Configuration

Create `metriq_gym/schemas/examples/my_benchmark.example.json`:

```json
{
  "benchmark_name": "My Benchmark",
  "num_qubits": 5,
  "shots": 1000,
  "custom_param": 0.5
}
```

## Step 4: Register the Benchmark

### Add to constants.py

```python
# metriq_gym/constants.py

class JobType(StrEnum):
    # ... existing entries ...
    MY_BENCHMARK = "My Benchmark"


SCHEMA_MAPPING = {
    # ... existing entries ...
    JobType.MY_BENCHMARK: "my_benchmark.schema.json",
}
```

### Add to registry.py

```python
# metriq_gym/registry.py

from metriq_gym.benchmarks.my_benchmark import MyBenchmark, MyBenchmarkData

BENCHMARK_HANDLERS: dict[JobType, type[Benchmark]] = {
    # ... existing entries ...
    JobType.MY_BENCHMARK: MyBenchmark,
}

BENCHMARK_DATA_CLASSES: dict[JobType, type[BenchmarkData]] = {
    # ... existing entries ...
    JobType.MY_BENCHMARK: MyBenchmarkData,
}
```

## Result Metrics

### Simple Numeric Metrics

For metrics without uncertainty:

```python
@dataclass
class MyResult(BenchmarkResult):
    score: float  # No uncertainty
```

### Metrics with Uncertainty

For metrics with statistical uncertainty:

```python
@dataclass
class MyResult(BenchmarkResult):
    fidelity: BenchmarkScore  # Has value and uncertainty
```

Usage:
```python
return MyResult(
    fidelity=BenchmarkScore(value=0.95, uncertainty=0.02)
)
```

### Result Export

The exporter automatically creates:

```json
{
  "results": {
    "values": {
      "score": 0.85,
      "fidelity": 0.95
    },
    "uncertainties": {
      "fidelity": 0.02
    }
  }
}
```

## Best Practices

### Configuration

- Use sensible defaults for optional parameters
- Validate parameters in `dispatch_handler`
- Document all parameters in the schema

### Circuit Creation

- Use Qiskit circuits for portability
- Respect device topology when possible
- Keep circuits transpiler-friendly

### Result Analysis

- Calculate meaningful uncertainty estimates
- Handle edge cases (zero shots, failed circuits)
- Return `None` for undefined metrics

### Testing

Create tests in `tests/test_benchmarks.py`:

```python
def test_my_benchmark_dispatch():
    config = {"benchmark_name": "My Benchmark", "num_qubits": 5}
    benchmark = MyBenchmark(config)
    data = benchmark.dispatch_handler(mock_device)
    assert data.num_circuits > 0

def test_my_benchmark_poll():
    # Test result processing
    ...
```

## Checklist

Before submitting your benchmark:

- [ ] Benchmark class implements `dispatch_handler` and `poll_handler`
- [ ] JSON Schema validates all configuration options
- [ ] Example configuration works with local simulator
- [ ] Benchmark registered in `constants.py` and `registry.py`
- [ ] Tests cover dispatch and poll logic
- [ ] Documentation explains benchmark purpose and metrics
