Adding a New Benchmark
######################

This guide explains how to integrate a new benchmark into **metriq-gym**. The process involves several steps:

1. Define the core Python structures (classes and :code:`dataclass` objects).
2. Create a JSON Schema to validate benchmark parameters.
3. Provide an example schema for convenience.
4. Register the new benchmark within the package so it becomes accessible to the rest of the system.

Defining a New Benchmark
************************

1. **Create a New Python File**

   In the :file:`benchmarks` directory of the :code:`metriq-gym` package, add a new file named
   :file:`benchmarks/<NEW_BENCHMARK>.py` (replace :code:`<NEW_BENCHMARK>` with a descriptive name).

2. **Implement the Benchmark Class**

   In this new file, define a class that inherits from :code:`Benchmark`. You must override:

   - :code:`dispatch_handler()`: Houses the logic for dispatching and scheduling the benchmark job to the quantum device.
   - :code:`poll_handler()`: Houses the logic for retrieving and processing results from the quantum device or simulator.

3. **Create the Data Classes**

   Define two :code:`dataclass` objects: one to store the input parameters (inheriting from :code:`BenchmarkData`), and one
   to hold the benchmark’s output results (inheriting from :code:`BenchmarkResult`).

   Example:

   .. code-block:: python

       from dataclasses import dataclass
       from metriq_gym.benchmarks.benchmark import (
           Benchmark,
           BenchmarkData,
           BenchmarkResult,
       )

       @dataclass
       class NewBenchmarkResult(BenchmarkResult):
           """Stores the result(s) from running New Benchmark."""
           pass

       @dataclass
       class NewBenchmarkData(BenchmarkData):
           """Stores the input parameters or metadata for New Benchmark."""
           pass

       class NewBenchmark(Benchmark):
           """Benchmark class for New Benchmark experiments."""

           def dispatch_handler(
               self,
               device: QuantumDevice
           ) -> NewBenchmarkData:
               # TODO: Implement logic for dispatching the job
               pass

           def poll_handler(
               self,
               job_data: BenchmarkData,
               result_data: list[GateModelResultData]
           ) -> NewBenchmarkResult:
               # TODO: Implement logic for retrieving and processing results
               pass

Defining the Schema
*******************

To standardize and validate the input parameters for each benchmark, **metriq-gym** uses JSON Schema. Add a new
file named :file:`new_benchmark.schema.json` to the :file:`schemas/` directory:

.. code-block:: json

    {
        "$id": "metriq-gym/new_benchmark.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "New Benchmark",
        "description": "Schema definition for New Benchmark, describing its configurable parameters.",
        "type": "object",
        "properties": {
            "benchmark_name": {
                "type": "string",
                "const": "New Benchmark",
                "description": "Name of the benchmark. Must be 'New Benchmark' for this schema."
            },
            "num_qubits": {
                "type": "integer",
                "description": "Number of qubits to be used in the circuit(s).",
                "minimum": 1,
                "examples": [5]
            },
            "shots": {
                "type": "integer",
                "description": "Number of measurement shots (repetitions) to use when running the benchmark.",
                "default": 1000,
                "minimum": 1,
                "examples": [1000]
            },
            "...": {
                "description": "Placeholder for additional properties as needed."
            }
        },
        "required": ["benchmark_name", "num_qubits"]
    }

This schema ensures that any job payload for the new benchmark meets the required format and constraints.

Example Schema
**************

Provide a sample JSON file demonstrating how to supply parameters for this benchmark. Place this file in
:file:`schemas/examples/new_benchmark.example.json`:

.. code-block:: json

    {
        "benchmark_name": "New Benchmark",
        "num_qubits": 5,
        "shots": 1000
    }

This file offers a reference for developers and users on how to structure the JSON payload for your new benchmark.

Registering the New Benchmark
*****************************

1. **Add to job_type.py**

   Open the :file:`metriq_gym/job_type.py` file and register your new benchmark in the :code:`JobType` enumeration:

   .. code-block:: python

       from metriq_gym.job_type import JobType
       from enum import StrEnum

       class JobType(StrEnum):
           NEW_BENCHMARK = "New Benchmark"
           ...

2. **Initialize in benchmarks/__init__.py**

   Within :file:`benchmarks/__init__.py`, import your benchmark classes and add them to the appropriate mappings:

   .. code-block:: python

       from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData
       from metriq_gym.benchmarks.new_benchmark import NewBenchmark, NewBenchmarkData
       ...
       from metriq_gym.job_type import JobType

       BENCHMARK_HANDLERS: dict[JobType, type[Benchmark]] = {
           JobType.NEW_BENCHMARK: NewBenchmark,
           ...
       }

       BENCHMARK_DATA_CLASSES: dict[JobType, type[BenchmarkData]] = {
           JobType.NEW_BENCHMARK: NewBenchmarkData,
           ...
       }

       SCHEMA_MAPPING = {
           JobType.NEW_BENCHMARK: "new_benchmark.schema.json",
           ...
       }

   By doing so, the new benchmark is linked to its job type, data class, and JSON schema.

Final Steps
***********

- **Testing**: Verify that your benchmark can be successfully dispatched, polled, and completed using an appropriate
  quantum device or simulator.
- **Documentation**: Update or create any user-facing docs describing how to run or configure this new benchmark.
- **Maintenance**: Ensure the schema and Python classes remain in sync if input parameters or benchmark logic changes.

With these steps, your new benchmark is fully integrated into **metriq-gym** and ready to be used!
