Adding a New Benchmark
######################

This guide explains how to integrate a new benchmark into **metriq-gym**. The process involves several steps:

1.  Define the core Python structures (classes and :code:`dataclass` objects).
2.  Create a JSON Schema to validate benchmark parameters.
3.  Provide an example schema for convenience.
4.  Register the new benchmark within the package so it becomes accessible to the rest of the system.

Defining a New Benchmark
************************

1.  **Create a New Python File**

    In the :file:`benchmarks` directory of the :code:`metriq-gym` package, add a new file named
    :file:`benchmarks/<NEW_BENCHMARK>.py` (replace :code:`<NEW_BENCHMARK>` with a descriptive name).

2.  **Implement the Benchmark Class**

    In this new file, define a class that inherits from :code:`Benchmark`. You must override:

    -   :code:`dispatch_handler()`: Houses the logic for preprocessing and dispatching the benchmark job to the quantum device. This may include things like creating the circuits, running those circuits on classical simulators, etc.
    -   :code:`poll_handler()`: Houses the logic for retrieving and processing results from the quantum device or simulator.

3.  **Create the Data Classes**

    Define two :code:`dataclass` objects: one that will contain intermediate data after pre-processing and dispatching
    (inheriting from :code:`BenchmarkData`), and one to hold the benchmark’s output results (inheriting from
    :code:`BenchmarkResult`).

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

Reporting Metrics and Uncertainty
********************************

Benchmarks should surface result metrics through the :class:`BenchmarkResult` subclass. For simple scalar metrics,
you can declare a numeric field (``float``/``int``). If your metric has a meaningful uncertainty, wrap it in
:class:`BenchmarkScore` (``value`` and ``uncertainty``).

The exporter includes two parallel maps under ``results`` in the payload:

- ``results.values`` — metric name → numeric value
- ``results.uncertainties`` — metric name → uncertainty (if any)

Example 1: numeric-only metric
------------------------------

.. code-block:: python

    from dataclasses import dataclass
    from metriq_gym.benchmarks.benchmark import BenchmarkResult

    @dataclass
    class MyResult(BenchmarkResult):
        clops_score: float  # no uncertainty reported; defaults to direction="higher"

Example 2: metric with uncertainty (default: higher-is-better)
--------------------------------------------------------------

.. code-block:: python

    from dataclasses import dataclass
    from metriq_gym.benchmarks.benchmark import BenchmarkResult, BenchmarkScore

    @dataclass
    class MyResult(BenchmarkResult):
        expectation_value: BenchmarkScore

    # Later in poll_handler(...):
    return MyResult(expectation_value=BenchmarkScore(value=0.73, uncertainty=0.04))

Notes
-----

- For plain numeric fields, uncertainties default to ``0.0``.
- If an uncertainty is ill-defined or not measured, you may omit it (treated as ``None``). Do not omit the metric if the value is still meaningful.
- The :class:`BenchmarkResult` object exposes ``values`` and ``uncertainties`` properties and a ``score`` computed field for aggregation.

Defining the Schema
*******************

To standardize and validate the input parameters for each benchmark, **metriq-gym** uses JSON Schema. Add a new file
named :file:`new_benchmark.schema.json` to the :file:`schemas/` directory. Note that this schema file is just an example
and should be modified to fit the specific requirements of your benchmark.

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

1.  **Add to `constants.py`**

    Open the :file:`metriq_gym/constants.py` file and add your new benchmark's name to the :code:`JobType` enumeration. The key (e.g., `NEW_BENCHMARK`) should be uppercase, and the value should be the human-readable string name.

    .. code-block:: python

        # In metriq_gym/constants.py
        from enum import StrEnum

        class JobType(StrEnum):
            NEW_BENCHMARK = "New Benchmark"
            ...

2.  **Add to `registry.py`**

    Open :file:`metriq_gym/registry.py` to map your new benchmark name to its implementation classes and schema.

    First, import your benchmark classes at the top of the file:

    .. code-block:: python

        # In metriq_gym/registry.py
        from metriq_gym.benchmarks.new_benchmark import NewBenchmark, NewBenchmarkData
        ...

    Then, add a new entry to each of the three mapping dictionaries: :code:`BENCHMARK_HANDLERS`, :code:`BENCHMARK_DATA_CLASSES`, and :code:`SCHEMA_MAPPING`.

    .. code-block:: python

        # In metriq_gym/registry.py

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

-   **Testing**: Verify that your benchmark can be successfully dispatched, polled, and completed using an appropriate
    quantum device or simulator.
-   **Documentation**: Update or create any user-facing docs describing how to run or configure this new benchmark.
-   **Maintenance**: Ensure the schema and Python classes remain in sync if input parameters or benchmark logic changes.

With these steps, your new benchmark is fully integrated into **metriq-gym** and ready to be used!
