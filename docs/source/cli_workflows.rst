CLI Workflows
#############

metriq-gym supports two types of resources: benchmarks and suites.

Single Benchmarks
=================

You can dispatch benchmark jobs by specifying a configuration file for the benchmark you wish to run.

.. code-block:: sh

   mgym job dispatch <BENCHMARK_CONFIG> --provider <PROVIDER> --device <DEVICE>

Refer to ``metriq_gym/schemas/examples/`` for example configuration files.

If running on quantum cloud hardware, jobs are added to a polling queue. Check status with:

.. code-block:: sh

   mgym job poll <METRIQ_GYM_JOB_ID>

Use ``mgym job poll`` with no arguments to list all dispatched jobs and select one interactively.

Export a job result to JSON with:

.. code-block:: sh

   mgym job poll <METRIQ_GYM_JOB_ID> --json

This creates ``<METRIQ_GYM_JOB_ID>.json`` in the current working directory by default.

Using Local Simulators
======================

Dispatch jobs to the local Qiskit Aer simulator:

.. code-block:: sh

   mgym job dispatch metriq_gym/schemas/examples/qml_kernel.example.json --provider local --device aer_simulator

Use ``ibm_<BACKEND>`` to emulate an IBM backend with Aer noise models:

.. code-block:: sh

   mgym job dispatch metriq_gym/schemas/examples/qml_kernel.example.json --provider local --device ibm_<BACKEND>

Poll local jobs the same way:

.. code-block:: sh

   mgym job poll <METRIQ_GYM_JOB_ID>

Benchmark Suites
================

Suites run multiple benchmarks together. Author ``suite.json`` like:

.. code-block:: json

   {
     "name": "test_suite",
     "description": "Just a test suite for the README",
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
      }
     ]
   }

Dispatch and poll the suite:

.. code-block:: sh

   mgym suite dispatch suite.json --provider <PROVIDER> --device <DEVICE>
   mgym suite poll <METRIQ_GYM_SUITE_ID>

Upload to GitHub
================

Publish benchmark results to ``unitaryfoundation/metriq-data``.

Commands:

.. code-block:: sh

   # Single job
   mgym job upload <METRIQ_GYM_JOB_ID>

   # Entire suite
   mgym suite upload <METRIQ_GYM_SUITE_ID>

Defaults:

* Target repo: ``unitaryfoundation/metriq-data`` (override with ``--repo`` or ``MGYM_UPLOAD_REPO``)
* Directory: ``metriq-gym/v<major.minor>/<provider>`` (override with ``--dir`` or ``MGYM_UPLOAD_DIR``)
* Uploads append records to ``results.json``

Estimate Job Resources
======================

Before dispatching, you can approximate the circuit footprint, gate counts, and (for
Quantinuum) HQCs:

.. code-block:: sh

   mgym job estimate metriq_gym/schemas/examples/wit.example.json \
       --provider quantinuum

The command prints aggregated totals and per-circuit statistics. HQCs are calculated
automatically for Quantinuum devices using the published H-series coefficients
(``HQC = 5 + C × (N₁ + 10N₂ + 5Nₘ)/5000``; where ``C`` is the number of shots, matching code usage: ``shots * (N₁ + 10N₂ + 5Nₘ)/5000``); for other providers, only gate counts are shown.
Benchmarks that depend on device topology (e.g. BSEQ, CLOPS, Mirror Circuits, LR-QAOA)
require ``--device`` to be supplied so the estimator can inspect connectivity.

.. code-block:: sh

   mgym job estimate metriq_gym/schemas/examples/wit.example.json \
       --provider ibm --device ibm_fez

Authentication
==============

* Set ``GITHUB_TOKEN`` (or ``GH_TOKEN``). External contributors should fork the data repo first.
* Token docs: https://docs.github.com/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

Credential Management
=====================

Copy ``.env.example`` to ``.env`` and populate provider API tokens before running on hardware.

Viewing Jobs
============

List all recorded jobs:

.. code-block:: sh

   mgym job view

Show details for a specific job:

.. code-block:: sh

   mgym job view <METRIQ_GYM_JOB_ID>

IBM BSEQ Example
================

Run the BSEQ benchmark on ``ibm_sherbrooke``:

.. code-block:: sh

   mgym job dispatch metriq_gym/schemas/examples/bseq.example.json --provider ibm --device ibm_sherbrooke

Sample dispatch output:

.. code-block:: text

   INFO - Starting job dispatch...
   INFO - Dispatching BSEQ benchmark from metriq_gym/schemas/examples/bseq.example.json on ibm_sherbrooke...
   INFO - Job dispatched with ID: 93a06a18-41d8-475a-a030-339fbf3accb9

Check queued jobs:

.. code-block:: text

   +--------------------------------------+------------+----------------+-----------------------------+
   | Metriq-gym Job Id                    | Provider   | Device         | Type           |
   +======================================+============+================+=============================+
   | 93a06a18-41d8-475a-a030-339fbf3accb9 | ibm        | ibm_sherbrooke | BSEQ           |
   +--------------------------------------+------------+----------------+-----------------------------+

Poll the job:

.. code-block:: sh

   mgym job poll 93a06a18-41d8-475a-a030-339fbf3accb9

Example completed result:

.. code-block:: text

   INFO - Polling job...
   BSEQResult(largest_connected_size=100, fraction_connected=0.7874)

If the job is still queued, the CLI reports the current queue position and asks you to try again later.

Use ``mgym job poll`` without arguments to choose a job interactively when the identifier is not handy.
