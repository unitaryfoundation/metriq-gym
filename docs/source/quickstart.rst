Metriq-Gym Quick Start
######################

Metriq-Gym provides a command-line interface for running quantum benchmark jobs on simulators and hardware.
This guide focuses on the essentials so you can submit your first job quickly.

Quick Start
***********

1. Install the package:

   .. code-block:: sh

      pip install metriq-gym

2. Download an example configuration (using the WIT—Wormhole-inspired teleportation—benchmark):

   .. code-block:: sh

      curl -O https://raw.githubusercontent.com/unitaryfoundation/metriq-gym/main/metriq_gym/schemas/examples/wit.example.json

   .. note::
      If the default branch or file location changes, update the URL above accordingly.
3. Dispatch the benchmark to a local simulator:

   .. code-block:: sh

      mgym job dispatch wit.example.json -p local -d aer_simulator

4. Poll for results:

   .. code-block:: sh

      mgym job poll latest

If the job completed, metrics such as expectation values are reported in your terminal. Use ``mgym job poll`` without
arguments to choose from recent jobs interactively.

Configuration Files
*******************

Each benchmark is configured via JSON documents stored under ``metriq_gym/schemas/``. The ``schemas/examples/``
directory contains ready-to-run templates for all supported benchmarks and suites. Customize a copy to switch
benchmarks, adjust qubit counts, or supply provider-specific options.

Next Steps
**********

- Head to :doc:`cli_workflows` for detailed CLI workflows, including suites, uploads, and credential management.
- Visit :doc:`developer_guide` to contribute code or build releases.
- Browse the :doc:`api` reference for module-level details.
